//===-- SPIRVLegalizePointerCast.cpp ----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// The LLVM IR has multiple legal patterns we cannot lower to Logical SPIR-V.
// This pass modifies such loads to have an IR we can directly lower to valid
// logical SPIR-V.
// OpenCL can avoid this because they rely on ptrcast, which is not supported
// by logical SPIR-V.
//
// This pass relies on the assign_ptr_type intrinsic to deduce the type of the
// pointed values, must replace all occurences of `ptrcast`. This is why
// unhandled cases are reported as unreachable: we MUST cover all cases.
//
// 1. Loading the first element of an array
//
//        %array = [10 x i32]
//        %value = load i32, ptr %array
//
//    LLVM can skip the GEP instruction, and only request loading the first 4
//    bytes. In logical SPIR-V, we need an OpAccessChain to access the first
//    element. This pass will add a getelementptr instruction before the load.
//
//
// 2. Implicit downcast from load
//
//        %1 = getelementptr <4 x i32>, ptr %vec4, i64 0
//        %2 = load <3 x i32>, ptr %1
//
//    The pointer in the GEP instruction is only used for offset computations,
//    but it doesn't NEED to match the pointed type. OpAccessChain however
//    requires this. Also, LLVM loads define the bitwidth of the load, not the
//    pointer. In this example, we can guess %vec4 is a vec4 thanks to the GEP
//    instruction basetype, but we only want to load the first 3 elements, hence
//    do a partial load. In logical SPIR-V, this is not legal. What we must do
//    is load the full vector (basetype), extract 3 elements, and recombine them
//    to form a 3-element vector.
//
//===----------------------------------------------------------------------===//

#include "SPIRVLegalizePointerCast.h"
#include "SPIRV.h"
#include "SPIRVSubtarget.h"
#include "SPIRVTargetMachine.h"
#include "SPIRVUtils.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/IntrinsicsSPIRV.h"
#include "llvm/Transforms/Utils/Cloning.h"
#include "llvm/Transforms/Utils/LowerMemIntrinsics.h"

using namespace llvm;

namespace {
class SPIRVLegalizePointerCastImpl {

  // Builds the `spv_assign_type` assigning |Ty| to |Value| at the current
  // builder position.
  void buildAssignType(IRBuilder<> &B, Type *Ty, Value *Arg) {
    Value *OfType = PoisonValue::get(Ty);
    CallInst *AssignCI = buildIntrWithMD(Intrinsic::spv_assign_type,
                                         {Arg->getType()}, OfType, Arg, {}, B);
    GR->addAssignPtrTypeInstr(Arg, AssignCI);
  }

  static FixedVectorType *makeVectorFromTotalBits(Type *ElemTy,
                                                  TypeSize TotalBits) {
    unsigned ElemBits = ElemTy->getScalarSizeInBits();
    assert(ElemBits && TotalBits % ElemBits == 0 &&
           "TotalBits must be divisible by element bit size");
    return FixedVectorType::get(ElemTy, TotalBits / ElemBits);
  }

  Value *resizeVectorBitsWithShuffle(IRBuilder<> &B, Value *V,
                                     FixedVectorType *DstTy) {
    auto *SrcTy = cast<FixedVectorType>(V->getType());
    assert(SrcTy->getElementType() == DstTy->getElementType() &&
           "shuffle resize expects identical element types");

    const unsigned NumNeeded = DstTy->getNumElements();
    const unsigned NumSource = SrcTy->getNumElements();

    SmallVector<int> Mask(NumNeeded);
    for (unsigned I = 0; I < NumNeeded; ++I)
      Mask[I] = (I < NumSource) ? static_cast<int>(I) : -1;

    Value *Resized = B.CreateShuffleVector(V, V, Mask);
    buildAssignType(B, DstTy, Resized);
    return Resized;
  }

  // Loads parts of the vector of type |SourceType| from the pointer |Source|
  // and create a new vector of type |TargetType|. |TargetType| must be a vector
  // type.
  // Returns the loaded value.
  Value *loadVectorFromVector(IRBuilder<> &B, FixedVectorType *SourceType,
                              FixedVectorType *TargetType, Value *Source) {
    LoadInst *NewLoad = B.CreateLoad(SourceType, Source);
    buildAssignType(B, SourceType, NewLoad);
    Value *AssignValue = NewLoad;
    if (TargetType->getElementType() != SourceType->getElementType()) {
      const DataLayout &DL = B.GetInsertBlock()->getModule()->getDataLayout();
      TypeSize TargetTypeSize = DL.getTypeSizeInBits(TargetType);
      TypeSize SourceTypeSize = DL.getTypeSizeInBits(SourceType);

      Value *BitcastSrcVal = NewLoad;
      FixedVectorType *BitcastSrcTy =
          cast<FixedVectorType>(BitcastSrcVal->getType());
      FixedVectorType *BitcastDstTy = TargetType;

      if (TargetTypeSize != SourceTypeSize) {
        unsigned TargetElemBits =
            TargetType->getElementType()->getScalarSizeInBits();
        if (SourceTypeSize % TargetElemBits == 0) {
          // No Resize needed. Same total bits as source, but use target element
          // type.
          BitcastDstTy = makeVectorFromTotalBits(TargetType->getElementType(),
                                                 SourceTypeSize);
        } else {
          // Resize source to target total bitwidth using source element type.
          BitcastSrcTy = makeVectorFromTotalBits(SourceType->getElementType(),
                                                 TargetTypeSize);
          BitcastSrcVal = resizeVectorBitsWithShuffle(B, NewLoad, BitcastSrcTy);
        }
      }
      AssignValue =
          B.CreateIntrinsic(Intrinsic::spv_bitcast,
                            {BitcastDstTy, BitcastSrcTy}, {BitcastSrcVal});
      buildAssignType(B, BitcastDstTy, AssignValue);
      if (BitcastDstTy == TargetType)
        return AssignValue;
    }

    assert(TargetType->getNumElements() < SourceType->getNumElements());
    SmallVector<int> Mask(/* Size= */ TargetType->getNumElements());
    for (unsigned I = 0; I < TargetType->getNumElements(); ++I)
      Mask[I] = I;
    Value *Output = B.CreateShuffleVector(AssignValue, AssignValue, Mask);
    buildAssignType(B, TargetType, Output);
    return Output;
  }

  // Returns true if |FromTy| has a memory layout compatible with loading or
  // storing |ToTy|.
  bool isCompatibleMemoryLayout(Type *ToTy, Type *FromTy) {
    if (ToTy == FromTy)
      return true;
    auto *SVT = dyn_cast<FixedVectorType>(FromTy);
    auto *DVT = dyn_cast<FixedVectorType>(ToTy);
    if (SVT && DVT)
      return true;
    auto *SAT = dyn_cast<ArrayType>(FromTy);
    if (SAT && DVT) {
      if (SAT->getElementType() == DVT->getElementType())
        return true;
      if (auto *MAT = dyn_cast<FixedVectorType>(SAT->getElementType()))
        if (MAT->getElementType() == DVT->getElementType())
          return true;
    }
    return false;
  }

  // Traverses the aggregate type to find the first sub-type that matches
  // the TargetElemType's memory layout, optionally emitting a GEP intrinsic.
  std::optional<std::pair<Value *, Type *>>
  getPointerToFirstCompatibleType(IRBuilder<> &B, Value *BasePtr,
                                  Type *PointerType, Type *TargetElemType,
                                  bool IsInBounds) {
    Type *CurrentTy = GR->findDeducedElementType(BasePtr);
    assert(CurrentTy && "Could not deduce aggregate type");
    SmallVector<Value *, 8> Args{/* isInBounds= */ B.getInt1(IsInBounds),
                                 BasePtr};
    Args.push_back(B.getInt32(0)); // Pointer offset

    while (!isCompatibleMemoryLayout(TargetElemType, CurrentTy)) {
      if (auto *ST = dyn_cast<StructType>(CurrentTy)) {
        if (ST->getNumElements() == 0)
          return std::nullopt;
        CurrentTy = ST->getTypeAtIndex(0u);
      } else if (auto *AT = dyn_cast<ArrayType>(CurrentTy)) {
        CurrentTy = AT->getElementType();
      } else if (auto *VT = dyn_cast<FixedVectorType>(CurrentTy)) {
        CurrentTy = VT->getElementType();
      } else {
        return std::nullopt;
      }
      Args.push_back(B.getInt32(0));
    }

    Value *GEP = BasePtr;
    if (Args.size() > 3) {
      std::array<Type *, 2> Types = {PointerType, BasePtr->getType()};
      GEP = B.CreateIntrinsic(Intrinsic::spv_gep, {Types}, {Args});
      GR->buildAssignPtr(B, CurrentTy, GEP);
    }

    return std::make_pair(GEP, CurrentTy);
  }

  // Drills down through aggregate types to find the first leaf sub-type whose
  // memory store size matches |TargetElemType|. Emits a GEP intrinsic if
  // needed. Returns the inner pointer and the leaf type. This is a fallback
  // used when isCompatibleMemoryLayout fails but the underlying memory still
  // has the same bit-width as the access (e.g. union punning float<->int).
  std::optional<std::pair<Value *, Type *>>
  getPointerToFirstSameSizeLeaf(IRBuilder<> &B, Value *BasePtr,
                                Type *PointerType, Type *TargetElemType,
                                bool IsInBounds) {
    Type *CurrentTy = GR->findDeducedElementType(BasePtr);
    if (!CurrentTy)
      return std::nullopt;
    const DataLayout &DL = B.GetInsertBlock()->getModule()->getDataLayout();
    if (!TargetElemType->isSized())
      return std::nullopt;
    TypeSize TargetSize = DL.getTypeStoreSize(TargetElemType);

    SmallVector<Value *, 8> Args{/* isInBounds= */ B.getInt1(IsInBounds),
                                 BasePtr};
    Args.push_back(B.getInt32(0)); // Pointer offset

    while (true) {
      // If sizes match and this is a primitive (non-aggregate, non-pointer)
      // type that bitcast can handle, accept it.
      if (CurrentTy->isSized() &&
          DL.getTypeStoreSize(CurrentTy) == TargetSize &&
          (CurrentTy->isIntegerTy() || CurrentTy->isFloatingPointTy() ||
           CurrentTy->isVectorTy()) &&
          (TargetElemType->isIntegerTy() ||
           TargetElemType->isFloatingPointTy() ||
           TargetElemType->isVectorTy()))
        break;

      if (auto *ST = dyn_cast<StructType>(CurrentTy)) {
        if (ST->getNumElements() == 0)
          return std::nullopt;
        CurrentTy = ST->getTypeAtIndex(0u);
      } else if (auto *AT = dyn_cast<ArrayType>(CurrentTy)) {
        CurrentTy = AT->getElementType();
      } else if (auto *VT = dyn_cast<FixedVectorType>(CurrentTy)) {
        // Stop at a vector if size matches; otherwise drill into elements.
        if (DL.getTypeStoreSize(VT) == TargetSize)
          break;
        CurrentTy = VT->getElementType();
      } else {
        return std::nullopt;
      }
      Args.push_back(B.getInt32(0));
    }

    Value *GEP = BasePtr;
    if (Args.size() > 3) {
      std::array<Type *, 2> Types = {PointerType, BasePtr->getType()};
      GEP = B.CreateIntrinsic(Intrinsic::spv_gep, {Types}, {Args});
      GR->buildAssignPtr(B, CurrentTy, GEP);
    }

    return std::make_pair(GEP, CurrentTy);
  }

  // Reinterprets a byte-typed deduced pointer as a wider value by loading
  // |TargetType|-many bytes individually and assembling them into a
  // <N x i8> vector, then bitcasting to |TargetType|. Returns null if the
  // deduced type isn't a byte array / single i8.
  // If |Source| is an spv_gep into a [N x i8] base whose final index is a
  // constant, rewrite the Source to instead point to the base [N x i8] and
  // populate |OutBase| with that base pointer plus |OutBaseOffset| with the
  // starting byte index. Returns true on success.
  bool unwrapByteGep(Value *Source, Value *&OutBase, uint64_t &OutBaseOffset,
                     Type *&OutBaseTy, Value *&OutDynOffset) {
    OutDynOffset = nullptr;
    // Peel through 3-arg no-op `spv_gep` wrappers (i.e. shape
    // (true|false, base, 0) — a single zero "pointer" index, semantically a
    // deref). These get emitted ahead of a ptrcast that re-types the pointer
    // and don't change the byte address.
    Value *Cur = Source;
    for (unsigned Depth = 0; Depth < 4; ++Depth) {
      auto *NopII = dyn_cast<IntrinsicInst>(Cur);
      if (!NopII || NopII->getIntrinsicID() != Intrinsic::spv_gep)
        break;
      if (NopII->arg_size() != 3)
        break;
      auto *Z = dyn_cast<ConstantInt>(NopII->getArgOperand(2));
      if (!Z || !Z->isZero())
        break;
      Cur = NopII->getArgOperand(1);
    }
    auto *II = dyn_cast<IntrinsicInst>(Cur);
    if (!II || II->getIntrinsicID() != Intrinsic::spv_gep)
      return false;
    // spv_gep args: (i1 inbounds, ptr base, i32 0 [, indices...]).
    if (II->arg_size() < 4)
      return false;
    Value *LastArg = II->getArgOperand(II->arg_size() - 1);
    auto *LastIdx = dyn_cast<ConstantInt>(LastArg);
    Value *Base = II->getArgOperand(1);
    Type *BaseTy = GR->findDeducedElementType(Base);
    Type *I8Ty = Type::getInt8Ty(II->getContext());
    if (!BaseTy)
      return false;
    auto *AT = dyn_cast<ArrayType>(BaseTy);
    if (!AT || AT->getElementType() != I8Ty)
      return false;
    // Only handle the simple shape: (true, base, 0, byteIdx).
    if (II->arg_size() != 4)
      return false;
    auto *Zero = dyn_cast<ConstantInt>(II->getArgOperand(2));
    if (!Zero || !Zero->isZero())
      return false;
    OutBase = Base;
    if (LastIdx)
      OutBaseOffset = LastIdx->getZExtValue();
    else {
      OutBaseOffset = 0;
      OutDynOffset = LastArg;
    }
    OutBaseTy = BaseTy;
    return true;
  }

  Value *tryLoadAsByteArray(IRBuilder<> &B, Value *Source, Type *PointerType,
                            Type *TargetType, Align Alignment) {
    Type *I8Ty = Type::getInt8Ty(B.getContext());
    // Fallback 6 (Phase H7): Source is `spv.resource.getpointer(handle, idx)`
    // where the buffer's element type is i8. Issue NumBytes successive
    // getpointer calls (idx, idx+1, ...) and assemble. Used when a device-
    // global is bound as a `[0 x i8]` storage buffer but the kernel reads
    // multi-byte values (e.g. struct field at byte offset N, sizeof(field)>1).
    if (auto *II = dyn_cast<IntrinsicInst>(Source)) {
      if (II->getIntrinsicID() == Intrinsic::spv_resource_getpointer &&
          TargetType->isSized()) {
        Type *Deduced = GR->findDeducedElementType(Source);
        if (Deduced == I8Ty) {
          const DataLayout &DL =
              B.GetInsertBlock()->getModule()->getDataLayout();
          unsigned NumBytes = DL.getTypeStoreSize(TargetType).getFixedValue();
          if (NumBytes == 0)
            return nullptr;
          Value *Handle = II->getArgOperand(0);
          Value *BaseIdx = II->getArgOperand(1);
          Type *I32Ty = Type::getInt32Ty(B.getContext());
          if (BaseIdx->getType() != I32Ty) {
            BaseIdx = B.CreateZExtOrTrunc(BaseIdx, I32Ty);
            if (auto *BI = dyn_cast<Instruction>(BaseIdx))
              buildAssignType(B, I32Ty, BI);
          }
          Type *PtrSB = II->getType();
          SmallVector<Value *, 8> Bytes;
          for (unsigned I = 0; I < NumBytes; ++I) {
            Value *Idx = BaseIdx;
            if (I != 0) {
              Idx = B.CreateAdd(BaseIdx, ConstantInt::get(I32Ty, I));
              if (auto *AI = dyn_cast<Instruction>(Idx))
                buildAssignType(B, I32Ty, AI);
            }
            Value *EP = B.CreateIntrinsic(
                PtrSB, Intrinsic::spv_resource_getpointer, {Handle, Idx});
            GR->buildAssignPtr(B, I8Ty, EP);
            Value *L = B.CreateLoad(I8Ty, EP);
            cast<LoadInst>(L)->setAlignment(I == 0 ? Alignment : Align(1));
            buildAssignType(B, I8Ty, L);
            Bytes.push_back(L);
          }
          Type *IntTy = IntegerType::get(B.getContext(), NumBytes * 8);
          Value *Acc = ConstantInt::get(IntTy, 0);
          for (unsigned I = 0; I < NumBytes; ++I) {
            Value *ZExt = B.CreateZExt(Bytes[I], IntTy);
            if (auto *ZI = dyn_cast<Instruction>(ZExt))
              buildAssignType(B, IntTy, ZI);
            if (I != 0) {
              Value *Shifted =
                  B.CreateShl(ZExt, ConstantInt::get(IntTy, I * 8));
              if (auto *SI = dyn_cast<Instruction>(Shifted))
                buildAssignType(B, IntTy, SI);
              ZExt = Shifted;
            }
            if (I == 0)
              Acc = ZExt;
            else {
              Acc = B.CreateOr(Acc, ZExt);
              if (auto *OI = dyn_cast<Instruction>(Acc))
                buildAssignType(B, IntTy, OI);
            }
          }
          if (TargetType == IntTy)
            return Acc;
          Value *Cast = B.CreateIntrinsic(Intrinsic::spv_bitcast,
                                          {TargetType, IntTy}, {Acc});
          buildAssignType(B, TargetType, Cast);
          return Cast;
        }
      }
    }
    // First: if Source is an spv_gep into [N x i8] with a constant byte
    // offset, emit per-byte spv_geps from the underlying base.
    Value *Base = nullptr;
    uint64_t BaseOffset = 0;
    Type *BaseTy = nullptr;
    Value *DynOffset = nullptr;
    if (unwrapByteGep(Source, Base, BaseOffset, BaseTy, DynOffset) &&
        TargetType->isSized()) {
      const DataLayout &DL = B.GetInsertBlock()->getModule()->getDataLayout();
      unsigned NumBytes = DL.getTypeStoreSize(TargetType).getFixedValue();
      if (NumBytes == 0)
        return nullptr;
      SmallVector<Value *, 4> Bytes;
      std::array<Type *, 2> GepTypes = {PointerType, Base->getType()};
      // Promote a dynamic byte offset to i32 so the per-byte indices match the
      // spv_gep convention used elsewhere in this pass.
      Value *DynOffI32 = nullptr;
      if (DynOffset) {
        DynOffI32 = DynOffset;
        if (DynOffI32->getType() != B.getInt32Ty()) {
          DynOffI32 = B.CreateZExtOrTrunc(DynOffI32, B.getInt32Ty());
          if (auto *CI = dyn_cast<Instruction>(DynOffI32))
            buildAssignType(B, B.getInt32Ty(), CI);
        }
      }
      for (unsigned I = 0; I < NumBytes; ++I) {
        Value *Idx;
        if (DynOffI32) {
          if (I == 0) {
            Idx = DynOffI32;
          } else {
            Idx = B.CreateAdd(DynOffI32, B.getInt32(I));
            if (auto *AI = dyn_cast<Instruction>(Idx))
              buildAssignType(B, B.getInt32Ty(), AI);
          }
        } else {
          Idx = B.getInt32(BaseOffset + I);
        }
        SmallVector<Value *, 4> Args = {B.getInt1(false), Base, B.getInt32(0),
                                        Idx};
        Value *EP =
            B.CreateIntrinsic(Intrinsic::spv_gep, {GepTypes}, {Args});
        GR->buildAssignPtr(B, I8Ty, EP);
        Value *L = B.CreateLoad(I8Ty, EP);
        cast<LoadInst>(L)->setAlignment(I == 0 ? Alignment : Align(1));
        buildAssignType(B, I8Ty, L);
        Bytes.push_back(L);
      }
      // Assemble bytes into an N*8-bit integer via zext+shift+or, avoiding
      // any vector<->scalar bitcasts which the GISel legalizer can't lower.
      Type *IntTy = IntegerType::get(B.getContext(), NumBytes * 8);
      Value *Acc = ConstantInt::get(IntTy, 0);
      for (unsigned I = 0; I < NumBytes; ++I) {
        Value *ZExt = B.CreateZExt(Bytes[I], IntTy);
        if (auto *ZI = dyn_cast<Instruction>(ZExt))
          buildAssignType(B, IntTy, ZI);
        if (I != 0) {
          Value *Shifted = B.CreateShl(ZExt, ConstantInt::get(IntTy, I * 8));
          if (auto *SI = dyn_cast<Instruction>(Shifted))
            buildAssignType(B, IntTy, SI);
          ZExt = Shifted;
        }
        if (I == 0) {
          Acc = ZExt;
        } else {
          Acc = B.CreateOr(Acc, ZExt);
          if (auto *OI = dyn_cast<Instruction>(Acc))
            buildAssignType(B, IntTy, OI);
        }
      }
      if (TargetType == IntTy)
        return Acc;
      Value *Cast = B.CreateIntrinsic(Intrinsic::spv_bitcast,
                                      {TargetType, IntTy}, {Acc});
      buildAssignType(B, TargetType, Cast);
      return Cast;
    }

    Type *Deduced = GR->findDeducedElementType(Source);
    if (!Deduced)
      return nullptr;
    // Drill into nested arrays of i8 until we hit either a single i8 or an
    // [N x i8] whose element type is i8. ChainIndices accumulates the
    // (always-zero) indices used to descend through any outer arrays. A
    // final per-byte index is appended in the loop below.
    Type *CurTy = Deduced;
    SmallVector<Value *, 4> ChainIndices;
    ChainIndices.push_back(B.getInt1(false));
    ChainIndices.push_back(Source);
    ChainIndices.push_back(B.getInt32(0)); // First index: pointer offset (0).
    bool DeducedIsByteArray = false;
    while (auto *AT = dyn_cast<ArrayType>(CurTy)) {
      if (AT->getElementType() == I8Ty) {
        DeducedIsByteArray = true;
        break;
      }
      CurTy = AT->getElementType();
      ChainIndices.push_back(B.getInt32(0));
    }
    if (!DeducedIsByteArray && CurTy != I8Ty)
      return nullptr;
    if (!TargetType->isSized())
      return nullptr;
    const DataLayout &DL = B.GetInsertBlock()->getModule()->getDataLayout();
    TypeSize TS = DL.getTypeStoreSize(TargetType);
    unsigned NumBytes = TS.getFixedValue();
    if (NumBytes == 0)
      return nullptr;
    // Load each byte and assemble into an N*8-bit integer via zext+shift+or,
    // avoiding any <N x i8> vectors (which would require the SPIR-V Vector16
    // capability for N==8 or N==16, unavailable in Vulkan/Shader mode).
    SmallVector<Value *, 4> Bytes;
    std::array<Type *, 2> GepTypes = {PointerType, Source->getType()};
    for (unsigned I = 0; I < NumBytes; ++I) {
      SmallVector<Value *, 8> Args(ChainIndices.begin(), ChainIndices.end());
      if (DeducedIsByteArray)
        Args.push_back(B.getInt32(I));
      else if (I != 0)
        // Single-i8 deduced type: only index 0 is valid; bail.
        return nullptr;
      Value *EP = B.CreateIntrinsic(Intrinsic::spv_gep, {GepTypes}, {Args});
      GR->buildAssignPtr(B, I8Ty, EP);
      Value *L = B.CreateLoad(I8Ty, EP);
      cast<LoadInst>(L)->setAlignment(I == 0 ? Alignment : Align(1));
      buildAssignType(B, I8Ty, L);
      Bytes.push_back(L);
    }
    Type *IntTy = IntegerType::get(B.getContext(), NumBytes * 8);
    Value *Acc = ConstantInt::get(IntTy, 0);
    for (unsigned I = 0; I < NumBytes; ++I) {
      Value *ZExt = B.CreateZExt(Bytes[I], IntTy);
      if (auto *ZI = dyn_cast<Instruction>(ZExt))
        buildAssignType(B, IntTy, ZI);
      if (I != 0) {
        Value *Shifted = B.CreateShl(ZExt, ConstantInt::get(IntTy, I * 8));
        if (auto *SI = dyn_cast<Instruction>(Shifted))
          buildAssignType(B, IntTy, SI);
        ZExt = Shifted;
      }
      if (I == 0) {
        Acc = ZExt;
      } else {
        Acc = B.CreateOr(Acc, ZExt);
        if (auto *OI = dyn_cast<Instruction>(Acc))
          buildAssignType(B, IntTy, OI);
      }
    }
    if (TargetType == IntTy)
      return Acc;
    Value *Cast = B.CreateIntrinsic(Intrinsic::spv_bitcast,
                                    {TargetType, IntTy}, {Acc});
    buildAssignType(B, TargetType, Cast);
    return Cast;
  }

  // Stores a wider value into a byte-typed deduced pointer by bitcasting to
  // <N x i8> and storing each byte via per-element GEPs. Returns true on
  // success.
  bool tryStoreAsByteArray(IRBuilder<> &B, Value *Src, Value *Dst,
                           Type *PointerType, Align Alignment) {
    Type *I8Ty = Type::getInt8Ty(B.getContext());
    Type *SrcTy = Src->getType();
    Value *Base = nullptr;
    uint64_t BaseOffset = 0;
    Type *BaseTy = nullptr;
    Value *DynOffset = nullptr;
    if (SrcTy->isSized() &&
        unwrapByteGep(Dst, Base, BaseOffset, BaseTy, DynOffset)) {
      const DataLayout &DL = B.GetInsertBlock()->getModule()->getDataLayout();
      unsigned NumBytes = DL.getTypeStoreSize(SrcTy).getFixedValue();
      if (NumBytes == 0)
        return false;
      Type *IntTy = IntegerType::get(B.getContext(), NumBytes * 8);
      // Bitcast Src -> i<N*8>, then extract each byte via lshr+trunc.
      Value *AsInt = Src;
      if (SrcTy != IntTy) {
        AsInt = B.CreateIntrinsic(Intrinsic::spv_bitcast, {IntTy, SrcTy},
                                  {Src});
        buildAssignType(B, IntTy, AsInt);
      }
      std::array<Type *, 2> GepTypes = {PointerType, Base->getType()};
      Value *DynOffI32 = nullptr;
      if (DynOffset) {
        DynOffI32 = DynOffset;
        if (DynOffI32->getType() != B.getInt32Ty()) {
          DynOffI32 = B.CreateZExtOrTrunc(DynOffI32, B.getInt32Ty());
          if (auto *CI = dyn_cast<Instruction>(DynOffI32))
            buildAssignType(B, B.getInt32Ty(), CI);
        }
      }
      for (unsigned I = 0; I < NumBytes; ++I) {
        Value *Shifted = AsInt;
        if (I != 0) {
          Shifted = B.CreateLShr(AsInt, ConstantInt::get(IntTy, I * 8));
          if (auto *SH = dyn_cast<Instruction>(Shifted))
            buildAssignType(B, IntTy, SH);
        }
        Value *Byte = B.CreateTrunc(Shifted, I8Ty);
        if (auto *TR = dyn_cast<Instruction>(Byte))
          buildAssignType(B, I8Ty, TR);
        Value *Idx;
        if (DynOffI32) {
          if (I == 0) {
            Idx = DynOffI32;
          } else {
            Idx = B.CreateAdd(DynOffI32, B.getInt32(I));
            if (auto *AI = dyn_cast<Instruction>(Idx))
              buildAssignType(B, B.getInt32Ty(), AI);
          }
        } else {
          Idx = B.getInt32(BaseOffset + I);
        }
        SmallVector<Value *, 4> Args = {B.getInt1(true), Base, B.getInt32(0),
                                        Idx};
        Value *EP =
            B.CreateIntrinsic(Intrinsic::spv_gep, {GepTypes}, {Args});
        GR->buildAssignPtr(B, I8Ty, EP);
        StoreInst *SI = B.CreateStore(Byte, EP);
        SI->setAlignment(I == 0 ? Alignment : Align(1));
      }
      return true;
    }
    Type *Deduced = GR->findDeducedElementType(Dst);
    if (!Deduced)
      return false;
    Type *CurTy = Deduced;
    SmallVector<Value *, 4> ChainIndices;
    ChainIndices.push_back(B.getInt1(true));
    ChainIndices.push_back(Dst);
    ChainIndices.push_back(B.getInt32(0));
    bool DeducedIsByteArray = false;
    while (auto *AT = dyn_cast<ArrayType>(CurTy)) {
      if (AT->getElementType() == I8Ty) {
        DeducedIsByteArray = true;
        break;
      }
      CurTy = AT->getElementType();
      ChainIndices.push_back(B.getInt32(0));
    }
    if (!DeducedIsByteArray && CurTy != I8Ty)
      return false;
    if (!SrcTy->isSized())
      return false;
    const DataLayout &DL = B.GetInsertBlock()->getModule()->getDataLayout();
    unsigned NumBytes = DL.getTypeStoreSize(SrcTy).getFixedValue();
    if (NumBytes == 0)
      return false;
    // Bitcast Src -> i<N*8> (scalar), then extract each byte via lshr+trunc,
    // avoiding any <N x i8> vectors (which would require the SPIR-V Vector16
    // capability for N==8 or N==16, unavailable in Vulkan/Shader mode).
    Type *IntTy = IntegerType::get(B.getContext(), NumBytes * 8);
    Value *AsInt = Src;
    if (SrcTy != IntTy) {
      AsInt = B.CreateIntrinsic(Intrinsic::spv_bitcast, {IntTy, SrcTy}, {Src});
      buildAssignType(B, IntTy, AsInt);
    }
    std::array<Type *, 2> GepTypes = {PointerType, Dst->getType()};
    for (unsigned I = 0; I < NumBytes; ++I) {
      SmallVector<Value *, 8> Args(ChainIndices.begin(), ChainIndices.end());
      if (DeducedIsByteArray)
        Args.push_back(B.getInt32(I));
      else if (I != 0)
        return false;
      Value *EP = B.CreateIntrinsic(Intrinsic::spv_gep, {GepTypes}, {Args});
      GR->buildAssignPtr(B, I8Ty, EP);
      Value *Shifted = AsInt;
      if (I != 0) {
        Shifted = B.CreateLShr(AsInt, ConstantInt::get(IntTy, I * 8));
        if (auto *SH = dyn_cast<Instruction>(Shifted))
          buildAssignType(B, IntTy, SH);
      }
      Value *Byte = B.CreateTrunc(Shifted, I8Ty);
      if (auto *TR = dyn_cast<Instruction>(Byte))
        buildAssignType(B, I8Ty, TR);
      StoreInst *SI = B.CreateStore(Byte, EP);
      SI->setAlignment(I == 0 ? Alignment : Align(1));
    }
    return true;
  }

  // If |Source| is an spv_gep representing a no-op deref (single 0 index,
  // i.e. semantically equivalent to its base pointer), unwrap it and
  // return the base. Returns null otherwise. This lets callers with a
  // generic-i8 deduced type for the GEP retry against the base, whose
  // deduced type may carry the real buffer element type (e.g. surfaced
  // via spv_resource_getpointer<VulkanBuffer<...>>). Skip when the
  // base's deduced type isn't more informative than what's already
  // available, to avoid disturbing OpenCL/physical paths.
  // If |Source| is an spv_gep of shape (*, base, 0, %idx) and |base|'s
  // deduced type is [K x ElemTy] where sizeof(ElemTy) == sizeof(TargetType),
  // emit a typed GEP+load and bitcast to TargetType. This handles the case
  // where SPIRVEmitIntrinsics emitted a typed access chain into an i32 (or
  // other) array but the spv_gep result's deduced element type fell back to
  // i8 (e.g. via spv_ptrcast/ptrcast lowering). Returns the loaded TargetType
  // value or nullptr if the pattern doesn't match.
  Value *tryTypedDynamicGepLoad(IRBuilder<> &B, Value *Source,
                                Type *PointerType, Type *TargetType,
                                Align Alignment) {
    auto *II = dyn_cast<IntrinsicInst>(Source);
    if (!II || II->getIntrinsicID() != Intrinsic::spv_gep)
      return nullptr;
    if (II->arg_size() != 4)
      return nullptr;
    auto *Zero = dyn_cast<ConstantInt>(II->getArgOperand(2));
    if (!Zero || !Zero->isZero())
      return nullptr;
    Value *Base = II->getArgOperand(1);
    Type *BaseTy = GR->findDeducedElementType(Base);
    if (!BaseTy)
      return nullptr;
    auto *AT = dyn_cast<ArrayType>(BaseTy);
    if (!AT)
      return nullptr;
    Type *ElemTy = AT->getElementType();
    if (!ElemTy->isSized() || !TargetType->isSized())
      return nullptr;
    const DataLayout &DL = B.GetInsertBlock()->getModule()->getDataLayout();
    if (DL.getTypeStoreSize(ElemTy) != DL.getTypeStoreSize(TargetType))
      return nullptr;
    // Must be a primitive ElemTy (int / fp / vector) so a bitcast is legal.
    if (!(ElemTy->isIntegerTy() || ElemTy->isFloatingPointTy() ||
          ElemTy->isVectorTy()))
      return nullptr;
    Value *Idx = II->getArgOperand(3);
    std::array<Type *, 2> GepTypes = {PointerType, Base->getType()};
    SmallVector<Value *, 4> Args = {B.getInt1(false), Base, B.getInt32(0), Idx};
    Value *EP = B.CreateIntrinsic(Intrinsic::spv_gep, {GepTypes}, {Args});
    GR->buildAssignPtr(B, ElemTy, EP);
    LoadInst *LI = B.CreateLoad(ElemTy, EP);
    LI->setAlignment(Alignment);
    buildAssignType(B, ElemTy, LI);
    if (ElemTy == TargetType)
      return LI;
    Value *Cast = B.CreateIntrinsic(Intrinsic::spv_bitcast,
                                    {TargetType, ElemTy}, {LI});
    buildAssignType(B, TargetType, Cast);
    return Cast;
  }

  // Mirror of tryTypedDynamicGepLoad for stores. Returns true on success.
  bool tryTypedDynamicGepStore(IRBuilder<> &B, Value *Src, Value *Dst,
                               Type *PointerType, Align Alignment) {
    auto *II = dyn_cast<IntrinsicInst>(Dst);
    if (!II || II->getIntrinsicID() != Intrinsic::spv_gep)
      return false;
    if (II->arg_size() != 4)
      return false;
    auto *Zero = dyn_cast<ConstantInt>(II->getArgOperand(2));
    if (!Zero || !Zero->isZero())
      return false;
    Value *Base = II->getArgOperand(1);
    Type *BaseTy = GR->findDeducedElementType(Base);
    if (!BaseTy)
      return false;
    auto *AT = dyn_cast<ArrayType>(BaseTy);
    if (!AT)
      return false;
    Type *ElemTy = AT->getElementType();
    Type *SrcTy = Src->getType();
    if (!ElemTy->isSized() || !SrcTy->isSized())
      return false;
    const DataLayout &DL = B.GetInsertBlock()->getModule()->getDataLayout();
    if (DL.getTypeStoreSize(ElemTy) != DL.getTypeStoreSize(SrcTy))
      return false;
    if (!(ElemTy->isIntegerTy() || ElemTy->isFloatingPointTy() ||
          ElemTy->isVectorTy()))
      return false;
    Value *Idx = II->getArgOperand(3);
    std::array<Type *, 2> GepTypes = {PointerType, Base->getType()};
    SmallVector<Value *, 4> Args = {B.getInt1(true), Base, B.getInt32(0), Idx};
    Value *EP = B.CreateIntrinsic(Intrinsic::spv_gep, {GepTypes}, {Args});
    GR->buildAssignPtr(B, ElemTy, EP);
    Value *Stored = Src;
    if (ElemTy != SrcTy) {
      Stored = B.CreateIntrinsic(Intrinsic::spv_bitcast, {ElemTy, SrcTy},
                                 {Src});
      buildAssignType(B, ElemTy, Stored);
    }
    StoreInst *SI = B.CreateStore(Stored, EP);
    SI->setAlignment(Alignment);
    return true;
  }

  // Fallback for narrow loads (i8 / i16) from a pointer whose deduced type
  // is a typed aggregate (struct/array/vector ending in a primitive leaf
  // like float / i32 / i64) — neither a byte array nor a same-size primitive.
  // Descends to the first primitive leaf, loads it as its native type,
  // bitcasts to an integer, and lshr+truncs out the requested byte/half.
  // This handles patterns produced by lowering memcpys between LDS-resident
  // typed aggregates and Vulkan storage buffers, where the source is e.g.
  // `[N x %struct.HIP_vector_type]` (16-byte vec4 elements) and the load is
  // a single byte.
  Value *tryLoadFromAggregateLeaf(IRBuilder<> &B, Value *Source,
                                  Type *PointerType, Type *TargetType,
                                  Align Alignment) {
    if (!TargetType->isIntegerTy() && !TargetType->isFloatingPointTy())
      return nullptr;
    if (!TargetType->isSized())
      return nullptr;
    const DataLayout &DL = B.GetInsertBlock()->getModule()->getDataLayout();
    TypeSize TS = DL.getTypeStoreSize(TargetType);
    Type *Deduced = GR->findDeducedElementType(Source);
    if (!Deduced)
      return nullptr;
    Type *CurTy = Deduced;
    SmallVector<Value *, 8> ChainIndices;
    ChainIndices.push_back(B.getInt1(false));
    ChainIndices.push_back(Source);
    ChainIndices.push_back(B.getInt32(0));
    // Descend struct/array/vector to a primitive leaf, always taking index 0.
    while (true) {
      if (auto *ST = dyn_cast<StructType>(CurTy)) {
        if (ST->getNumElements() == 0)
          return nullptr;
        CurTy = ST->getElementType(0);
      } else if (auto *AT = dyn_cast<ArrayType>(CurTy)) {
        CurTy = AT->getElementType();
      } else if (auto *VT = dyn_cast<FixedVectorType>(CurTy)) {
        CurTy = VT->getElementType();
      } else {
        break;
      }
      ChainIndices.push_back(B.getInt32(0));
    }
    // CurTy must be a primitive type strictly larger than the target.
    if (!CurTy->isIntegerTy() && !CurTy->isFloatingPointTy())
      return nullptr;
    if (!CurTy->isSized())
      return nullptr;
    TypeSize LeafSize = DL.getTypeStoreSize(CurTy);
    if (LeafSize <= TS)
      return nullptr;
    Value *EP = Source;
    if (ChainIndices.size() > 3) {
      std::array<Type *, 2> GepTypes = {PointerType, Source->getType()};
      EP = B.CreateIntrinsic(Intrinsic::spv_gep, {GepTypes}, {ChainIndices});
      GR->buildAssignPtr(B, CurTy, EP);
    }
    LoadInst *LI = B.CreateLoad(CurTy, EP);
    LI->setAlignment(Alignment);
    buildAssignType(B, CurTy, LI);
    // Bitcast leaf to an integer of the same width.
    unsigned LeafBits = (unsigned)(LeafSize.getFixedValue() * 8);
    Type *LeafIntTy = IntegerType::get(B.getContext(), LeafBits);
    Value *AsInt = LI;
    if (CurTy != LeafIntTy) {
      AsInt = B.CreateIntrinsic(Intrinsic::spv_bitcast, {LeafIntTy, CurTy},
                                {LI});
      buildAssignType(B, LeafIntTy, AsInt);
    }
    // Truncate to TargetType-width (extracting the low bits, which corresponds
    // to byte offset 0 of the leaf — the access chain has already pointed us
    // to the right starting position via the leading-zero descent).
    unsigned TargetBits = (unsigned)(TS.getFixedValue() * 8);
    if (TargetBits < LeafBits) {
      Type *TruncIntTy = IntegerType::get(B.getContext(), TargetBits);
      AsInt = B.CreateTrunc(AsInt, TruncIntTy);
      if (auto *TI = dyn_cast<Instruction>(AsInt))
        buildAssignType(B, TruncIntTy, TI);
    }
    if (AsInt->getType() == TargetType)
      return AsInt;
    Value *Cast = B.CreateIntrinsic(Intrinsic::spv_bitcast,
                                    {TargetType, AsInt->getType()}, {AsInt});
    buildAssignType(B, TargetType, Cast);
    return Cast;
  }

  // Mirror of tryLoadFromAggregateLeaf for stores. If |Src| is a narrow int
  // (i8 / i16 / ...) and |Dst|'s deduced type is a struct/array/vector
  // descending (always taking index 0) to a primitive leaf strictly larger
  // than |Src|, emit a read-modify-write: load leaf, bitcast to int, clear
  // the low TargetBits, OR-in zext(Src), bitcast back, store.
  // Handles the byte-by-byte memcpy-into-typed-pointer pattern emitted by
  // O2 for cases like `*((char*)&struct_var) = b;`. Returns true on success.
  bool tryStoreToAggregateLeaf(IRBuilder<> &B, Value *Src, Value *Dst,
                               Type *PointerType, Align Alignment) {
    Type *SrcTy = Src->getType();
    if (!SrcTy->isIntegerTy() || !SrcTy->isSized())
      return false;
    Type *Deduced = GR->findDeducedElementType(Dst);
    if (!Deduced)
      return false;
    Type *CurTy = Deduced;
    SmallVector<Value *, 8> ChainIndices;
    ChainIndices.push_back(B.getInt1(true));
    ChainIndices.push_back(Dst);
    ChainIndices.push_back(B.getInt32(0));
    while (true) {
      if (auto *ST = dyn_cast<StructType>(CurTy)) {
        if (ST->getNumElements() == 0)
          return false;
        CurTy = ST->getElementType(0);
      } else if (auto *AT = dyn_cast<ArrayType>(CurTy)) {
        CurTy = AT->getElementType();
      } else if (auto *VT = dyn_cast<FixedVectorType>(CurTy)) {
        CurTy = VT->getElementType();
      } else {
        break;
      }
      ChainIndices.push_back(B.getInt32(0));
    }
    if (!CurTy->isIntegerTy() && !CurTy->isFloatingPointTy())
      return false;
    if (!CurTy->isSized())
      return false;
    const DataLayout &DL = B.GetInsertBlock()->getModule()->getDataLayout();
    TypeSize TS = DL.getTypeStoreSize(SrcTy);
    TypeSize LeafSize = DL.getTypeStoreSize(CurTy);
    if (LeafSize <= TS)
      return false;
    Value *EP = Dst;
    if (ChainIndices.size() > 3) {
      std::array<Type *, 2> GepTypes = {PointerType, Dst->getType()};
      EP = B.CreateIntrinsic(Intrinsic::spv_gep, {GepTypes}, ChainIndices);
      GR->buildAssignPtr(B, CurTy, EP);
    }
    LoadInst *LI = B.CreateLoad(CurTy, EP);
    LI->setAlignment(Alignment);
    buildAssignType(B, CurTy, LI);
    unsigned LeafBits = (unsigned)(LeafSize.getFixedValue() * 8);
    Type *LeafIntTy = IntegerType::get(B.getContext(), LeafBits);
    Value *AsInt = LI;
    if (CurTy != LeafIntTy) {
      AsInt = B.CreateIntrinsic(Intrinsic::spv_bitcast, {LeafIntTy, CurTy},
                                {LI});
      buildAssignType(B, LeafIntTy, AsInt);
    }
    unsigned TargetBits = (unsigned)(TS.getFixedValue() * 8);
    APInt MaskAP = APInt::getLowBitsSet(LeafBits, TargetBits);
    Value *Cleared =
        B.CreateAnd(AsInt, ConstantInt::get(LeafIntTy, ~MaskAP));
    if (auto *AI = dyn_cast<Instruction>(Cleared))
      buildAssignType(B, LeafIntTy, AI);
    Value *SrcZExt = Src;
    if (SrcTy != LeafIntTy) {
      SrcZExt = B.CreateZExt(Src, LeafIntTy);
      if (auto *ZI = dyn_cast<Instruction>(SrcZExt))
        buildAssignType(B, LeafIntTy, ZI);
    }
    Value *Merged = B.CreateOr(Cleared, SrcZExt);
    if (auto *OI = dyn_cast<Instruction>(Merged))
      buildAssignType(B, LeafIntTy, OI);
    Value *Stored = Merged;
    if (CurTy != LeafIntTy) {
      Stored = B.CreateIntrinsic(Intrinsic::spv_bitcast, {CurTy, LeafIntTy},
                                 {Merged});
      buildAssignType(B, CurTy, Stored);
    }
    StoreInst *SI = B.CreateStore(Stored, EP);
    SI->setAlignment(Alignment);
    return true;
  }

  // Resolve a ConstantExpr `getelementptr inbounds (i8, ptr %base, i64 N)`
  // (byte-offset GEP into a typed global) into a typed access-chain landing
  // at the leaf field whose store size matches |TargetSize|. Walks |BaseTy|
  // by struct/array/vector layout using the DataLayout. Returns the leaf
  // type plus the chain indices (each i32) suitable for an spv_gep, or
  // std::nullopt if the descent can't land cleanly at byte offset N with
  // store size equal to |TargetSize|.
  std::optional<std::pair<Type *, SmallVector<unsigned, 4>>>
  resolveByteOffsetToTypedChain(Type *BaseTy, uint64_t ByteOffset,
                                TypeSize TargetSize, const DataLayout &DL) {
    SmallVector<unsigned, 4> Indices;
    Type *CurTy = BaseTy;
    uint64_t RemainOff = ByteOffset;
    while (true) {
      if (RemainOff == 0 && DL.getTypeStoreSize(CurTy) == TargetSize)
        return std::make_pair(CurTy, Indices);
      if (auto *AT = dyn_cast<ArrayType>(CurTy)) {
        uint64_t ElSz =
            DL.getTypeAllocSize(AT->getElementType()).getFixedValue();
        if (ElSz == 0)
          return std::nullopt;
        uint64_t Idx = RemainOff / ElSz;
        if (Idx >= AT->getNumElements())
          return std::nullopt;
        RemainOff %= ElSz;
        Indices.push_back((unsigned)Idx);
        CurTy = AT->getElementType();
        continue;
      }
      if (auto *ST = dyn_cast<StructType>(CurTy)) {
        const StructLayout *SL = DL.getStructLayout(ST);
        if (RemainOff >= SL->getSizeInBytes())
          return std::nullopt;
        unsigned FldIdx = SL->getElementContainingOffset(RemainOff);
        RemainOff -= SL->getElementOffset(FldIdx);
        Indices.push_back(FldIdx);
        CurTy = ST->getElementType(FldIdx);
        continue;
      }
      if (auto *VT = dyn_cast<FixedVectorType>(CurTy)) {
        uint64_t ElSz =
            DL.getTypeAllocSize(VT->getElementType()).getFixedValue();
        if (ElSz == 0)
          return std::nullopt;
        uint64_t Idx = RemainOff / ElSz;
        if (Idx >= VT->getNumElements())
          return std::nullopt;
        RemainOff %= ElSz;
        Indices.push_back((unsigned)Idx);
        CurTy = VT->getElementType();
        continue;
      }
      // Primitive leaf — only acceptable if it matches the target.
      if (RemainOff == 0 && DL.getTypeStoreSize(CurTy) == TargetSize)
        return std::make_pair(CurTy, Indices);
      return std::nullopt;
    }
  }

  // If |Source| is a ConstantExpr byte-offset GEP into a typed global
  // (e.g. `getelementptr inbounds (i8, ptr @sdata, i64 24)`), and |@sdata|
  // has a deduced aggregate type whose layout has a leaf at the requested
  // byte offset matching |TargetType|'s store size, emit a typed spv_gep
  // descent + load + bitcast and return the value. Returns null if the
  // pattern doesn't match.
  Value *tryLoadFromConstantByteGep(IRBuilder<> &B, Value *Source,
                                    Type *PointerType, Type *TargetType,
                                    Align Alignment) {
    auto *CE = dyn_cast<ConstantExpr>(Source);
    if (!CE || CE->getOpcode() != Instruction::GetElementPtr)
      return nullptr;
    auto *GO = cast<GEPOperator>(CE);
    if (!GO->getSourceElementType()->isIntegerTy(8))
      return nullptr;
    if (CE->getNumOperands() != 2)
      return nullptr;
    auto *OffC = dyn_cast<ConstantInt>(CE->getOperand(1));
    if (!OffC)
      return nullptr;
    uint64_t ByteOffset = OffC->getZExtValue();
    Value *Base = CE->getOperand(0);
    Type *BaseDeduced = GR->findDeducedElementType(Base);
    if (!BaseDeduced || !TargetType->isSized())
      return nullptr;
    const DataLayout &DL = B.GetInsertBlock()->getModule()->getDataLayout();
    TypeSize TS = DL.getTypeStoreSize(TargetType);
    auto Chain = resolveByteOffsetToTypedChain(BaseDeduced, ByteOffset, TS, DL);
    if (!Chain)
      return nullptr;
    auto [LeafTy, Indices] = *Chain;
    SmallVector<Value *, 8> Args;
    Args.push_back(B.getInt1(false));
    Args.push_back(Base);
    Args.push_back(B.getInt32(0));
    for (unsigned Idx : Indices)
      Args.push_back(B.getInt32(Idx));
    std::array<Type *, 2> GepTypes = {PointerType, Base->getType()};
    Value *EP = Base;
    if (Indices.size() > 0) {
      EP = B.CreateIntrinsic(Intrinsic::spv_gep, {GepTypes}, Args);
      GR->buildAssignPtr(B, LeafTy, EP);
    }
    LoadInst *LI = B.CreateLoad(LeafTy, EP);
    LI->setAlignment(Alignment);
    buildAssignType(B, LeafTy, LI);
    if (LeafTy == TargetType)
      return LI;
    Value *Cast = B.CreateIntrinsic(Intrinsic::spv_bitcast, {TargetType, LeafTy},
                                    {LI});
    buildAssignType(B, TargetType, Cast);
    return Cast;
  }
  // If |Source|'s deduced type is a struct whose total store size equals
  // |TargetType|'s store size and whose fields are all primitive scalars
  // (int / float), load each field at its native type, bitcast/zext to the
  // target-wide integer, shift by the field's byte-offset, and OR them
  // together. Returns the resulting value bitcast to |TargetType|, or null
  // if the pattern doesn't match.
  // Handles `load i64, ptr %p` where %p's deduced type is `%struct.Pair =
  // {i32, i32}` — a same-size scalar reinterpret over a typed pointer.
  // Flatten the leaves of an aggregate type (struct/array/vector of
  // primitives, recursively). Fills |Leaves| with (leaf type, byte offset
  // from the root, chain of GEP indices descending into the root) for each
  // primitive leaf. Returns false if any leaf is non-primitive or unsized.
  bool flattenAggregateLeaves(
      Type *T, const DataLayout &DL, uint64_t BaseOff,
      SmallVectorImpl<unsigned> &Chain,
      SmallVectorImpl<std::tuple<Type *, uint64_t, SmallVector<unsigned, 4>>>
          &Leaves) {
    if (T->isIntegerTy() || T->isFloatingPointTy()) {
      Leaves.emplace_back(T, BaseOff,
                          SmallVector<unsigned, 4>(Chain.begin(), Chain.end()));
      return true;
    }
    if (auto *ST = dyn_cast<StructType>(T)) {
      const StructLayout *SL = DL.getStructLayout(ST);
      for (unsigned I = 0; I < ST->getNumElements(); ++I) {
        Chain.push_back(I);
        if (!flattenAggregateLeaves(ST->getElementType(I), DL,
                                    BaseOff + SL->getElementOffset(I), Chain,
                                    Leaves))
          return false;
        Chain.pop_back();
      }
      return true;
    }
    if (auto *AT = dyn_cast<ArrayType>(T)) {
      uint64_t ElSz = DL.getTypeAllocSize(AT->getElementType()).getFixedValue();
      for (unsigned I = 0; I < AT->getNumElements(); ++I) {
        Chain.push_back(I);
        if (!flattenAggregateLeaves(AT->getElementType(), DL, BaseOff + I * ElSz,
                                    Chain, Leaves))
          return false;
        Chain.pop_back();
      }
      return true;
    }
    if (auto *VT = dyn_cast<FixedVectorType>(T)) {
      uint64_t ElSz = DL.getTypeAllocSize(VT->getElementType()).getFixedValue();
      for (unsigned I = 0; I < VT->getNumElements(); ++I) {
        Chain.push_back(I);
        if (!flattenAggregateLeaves(VT->getElementType(), DL, BaseOff + I * ElSz,
                                    Chain, Leaves))
          return false;
        Chain.pop_back();
      }
      return true;
    }
    return false;
  }

  Value *tryLoadStructAsScalarInt(IRBuilder<> &B, Value *Source,
                                  Type *PointerType, Type *TargetType,
                                  Align Alignment) {
    if (!TargetType->isIntegerTy() || !TargetType->isSized())
      return nullptr;
    Type *Deduced = GR->findDeducedElementType(Source);
    if (!Deduced || !Deduced->isAggregateType())
      return nullptr;
    const DataLayout &DL = B.GetInsertBlock()->getModule()->getDataLayout();
    TypeSize AggSize = DL.getTypeStoreSize(Deduced);
    TypeSize TS = DL.getTypeStoreSize(TargetType);
    if (AggSize != TS)
      return nullptr;
    SmallVector<std::tuple<Type *, uint64_t, SmallVector<unsigned, 4>>, 4>
        Leaves;
    SmallVector<unsigned, 4> Chain;
    if (!flattenAggregateLeaves(Deduced, DL, 0, Chain, Leaves))
      return nullptr;
    if (Leaves.empty())
      return nullptr;
    unsigned TargetBits = (unsigned)(TS.getFixedValue() * 8);
    Type *IntTy = IntegerType::get(B.getContext(), TargetBits);
    std::array<Type *, 2> GepTypes = {PointerType, Source->getType()};
    Value *Acc = nullptr;
    for (auto &[FT, Off, Idxs] : Leaves) {
      if (!FT->isSized())
        return nullptr;
      unsigned FBits = (unsigned)(DL.getTypeStoreSize(FT).getFixedValue() * 8);
      SmallVector<Value *, 8> Args = {B.getInt1(false), Source, B.getInt32(0)};
      for (unsigned I : Idxs)
        Args.push_back(B.getInt32(I));
      Value *EP = B.CreateIntrinsic(Intrinsic::spv_gep, {GepTypes}, Args);
      GR->buildAssignPtr(B, FT, EP);
      LoadInst *LI = B.CreateLoad(FT, EP);
      LI->setAlignment(Off == 0 ? Alignment
                                : Align(FBits / 8 ? FBits / 8 : 1));
      buildAssignType(B, FT, LI);
      Type *FldIntTy = IntegerType::get(B.getContext(), FBits);
      Value *AsInt = LI;
      if (FT != FldIntTy) {
        AsInt = B.CreateIntrinsic(Intrinsic::spv_bitcast, {FldIntTy, FT}, {LI});
        buildAssignType(B, FldIntTy, AsInt);
      }
      if (FBits < TargetBits) {
        AsInt = B.CreateZExt(AsInt, IntTy);
        if (auto *Inst = dyn_cast<Instruction>(AsInt))
          buildAssignType(B, IntTy, Inst);
      }
      if (Off != 0) {
        AsInt = B.CreateShl(AsInt, ConstantInt::get(IntTy, Off * 8));
        if (auto *Inst = dyn_cast<Instruction>(AsInt))
          buildAssignType(B, IntTy, Inst);
      }
      if (!Acc) {
        Acc = AsInt;
      } else {
        Acc = B.CreateOr(Acc, AsInt);
        if (auto *Inst = dyn_cast<Instruction>(Acc))
          buildAssignType(B, IntTy, Inst);
      }
    }
    if (TargetType == IntTy)
      return Acc;
    Value *Cast = B.CreateIntrinsic(Intrinsic::spv_bitcast, {TargetType, IntTy},
                                    {Acc});
    buildAssignType(B, TargetType, Cast);
    return Cast;
  }

  // Mirror of tryLoadFromConstantByteGep for stores. If |Dst| is a
  // ConstantExpr `getelementptr inbounds (i8, ptr @global, i64 N)` and the
  // global has a deduced aggregate type whose layout has a leaf at byte
  // offset N matching |Src|'s store size, emit a typed spv_gep descent +
  // store (with a leading bitcast if needed). Handles LDS-resident globals
  // like `[N x double] @smem_svds` accessed via byte-stride stores.
  bool tryStoreToConstantByteGep(IRBuilder<> &B, Value *Src, Value *Dst,
                                 Type *PointerType, Align Alignment) {
    auto *CE = dyn_cast<ConstantExpr>(Dst);
    if (!CE || CE->getOpcode() != Instruction::GetElementPtr)
      return false;
    auto *GO = cast<GEPOperator>(CE);
    if (!GO->getSourceElementType()->isIntegerTy(8))
      return false;
    if (CE->getNumOperands() != 2)
      return false;
    auto *OffC = dyn_cast<ConstantInt>(CE->getOperand(1));
    if (!OffC)
      return false;
    uint64_t ByteOffset = OffC->getZExtValue();
    Value *Base = CE->getOperand(0);
    Type *BaseDeduced = GR->findDeducedElementType(Base);
    Type *SrcTy = Src->getType();
    if (!BaseDeduced || !SrcTy->isSized())
      return false;
    const DataLayout &DL = B.GetInsertBlock()->getModule()->getDataLayout();
    TypeSize TS = DL.getTypeStoreSize(SrcTy);
    auto Chain = resolveByteOffsetToTypedChain(BaseDeduced, ByteOffset, TS, DL);
    if (!Chain)
      return false;
    auto [LeafTy, Indices] = *Chain;
    SmallVector<Value *, 8> Args;
    Args.push_back(B.getInt1(true));
    Args.push_back(Base);
    Args.push_back(B.getInt32(0));
    for (unsigned Idx : Indices)
      Args.push_back(B.getInt32(Idx));
    std::array<Type *, 2> GepTypes = {PointerType, Base->getType()};
    Value *EP = Base;
    if (Indices.size() > 0) {
      EP = B.CreateIntrinsic(Intrinsic::spv_gep, {GepTypes}, Args);
      GR->buildAssignPtr(B, LeafTy, EP);
    }
    Value *Stored = Src;
    if (LeafTy != SrcTy) {
      Stored = B.CreateIntrinsic(Intrinsic::spv_bitcast, {LeafTy, SrcTy}, {Src});
      buildAssignType(B, LeafTy, Stored);
    }
    StoreInst *SI = B.CreateStore(Stored, EP);
    SI->setAlignment(Alignment);
    return true;
  }

  // Mirror of tryLoadStructAsScalarInt for stores. If |Dst|'s deduced type
  // is a struct of primitives whose total store size matches |Src|'s integer
  // type, split |Src| by field offsets (lshr + trunc + bitcast) and store
  // each field at its typed pointer. Returns true on success.
  bool tryStoreScalarIntAsStruct(IRBuilder<> &B, Value *Src, Value *Dst,
                                 Type *PointerType, Align Alignment) {
    Type *SrcTy = Src->getType();
    if (!SrcTy->isIntegerTy() || !SrcTy->isSized())
      return false;
    Type *Deduced = GR->findDeducedElementType(Dst);
    if (!Deduced || !Deduced->isAggregateType())
      return false;
    const DataLayout &DL = B.GetInsertBlock()->getModule()->getDataLayout();
    TypeSize AggSize = DL.getTypeStoreSize(Deduced);
    TypeSize TS = DL.getTypeStoreSize(SrcTy);
    // Allow partial stores: TS can be < AggSize. We'll only write leaves
    // whose offset+size lies fully within the source's bytes.
    if (AggSize < TS)
      return false;
    SmallVector<std::tuple<Type *, uint64_t, SmallVector<unsigned, 4>>, 4>
        Leaves;
    SmallVector<unsigned, 4> Chain;
    if (!flattenAggregateLeaves(Deduced, DL, 0, Chain, Leaves))
      return false;
    if (Leaves.empty())
      return false;
    unsigned TargetBits = (unsigned)(TS.getFixedValue() * 8);
    Type *IntTy = IntegerType::get(B.getContext(), TargetBits);
    Value *AsInt = Src;
    if (SrcTy != IntTy) {
      AsInt = B.CreateIntrinsic(Intrinsic::spv_bitcast, {IntTy, SrcTy}, {Src});
      buildAssignType(B, IntTy, AsInt);
    }
    std::array<Type *, 2> GepTypes = {PointerType, Dst->getType()};
    uint64_t SrcBytes = TS.getFixedValue();
    for (auto &[FT, Off, Idxs] : Leaves) {
      if (!FT->isSized())
        return false;
      unsigned FBits = (unsigned)(DL.getTypeStoreSize(FT).getFixedValue() * 8);
      uint64_t FBytes = FBits / 8;
      // Skip leaves that fall outside Src's byte range (partial store).
      if (Off >= SrcBytes)
        continue;
      if (Off + FBytes > SrcBytes)
        return false; // partial leaf — bail, RMW would be needed.
      Value *Shifted = AsInt;
      if (Off != 0) {
        Shifted = B.CreateLShr(AsInt, ConstantInt::get(IntTy, Off * 8));
        if (auto *Inst = dyn_cast<Instruction>(Shifted))
          buildAssignType(B, IntTy, Inst);
      }
      Type *FldIntTy = IntegerType::get(B.getContext(), FBits);
      Value *FldVal = Shifted;
      if (FBits < TargetBits) {
        FldVal = B.CreateTrunc(Shifted, FldIntTy);
        if (auto *Inst = dyn_cast<Instruction>(FldVal))
          buildAssignType(B, FldIntTy, Inst);
      }
      if (FT != FldIntTy) {
        FldVal = B.CreateIntrinsic(Intrinsic::spv_bitcast, {FT, FldIntTy},
                                   {FldVal});
        buildAssignType(B, FT, FldVal);
      }
      SmallVector<Value *, 8> Args = {B.getInt1(true), Dst, B.getInt32(0)};
      for (unsigned I : Idxs)
        Args.push_back(B.getInt32(I));
      Value *EP = B.CreateIntrinsic(Intrinsic::spv_gep, {GepTypes}, Args);
      GR->buildAssignPtr(B, FT, EP);
      StoreInst *SI = B.CreateStore(FldVal, EP);
      SI->setAlignment(Off == 0 ? Alignment
                                : Align(FBits / 8 ? FBits / 8 : 1));
    }
    return true;
  }

  Value *peelNoOpSpvGep(Value *Source) {
    // Walk through up to 4 levels of 3-arg no-op `spv_gep(true|false, base, 0)`
    // wrappers. SCEV / LSR sometimes re-derives the same address as a chain of
    // such GEPs, and SPIRVEmitIntrinsics assigns each an `i8` deduced type
    // even when the original base has a richer typed element type.
    Value *Cur = Source;
    for (unsigned Depth = 0; Depth < 4; ++Depth) {
      auto *II = dyn_cast<IntrinsicInst>(Cur);
      if (!II || II->getIntrinsicID() != Intrinsic::spv_gep)
        break;
      if (II->arg_size() != 3)
        break;
      auto *CI0 = dyn_cast<ConstantInt>(II->getArgOperand(2));
      if (!CI0 || !CI0->isZero())
        break;
      Cur = II->getArgOperand(1);
    }
    if (Cur == Source)
      return nullptr;
    Type *SrcTy = GR->findDeducedElementType(Source);
    Type *BaseTy = GR->findDeducedElementType(Cur);
    if (!BaseTy || BaseTy == SrcTy)
      return nullptr;
    return Cur;
  }

  // Builds a legalized load from a pointer, drilling down through
  // memory layouts to find a compatible type. Load flags will be
  // copied from |BadLoad|, which should be the load being legalized.
  Value *buildLegalizedLoad(IRBuilder<> &B, Type *ElementType, Value *Source,
                            LoadInst *BadLoad) {
    // Load from undef/poison is undefined behaviour; return poison of
    // the requested element type.
    if (isa<UndefValue>(Source) || isa<PoisonValue>(Source)) {
      Value *P = PoisonValue::get(ElementType);
      return P;
    }
    auto ResultOpt = getPointerToFirstCompatibleType(
        B, Source, BadLoad->getPointerOperandType(), ElementType, false);
    if (!ResultOpt) {
      // Fallback 1: same-size leaf (load + bitcast, e.g. float<->i32).
      auto LeafOpt = getPointerToFirstSameSizeLeaf(
          B, Source, BadLoad->getPointerOperandType(), ElementType, false);
      if (LeafOpt) {
        auto [LeafGEP, LeafTy] = *LeafOpt;
        LoadInst *LI = B.CreateLoad(LeafTy, LeafGEP);
        LI->setAlignment(BadLoad->getAlign());
        buildAssignType(B, LeafTy, LI);
        if (LeafTy == ElementType)
          return LI;
        Value *Cast = B.CreateIntrinsic(Intrinsic::spv_bitcast,
                                        {ElementType, LeafTy}, {LI});
        buildAssignType(B, ElementType, Cast);
        return Cast;
      }
      // Fallback 2: byte-array reinterpretation. Deduced is [N x i8] or i8;
      // load TargetSize bytes via per-element GEPs, build <TargetSize x i8>,
      // and bitcast to ElementType. Emitted by O2 byte-indexed GEPs.
      Value *ByteVal = tryLoadAsByteArray(B, Source,
                                          BadLoad->getPointerOperandType(),
                                          ElementType, BadLoad->getAlign());
      if (!ByteVal) {
        // Fallback 3: peel a no-op spv_gep deref and retry against the
        // base pointer, which may have a richer deduced element type
        // (e.g. spv_resource_getpointer<VulkanBuffer<[0 x float]>>).
        if (Value *Base = peelNoOpSpvGep(Source))
          return buildLegalizedLoad(B, ElementType, Base, BadLoad);
        // Fallback 4: typed dynamic-index spv_gep into [K x ElemTy] where
        // ElemTy and ElementType have the same store size. Emit a typed
        // GEP+load+bitcast directly. Covers byte-stride GEPs (e.g.
        // `gep [4 x i8], ptr s_key, %idx`) into i32-array shared memory
        // where SPIRVEmitIntrinsics fell back to an i8-typed access chain.
        ByteVal = tryTypedDynamicGepLoad(
            B, Source, BadLoad->getPointerOperandType(), ElementType,
            BadLoad->getAlign());
        // Fallback 5: load from a typed aggregate's first primitive leaf and
        // narrow to the target type. Covers per-byte loads from LDS globals
        // typed as `[N x %struct.<vector>]` after memcpy lowering.
        if (!ByteVal)
          ByteVal = tryLoadFromAggregateLeaf(
              B, Source, BadLoad->getPointerOperandType(), ElementType,
              BadLoad->getAlign());
        // Fallback 5b: ConstantExpr byte-offset GEP into a typed global —
        // walk the aggregate's layout to find the leaf field at that byte
        // offset whose store size matches ElementType. Emits a typed
        // spv_gep + load. Covers HIP shared-memory globals typed as
        // [N x %struct.HIP_vector_type] read by `load double, ptr + i64 N`.
        if (!ByteVal)
          ByteVal = tryLoadFromConstantByteGep(
              B, Source, BadLoad->getPointerOperandType(), ElementType,
              BadLoad->getAlign());
        // Fallback 5c: deduced type is a struct of primitives whose total
        // store size matches ElementType — load each field, zext+shl, OR
        // together. Handles `load i64, ptr %p` where %p has deduced
        // %struct.Pair = {i32, i32}.
        if (!ByteVal)
          ByteVal = tryLoadStructAsScalarInt(
              B, Source, BadLoad->getPointerOperandType(), ElementType,
              BadLoad->getAlign());
        // Fallback 6: split a load whose Source is `select i1 c, ptrA, ptrB`
        // into two legalized loads (one per arm) plus a value-level select.
        // Logical SPIR-V cannot select between resource pointers, so we must
        // materialize both loads. Both arms must be dereferenceable, which
        // is the precondition for the original load (Source is dereferenced
        // unconditionally in the source IR). Emitted e.g. by ternary
        // expressions selecting between two device-global arrays bound as
        // distinct `[0 x i8]` Vulkan buffers.
        if (!ByteVal) {
          if (auto *Sel = dyn_cast<SelectInst>(Source)) {
            Value *TVal = buildLegalizedLoad(B, ElementType,
                                             Sel->getTrueValue(), BadLoad);
            Value *FVal = buildLegalizedLoad(B, ElementType,
                                             Sel->getFalseValue(), BadLoad);
            ByteVal = B.CreateSelect(Sel->getCondition(), TVal, FVal);
            buildAssignType(B, ElementType, ByteVal);
          }
        }
      }
      assert(ByteVal &&
             "Failed to load from aggregate: Could not find compatible "
             "memory layout (and no fallback applicable).");
      return ByteVal;
    }
    auto [GEP, CurrentTy] = *ResultOpt;

    auto *SAT = dyn_cast<ArrayType>(CurrentTy);
    auto *SVT = dyn_cast<FixedVectorType>(CurrentTy);
    auto *DVT = dyn_cast<FixedVectorType>(ElementType);
    auto *MAT =
        SAT ? dyn_cast<FixedVectorType>(SAT->getElementType()) : nullptr;

    if (ElementType == CurrentTy) {
      LoadInst *LI = B.CreateLoad(ElementType, GEP);
      LI->setAlignment(BadLoad->getAlign());
      buildAssignType(B, ElementType, LI);
      return LI;
    }
    if (SVT && DVT)
      return loadVectorFromVector(B, SVT, DVT, GEP);
    if (SAT && DVT && SAT->getElementType() == DVT->getElementType())
      return loadVectorFromArray(B, DVT, GEP);
    if (MAT && DVT && MAT->getElementType() == DVT->getElementType())
      return loadVectorFromMatrixArray(B, DVT, GEP, MAT);

    llvm_unreachable("Failed to load from aggregate.");
  }
  Value *
  buildVectorFromLoadedElements(IRBuilder<> &B, FixedVectorType *TargetType,
                                SmallVector<Value *, 4> &LoadedElements) {
    // <1 x T> shares the SPIR-V type with T, so emitting OpCompositeInsert on
    // a scalar would be invalid. Bridge with spv_bitcast instead.
    if (TargetType->getNumElements() == 1) {
      Value *Scalar = LoadedElements[0];
      Value *NewVector = B.CreateIntrinsic(
          Intrinsic::spv_bitcast, {TargetType, Scalar->getType()}, {Scalar});
      buildAssignType(B, TargetType, NewVector);
      return NewVector;
    }

    // Build the vector from the loaded elements.
    Value *NewVector = PoisonValue::get(TargetType);
    buildAssignType(B, TargetType, NewVector);

    for (unsigned I = 0, E = TargetType->getNumElements(); I < E; ++I) {
      Value *Index = B.getInt32(I);
      SmallVector<Type *, 4> Types = {TargetType, TargetType,
                                      TargetType->getElementType(),
                                      Index->getType()};
      SmallVector<Value *> Args = {NewVector, LoadedElements[I], Index};
      NewVector = B.CreateIntrinsic(Intrinsic::spv_insertelt, {Types}, {Args});
      buildAssignType(B, TargetType, NewVector);
    }
    return NewVector;
  }

  // Loads elements from a matrix with an array of vector memory layout and
  // constructs a vector.
  Value *loadVectorFromMatrixArray(IRBuilder<> &B, FixedVectorType *TargetType,
                                   Value *Source,
                                   FixedVectorType *ArrElemVecTy) {
    Type *TargetElemTy = TargetType->getElementType();
    unsigned ScalarsPerArrayElement = ArrElemVecTy->getNumElements();
    // Load each element of the array.
    SmallVector<Value *, 4> LoadedElements;
    std::array<Type *, 2> Types = {Source->getType(), Source->getType()};
    for (unsigned I = 0, E = TargetType->getNumElements(); I < E; ++I) {
      unsigned ArrayIndex = I / ScalarsPerArrayElement;
      unsigned ElementIndexInArrayElem = I % ScalarsPerArrayElement;
      // Create a GEP to access the i-th element of the array.
      std::array<Value *, 4> Args = {
          B.getInt1(/*Inbounds=*/false), Source, B.getInt32(0),
          ConstantInt::get(B.getInt32Ty(), ArrayIndex)};
      auto *ElementPtr = B.CreateIntrinsic(Intrinsic::spv_gep, {Types}, {Args});
      GR->buildAssignPtr(B, ArrElemVecTy, ElementPtr);
      Value *LoadVec = B.CreateLoad(ArrElemVecTy, ElementPtr);
      buildAssignType(B, ArrElemVecTy, LoadVec);
      LoadedElements.push_back(makeExtractElement(B, TargetElemTy, LoadVec,
                                                  ElementIndexInArrayElem));
    }
    return buildVectorFromLoadedElements(B, TargetType, LoadedElements);
  }
  // Loads elements from an array and constructs a vector.
  Value *loadVectorFromArray(IRBuilder<> &B, FixedVectorType *TargetType,
                             Value *Source) {
    // Load each element of the array.
    SmallVector<Value *, 4> LoadedElements;
    std::array<Type *, 2> Types = {Source->getType(), Source->getType()};
    for (unsigned I = 0, E = TargetType->getNumElements(); I < E; ++I) {
      // Create a GEP to access the i-th element of the array.
      std::array<Value *, 4> Args = {B.getInt1(/*Inbounds=*/false), Source,
                                     B.getInt32(0),
                                     ConstantInt::get(B.getInt32Ty(), I)};
      auto *ElementPtr = B.CreateIntrinsic(Intrinsic::spv_gep, {Types}, {Args});
      GR->buildAssignPtr(B, TargetType->getElementType(), ElementPtr);

      // Load the value from the element pointer.
      Value *Load = B.CreateLoad(TargetType->getElementType(), ElementPtr);
      buildAssignType(B, TargetType->getElementType(), Load);
      LoadedElements.push_back(Load);
    }
    return buildVectorFromLoadedElements(B, TargetType, LoadedElements);
  }

  // Stores elements from a vector into a matrix (an array of vectors).
  void storeMatrixArrayFromVector(IRBuilder<> &B, Value *SrcVector,
                                  Value *DstArrayPtr, ArrayType *ArrTy,
                                  Align Alignment) {
    auto *SrcVecTy = cast<FixedVectorType>(SrcVector->getType());
    auto *ArrElemVecTy = cast<FixedVectorType>(ArrTy->getElementType());
    Type *ElemTy = ArrElemVecTy->getElementType();
    unsigned ScalarsPerArrayElement = ArrElemVecTy->getNumElements();
    unsigned SrcNumElements = SrcVecTy->getNumElements();
    assert(
        SrcNumElements % ScalarsPerArrayElement == 0 &&
        "Source vector size must be a multiple of array element vector size");

    std::array<Type *, 2> Types = {DstArrayPtr->getType(),
                                   DstArrayPtr->getType()};

    for (unsigned I = 0; I < SrcNumElements; I += ScalarsPerArrayElement) {
      unsigned ArrayIndex = I / ScalarsPerArrayElement;
      // Create a GEP to access the array element.
      std::array<Value *, 4> Args = {
          B.getInt1(/*Inbounds=*/false), DstArrayPtr, B.getInt32(0),
          ConstantInt::get(B.getInt32Ty(), ArrayIndex)};
      auto *ElementPtr = B.CreateIntrinsic(Intrinsic::spv_gep, {Types}, {Args});
      GR->buildAssignPtr(B, ArrElemVecTy, ElementPtr);

      // Extract scalar elements from the source vector for this array slot.
      SmallVector<Value *, 4> Elements;
      for (unsigned J = 0; J < ScalarsPerArrayElement; ++J)
        Elements.push_back(makeExtractElement(B, ElemTy, SrcVector, I + J));

      // Build a vector from the extracted elements and store it.
      Value *Vec = buildVectorFromLoadedElements(B, ArrElemVecTy, Elements);
      StoreInst *SI = B.CreateStore(Vec, ElementPtr);
      SI->setAlignment(Alignment);
    }
  }

  // Stores elements from a vector into an array.
  void storeArrayFromVector(IRBuilder<> &B, Value *SrcVector,
                            Value *DstArrayPtr, ArrayType *ArrTy,
                            Align Alignment) {
    auto *VecTy = cast<FixedVectorType>(SrcVector->getType());
    Type *ElemTy = ArrTy->getElementType();

    // Ensure the element types of the array and vector are the same.
    assert(VecTy->getElementType() == ElemTy &&
           "Element types of array and vector must be the same.");
    std::array<Type *, 2> Types = {DstArrayPtr->getType(),
                                   DstArrayPtr->getType()};

    for (unsigned I = 0, E = VecTy->getNumElements(); I < E; ++I) {
      // Create a GEP to access the i-th element of the array.
      std::array<Value *, 4> Args = {B.getInt1(/*Inbounds=*/false), DstArrayPtr,
                                     B.getInt32(0),
                                     ConstantInt::get(B.getInt32Ty(), I)};
      auto *ElementPtr = B.CreateIntrinsic(Intrinsic::spv_gep, {Types}, {Args});
      GR->buildAssignPtr(B, ElemTy, ElementPtr);

      // Extract the element from the vector and store it.
      Value *Element =
          E == 1 ? SrcVector : makeExtractElement(B, ElemTy, SrcVector, I);
      StoreInst *SI = B.CreateStore(Element, ElementPtr);
      SI->setAlignment(Alignment);
    }
  }

  // Replaces the load instruction to get rid of the ptrcast used as source
  // operand.
  void transformLoad(IRBuilder<> &B, LoadInst *LI, Value *CastedOperand,
                     Value *OriginalOperand) {
    Type *ToTy = GR->findDeducedElementType(CastedOperand);
    B.SetInsertPoint(LI);

    Value *Output = buildLegalizedLoad(B, ToTy, OriginalOperand, LI);

    GR->replaceAllUsesWith(LI, Output, /* DeleteOld= */ true);
    DeadInstructions.push_back(LI);
  }

  // Creates an spv_insertelt instruction (equivalent to llvm's insertelement).
  Value *makeInsertElement(IRBuilder<> &B, Value *Vector, Value *Element,
                           unsigned Index) {
    Type *Int32Ty = Type::getInt32Ty(B.getContext());
    SmallVector<Type *, 4> Types = {Vector->getType(), Vector->getType(),
                                    Element->getType(), Int32Ty};
    SmallVector<Value *> Args = {Vector, Element, B.getInt32(Index)};
    Value *NewI = B.CreateIntrinsic(Intrinsic::spv_insertelt, {Types}, {Args});
    buildAssignType(B, Vector->getType(), NewI);
    return NewI;
  }

  // Creates an spv_extractelt instruction (equivalent to llvm's
  // extractelement).
  Value *makeExtractElement(IRBuilder<> &B, Type *ElementType, Value *Vector,
                            unsigned Index) {
    Type *Int32Ty = Type::getInt32Ty(B.getContext());
    SmallVector<Type *, 3> Types = {ElementType, Vector->getType(), Int32Ty};
    SmallVector<Value *> Args = {Vector, B.getInt32(Index)};
    Value *NewI = B.CreateIntrinsic(Intrinsic::spv_extractelt, {Types}, {Args});
    buildAssignType(B, ElementType, NewI);
    return NewI;
  }

  // Stores the given Src vector operand into the Dst vector, adjusting the size
  // if required.
  Value *storeVectorFromVector(IRBuilder<> &B, Value *Src, Value *Dst,
                               Align Alignment) {
    FixedVectorType *SrcType = cast<FixedVectorType>(Src->getType());
    FixedVectorType *DstType =
        cast<FixedVectorType>(GR->findDeducedElementType(Dst));
    auto dstNumElements = DstType->getNumElements();
    auto srcNumElements = SrcType->getNumElements();

    // if the element type differs, it is a bitcast.
    if (DstType->getElementType() != SrcType->getElementType()) {
      // Support bitcast between vectors of different sizes only if
      // the total bitwidth is the same.
      [[maybe_unused]] auto dstBitWidth =
          DstType->getElementType()->getScalarSizeInBits() * dstNumElements;
      [[maybe_unused]] auto srcBitWidth =
          SrcType->getElementType()->getScalarSizeInBits() * srcNumElements;
      assert(dstBitWidth == srcBitWidth &&
             "Unsupported bitcast between vectors of different sizes.");

      Src =
          B.CreateIntrinsic(Intrinsic::spv_bitcast, {DstType, SrcType}, {Src});
      buildAssignType(B, DstType, Src);
      SrcType = DstType;

      StoreInst *SI = B.CreateStore(Src, Dst);
      SI->setAlignment(Alignment);
      return SI;
    }

    assert(DstType->getNumElements() >= SrcType->getNumElements());
    LoadInst *LI = B.CreateLoad(DstType, Dst);
    LI->setAlignment(Alignment);
    Value *OldValues = LI;
    buildAssignType(B, OldValues->getType(), OldValues);
    Value *NewValues = Src;

    for (unsigned I = 0; I < SrcType->getNumElements(); ++I) {
      Value *Element =
          makeExtractElement(B, SrcType->getElementType(), NewValues, I);
      OldValues = makeInsertElement(B, OldValues, Element, I);
    }

    StoreInst *SI = B.CreateStore(OldValues, Dst);
    SI->setAlignment(Alignment);
    return SI;
  }

  // Builds a legalized store to a pointer, drilling down through
  // memory layouts to find a compatible type.
  void buildLegalizedStore(IRBuilder<> &B, Value *Src, Value *Dst,
                           Align Alignment) {
    // Store to undef/poison is undefined behaviour; drop it.
    if (isa<UndefValue>(Dst) || isa<PoisonValue>(Dst))
      return;
    auto ResultOpt = getPointerToFirstCompatibleType(B, Dst, Dst->getType(),
                                                     Src->getType(), true);
    if (!ResultOpt) {
      // Fallback 1: same-size leaf (bitcast + store).
      auto LeafOpt = getPointerToFirstSameSizeLeaf(
          B, Dst, Dst->getType(), Src->getType(), true);
      if (LeafOpt) {
        auto [LeafGEP, LeafTy] = *LeafOpt;
        Value *StoredVal = Src;
        if (LeafTy != Src->getType()) {
          StoredVal = B.CreateIntrinsic(Intrinsic::spv_bitcast,
                                        {LeafTy, Src->getType()}, {Src});
          buildAssignType(B, LeafTy, StoredVal);
        }
        StoreInst *SI = B.CreateStore(StoredVal, LeafGEP);
        SI->setAlignment(Alignment);
        return;
      }
      // Fallback 2: store as bytes into deduced [N x i8].
      bool Done =
          tryStoreAsByteArray(B, Src, Dst, Dst->getType(), Alignment);
      if (!Done) {
        // Fallback 3: peel a no-op spv_gep deref and retry against the
        // base pointer, which may have a richer deduced element type.
        if (Value *Base = peelNoOpSpvGep(Dst)) {
          buildLegalizedStore(B, Src, Base, Alignment);
          return;
        }
        // Fallback 4: typed dynamic-index spv_gep into [K x ElemTy] where
        // ElemTy has the same store size as Src. Emit a typed GEP+store
        // (with a leading bitcast if needed). Covers byte-stride GEPs into
        // typed arrays in shared memory.
        Done = tryTypedDynamicGepStore(B, Src, Dst, Dst->getType(), Alignment);
      }
      if (!Done) {
        // Fallback 5: narrow-int store into typed aggregate via read-modify-
        // write of the first primitive leaf. Handles byte-stride memcpy into
        // `ptr addrspace(4) (deduced %struct.X)` patterns.
        Done = tryStoreToAggregateLeaf(B, Src, Dst, Dst->getType(), Alignment);
      }
      if (!Done) {
        // Fallback 6: scalar-int store into a struct of primitives whose
        // total size matches. Splits the source int across fields by offset.
        Done =
            tryStoreScalarIntAsStruct(B, Src, Dst, Dst->getType(), Alignment);
      }
      if (!Done) {
        // Fallback 7: ConstantExpr byte-offset GEP into a typed global.
        // Mirror of tryLoadFromConstantByteGep — descends the global's
        // layout by byte offset and emits a typed spv_gep + store.
        Done =
            tryStoreToConstantByteGep(B, Src, Dst, Dst->getType(), Alignment);
      }
      assert(Done &&
             "Failed to store to aggregate: Could not find compatible "
             "memory layout (and no fallback applicable).");
      return;
    }
    auto [GEP, CurrentTy] = *ResultOpt;

    auto *DAT = dyn_cast<ArrayType>(CurrentTy);
    auto *DVT = dyn_cast<FixedVectorType>(CurrentTy);
    auto *SVT = dyn_cast<FixedVectorType>(Src->getType());
    auto *DMAT =
        DAT ? dyn_cast<FixedVectorType>(DAT->getElementType()) : nullptr;

    if (Src->getType() == CurrentTy) {
      StoreInst *SI = B.CreateStore(Src, GEP);
      SI->setAlignment(Alignment);
      return;
    }
    if (DVT && SVT) {
      storeVectorFromVector(B, Src, GEP, Alignment);
      return;
    }
    if (DAT && SVT && SVT->getElementType() == DAT->getElementType()) {
      storeArrayFromVector(B, Src, GEP, DAT, Alignment);
      return;
    }
    if (DMAT && SVT && DMAT->getElementType() == SVT->getElementType()) {
      storeMatrixArrayFromVector(B, Src, GEP, DAT, Alignment);
      return;
    }

    llvm_unreachable("Failed to store to aggregate.");
  }

  // Transforms a store instruction (or SPV intrinsic) using a ptrcast as
  // operand into a valid logical SPIR-V store with no ptrcast.
  void transformStore(IRBuilder<> &B, Instruction *BadStore, Value *Src,
                      Value *Dst, Align Alignment) {
    B.SetInsertPoint(BadStore);
    buildLegalizedStore(B, Src, Dst, Alignment);
    DeadInstructions.push_back(BadStore);
  }

  void legalizePointerCast(IntrinsicInst *II) {
    Value *CastedOperand = II;
    Value *OriginalOperand = II->getOperand(0);

    IRBuilder<> B(II->getContext());
    std::vector<Value *> Users;
    for (Use &U : II->uses())
      Users.push_back(U.getUser());

    // PhysicalStorageBuffer (addrspace 5) pointers natively support arbitrary
    // byte-addressable access in SPIR-V (the BDA / Vulkan BDA model). When
    // SPIRVEmitIntrinsics inserts an spv_ptrcast for a load/store through a
    // PSB pointer (e.g. `load float, ptr addrspace(5) %byte_gep` where the
    // byte-stride GEP's deduced element type is i8), no legalization is
    // required: the OpLoad/OpStore emit directly with the value's element
    // type, matching how atomicrmw is handled below. Just rewire each
    // load/store user back to the original pointer and skip the rest of
    // legalization.
    bool IsPSB = false;
    if (auto *PT = dyn_cast<PointerType>(OriginalOperand->getType()))
      IsPSB = (PT->getAddressSpace() == 5);
    if (IsPSB) {
      for (Value *User : Users) {
        if (auto *LI = dyn_cast<LoadInst>(User)) {
          LI->setOperand(LoadInst::getPointerOperandIndex(), OriginalOperand);
          continue;
        }
        if (auto *SI = dyn_cast<StoreInst>(User)) {
          SI->setOperand(StoreInst::getPointerOperandIndex(), OriginalOperand);
          continue;
        }
        if (auto *ARMW = dyn_cast<AtomicRMWInst>(User)) {
          ARMW->setOperand(AtomicRMWInst::getPointerOperandIndex(),
                           OriginalOperand);
          continue;
        }
        if (auto *CX = dyn_cast<AtomicCmpXchgInst>(User)) {
          CX->setOperand(AtomicCmpXchgInst::getPointerOperandIndex(),
                         OriginalOperand);
          continue;
        }
        if (auto *Intrin = dyn_cast<IntrinsicInst>(User)) {
          if (Intrin->getIntrinsicID() == Intrinsic::spv_assign_ptr_type) {
            DeadInstructions.push_back(Intrin);
            continue;
          }
          if (Intrin->getIntrinsicID() == Intrinsic::spv_gep) {
            GR->replaceAllUsesWith(CastedOperand, OriginalOperand,
                                   /* DeleteOld= */ false);
            continue;
          }
          if (Intrin->getIntrinsicID() == Intrinsic::spv_store) {
            // spv_store args: (val, ptr, isVolatile, align). Rewire ptr.
            Intrin->setArgOperand(1, OriginalOperand);
            continue;
          }
        }
        // Anything else: fall back to the general path below.
      }
      DeadInstructions.push_back(II);
      return;
    }

    for (Value *User : Users) {
      if (LoadInst *LI = dyn_cast<LoadInst>(User)) {
        transformLoad(B, LI, CastedOperand, OriginalOperand);
        continue;
      }

      if (StoreInst *SI = dyn_cast<StoreInst>(User)) {
        transformStore(B, SI, SI->getValueOperand(), OriginalOperand,
                       SI->getAlign());
        continue;
      }

      // atomicrmw / cmpxchg: keep the instruction but rewire the pointer
      // operand to the original (pre-ptrcast) pointer. The element type for
      // type deduction is determined from the value operand by
      // SPIRVEmitIntrinsics, so dropping the spv_ptrcast here is safe.
      if (auto *ARMW = dyn_cast<AtomicRMWInst>(User)) {
        ARMW->setOperand(AtomicRMWInst::getPointerOperandIndex(),
                         OriginalOperand);
        continue;
      }
      if (auto *CX = dyn_cast<AtomicCmpXchgInst>(User)) {
        CX->setOperand(AtomicCmpXchgInst::getPointerOperandIndex(),
                       OriginalOperand);
        continue;
      }

      if (IntrinsicInst *Intrin = dyn_cast<IntrinsicInst>(User)) {
        if (Intrin->getIntrinsicID() == Intrinsic::spv_assign_ptr_type) {
          DeadInstructions.push_back(Intrin);
          continue;
        }

        if (Intrin->getIntrinsicID() == Intrinsic::spv_gep) {
          GR->replaceAllUsesWith(CastedOperand, OriginalOperand,
                                 /* DeleteOld= */ false);
          continue;
        }

        if (Intrin->getIntrinsicID() == Intrinsic::spv_store) {
          Align Alignment;
          if (ConstantInt *C = dyn_cast<ConstantInt>(Intrin->getOperand(3)))
            Alignment = Align(C->getZExtValue());
          // Phase H7: spv_load of an aggregate yields an i32 placeholder whose
          // semantic type is the aggregate (recorded via spv_assign_type
          // metadata). When the matching spv_store targets a pointer whose
          // deduced element type is the same aggregate, the legaliser's
          // drill-down logic can't find an i32 leaf at offset 0 and asserts.
          // Detect that pattern and just rewire the store to the original
          // (un-ptrcast) pointer; the backend selector handles the
          // aggregate-via-i32-placeholder OpLoad/OpStore pair natively.
          Value *StoredVal = Intrin->getArgOperand(0);
          Type *DstAggrTy = GR->findDeducedElementType(OriginalOperand);
          if (DstAggrTy && DstAggrTy->isAggregateType()) {
            if (auto *SrcCI = dyn_cast<IntrinsicInst>(StoredVal)) {
              if (SrcCI->getIntrinsicID() == Intrinsic::spv_load) {
                Type *SrcAggrTy = GR->findDeducedElementType(
                    SrcCI->getArgOperand(0));
                if (SrcAggrTy == DstAggrTy) {
                  Intrin->setArgOperand(1, OriginalOperand);
                  continue;
                }
              }
            }
          }
          transformStore(B, Intrin, StoredVal, OriginalOperand, Alignment);
          continue;
        }
      }

      // Generic call site (non-intrinsic / non-recognized intrinsic): rewire
      // every operand position that uses the casted pointer to the original
      // pointer. The callee's parameter type drives type deduction in
      // SPIRVEmitIntrinsics, so dropping the spv_ptrcast at the call boundary
      // is safe — equivalent to the atomicrmw / cmpxchg handling above.
      if (auto *CB = dyn_cast<CallBase>(User)) {
        for (unsigned I = 0, E = CB->arg_size(); I < E; ++I)
          if (CB->getArgOperand(I) == CastedOperand)
            CB->setArgOperand(I, OriginalOperand);
        continue;
      }

      // SelectInst / PHINode / ICmp / ptrtoint: pointer arithmetic /
      // control-flow merge / comparison nodes. Rewire each operand position
      // referencing the cast to the original pointer. The cast doesn't
      // change the byte address, and the merged/compared pointer's element
      // type is recovered downstream by SPIRVEmitIntrinsics from the
      // load/store/atomic context that consumes the merge result.
      if (isa<SelectInst>(User) || isa<PHINode>(User) || isa<CmpInst>(User) ||
          isa<PtrToIntInst>(User)) {
        auto *UI = cast<Instruction>(User);
        for (unsigned I = 0, E = UI->getNumOperands(); I < E; ++I)
          if (UI->getOperand(I) == CastedOperand)
            UI->setOperand(I, OriginalOperand);
        continue;
      }

      llvm_unreachable("Unsupported ptrcast user. Please fix.");
    }

    DeadInstructions.push_back(II);
  }

public:
  SPIRVLegalizePointerCastImpl(const SPIRVTargetMachine &TM) : TM(TM) {}

  bool run(Function &F) {
    const SPIRVSubtarget &ST = TM.getSubtarget<SPIRVSubtarget>(F);
    GR = ST.getSPIRVGlobalRegistry();
    DeadInstructions.clear();

    std::vector<IntrinsicInst *> WorkList;
    for (auto &BB : F) {
      for (auto &I : BB) {
        auto *II = dyn_cast<IntrinsicInst>(&I);
        if (II && II->getIntrinsicID() == Intrinsic::spv_ptrcast)
          WorkList.push_back(II);
      }
    }

    for (IntrinsicInst *II : WorkList)
      legalizePointerCast(II);

    for (Instruction *I : DeadInstructions)
      I->eraseFromParent();

    return DeadInstructions.size() != 0;
  }

private:
  const SPIRVTargetMachine &TM;
  SPIRVGlobalRegistry *GR = nullptr;
  std::vector<Instruction *> DeadInstructions;
};

class SPIRVLegalizePointerCastLegacy : public FunctionPass {
public:
  static char ID;
  SPIRVLegalizePointerCastLegacy(const SPIRVTargetMachine &TM)
      : FunctionPass(ID), TM(TM) {}

  bool runOnFunction(Function &F) override {
    return SPIRVLegalizePointerCastImpl(TM).run(F);
  }

private:
  const SPIRVTargetMachine &TM;
};
} // namespace

PreservedAnalyses SPIRVLegalizePointerCast::run(Function &F,
                                                FunctionAnalysisManager &AM) {
  return SPIRVLegalizePointerCastImpl(TM).run(F) ? PreservedAnalyses::none()
                                                 : PreservedAnalyses::all();
}

char SPIRVLegalizePointerCastLegacy::ID = 0;
INITIALIZE_PASS(SPIRVLegalizePointerCastLegacy, "spirv-legalize-pointer-cast",
                "SPIRV legalize pointer cast pass", false, false)

FunctionPass *llvm::createSPIRVLegalizePointerCastPass(SPIRVTargetMachine *TM) {
  return new SPIRVLegalizePointerCastLegacy(*TM);
}
