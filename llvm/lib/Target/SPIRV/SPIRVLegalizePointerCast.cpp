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
                     Type *&OutBaseTy) {
    auto *II = dyn_cast<IntrinsicInst>(Source);
    if (!II || II->getIntrinsicID() != Intrinsic::spv_gep)
      return false;
    // spv_gep args: (i1 inbounds, ptr base, i32 0 [, indices...]).
    if (II->arg_size() < 4)
      return false;
    auto *LastIdx = dyn_cast<ConstantInt>(II->getArgOperand(II->arg_size() - 1));
    if (!LastIdx)
      return false;
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
    OutBaseOffset = LastIdx->getZExtValue();
    OutBaseTy = BaseTy;
    return true;
  }

  Value *tryLoadAsByteArray(IRBuilder<> &B, Value *Source, Type *PointerType,
                            Type *TargetType, Align Alignment) {
    Type *I8Ty = Type::getInt8Ty(B.getContext());
    // First: if Source is an spv_gep into [N x i8] with a constant byte
    // offset, emit per-byte spv_geps from the underlying base.
    Value *Base = nullptr;
    uint64_t BaseOffset = 0;
    Type *BaseTy = nullptr;
    if (unwrapByteGep(Source, Base, BaseOffset, BaseTy) &&
        TargetType->isSized()) {
      const DataLayout &DL = B.GetInsertBlock()->getModule()->getDataLayout();
      unsigned NumBytes = DL.getTypeStoreSize(TargetType).getFixedValue();
      if (NumBytes == 0)
        return nullptr;
      SmallVector<Value *, 4> Bytes;
      std::array<Type *, 2> GepTypes = {PointerType, Base->getType()};
      for (unsigned I = 0; I < NumBytes; ++I) {
        SmallVector<Value *, 4> Args = {B.getInt1(false), Base, B.getInt32(0),
                                        B.getInt32(BaseOffset + I)};
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
    if (SrcTy->isSized() &&
        unwrapByteGep(Dst, Base, BaseOffset, BaseTy)) {
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
        SmallVector<Value *, 4> Args = {B.getInt1(true), Base, B.getInt32(0),
                                        B.getInt32(BaseOffset + I)};
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
  Value *peelNoOpSpvGep(Value *Source) {
    auto *II = dyn_cast<IntrinsicInst>(Source);
    if (!II || II->getIntrinsicID() != Intrinsic::spv_gep)
      return nullptr;
    if (II->arg_size() != 3)
      return nullptr;
    auto *CI0 = dyn_cast<ConstantInt>(II->getArgOperand(2));
    if (!CI0 || !CI0->isZero())
      return nullptr;
    Value *Base = II->getArgOperand(1);
    Type *SrcTy = GR->findDeducedElementType(Source);
    Type *BaseTy = GR->findDeducedElementType(Base);
    if (!BaseTy || BaseTy == SrcTy)
      return nullptr;
    return Base;
  }

  // Builds a legalized load from a pointer, drilling down through
  // memory layouts to find a compatible type. Load flags will be
  // copied from |BadLoad|, which should be the load being legalized.
  Value *buildLegalizedLoad(IRBuilder<> &B, Type *ElementType, Value *Source,
                            LoadInst *BadLoad) {
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
          transformStore(B, Intrin, Intrin->getArgOperand(0), OriginalOperand,
                         Alignment);
          continue;
        }
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
