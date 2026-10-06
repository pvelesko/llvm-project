//===- HIPSPVLowerToHLSLShape.cpp - HIP kernels to Vulkan shaders ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// For each spir_kernel function this pass creates a compute entry point
// (hlsl.shader="compute") without parameters:
// - a pointer argument becomes a StorageBuffer resource at descriptor set 0,
//   bound by the runtime to the buffer it points into; every pointer derived
//   from it must be a whole number of elements from it, and each load/store
//   goes through llvm.spv.resource.getpointer;
// - each argument becomes a field of one push constant block: a scalar its
//   value (a bool as i8), a struct passed by value its contents (copied to a
//   local the kernel may write), a pointer its byte offset into its buffer
//   (an i64);
// - OpenCL work-item builtins become the matching llvm.spv intrinsics.
// The work-group size of kernels without reqd_work_group_size, and
// get_local_size outside the body of a kernel with it, are spec constants 0-2:
// the runtime must specialize them with the block size of every launch,
// including launches of reqd_work_group_size kernels.
//
//===----------------------------------------------------------------------===//

#include "SPIRV.h"
#include "SPIRVBuiltins.h"
#include "SPIRVUtils.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/IR/DiagnosticInfo.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicsSPIRV.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Operator.h"
#include "llvm/IR/PatternMatch.h"
#include "llvm/Pass.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/KnownBits.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"

using namespace llvm;
using namespace llvm::PatternMatch;

static cl::opt<bool> DeferUnsupported(
    "hipspv-defer-unsupported",
    cl::desc("Report kernels that cannot be lowered as warnings, and in an "
             "entry point named \"__hipspv_error: <message>\" for the runtime "
             "to report when it loads the module"),
    cl::init(false));

namespace llvm::SPIRV {
#define GET_BuiltinGroup_DECL
#include "SPIRVGenTables.inc"
} // namespace llvm::SPIRV

namespace {
struct WorkItemBuiltin {
  StringLiteral Name;
  Intrinsic::ID ID;
  // Result for a dimension other than 0, 1 or 2.
  uint64_t OutOfRange;
};

} // namespace

// OpenCL C work-item functions as mangled by Clang (uint dimension argument).
static constexpr WorkItemBuiltin WorkItemBuiltins[] = {
    {"_Z12get_local_idj", Intrinsic::spv_thread_id_in_group, 0},
    {"_Z13get_global_idj", Intrinsic::not_intrinsic, 0},
    {"_Z12get_group_idj", Intrinsic::spv_group_id, 0},
    {"_Z14get_num_groupsj", Intrinsic::spv_num_workgroups, 1},
    // Spec constants 0-2, see lowerKernel.
    {"_Z14get_local_sizej", Intrinsic::not_intrinsic, 1},
};

// Work-group size dimension D as seen from CI's function.
static Value *localSize(IRBuilder<> &B, CallInst *CI, unsigned D) {
  if (MDNode *N = CI->getFunction()->getMetadata("reqd_work_group_size");
      N && D < N->getNumOperands())
    return B.getInt32(
        mdconst::extract<ConstantInt>(N->getOperand(D))->getZExtValue());
  FunctionCallee SpecConstant = CI->getModule()->getOrInsertFunction(
      "_Z20__spirv_SpecConstantii", B.getInt32Ty(), B.getInt32Ty(),
      B.getInt32Ty());
  return B.CreateCall(SpecConstant, {B.getInt32(D), B.getInt32(1)});
}

// Global ID dimension D in 64 bits, which GlobalInvocationId is not.
static Value *globalId(IRBuilder<> &B, CallInst *CI, unsigned D) {
  Type *I64 = B.getInt64Ty();
  Value *Group =
      B.CreateZExt(B.CreateIntrinsic(B.getInt32Ty(), Intrinsic::spv_group_id,
                                     {B.getInt32(D)}),
                   I64);
  Value *Size = B.CreateZExt(localSize(B, CI, D), I64);
  Value *Local = B.CreateZExt(
      B.CreateIntrinsic(B.getInt32Ty(), Intrinsic::spv_thread_id_in_group,
                        {B.getInt32(D)}),
      I64);
  Value *Base = B.CreateMul(Group, Size);
  return B.CreateAdd(Base, Local);
}

// Replaces calls to the work-item builtins with a constant dimension. The
// builtins return size_t; the Vulkan builtins are 32-bit.
static bool lowerWorkItemBuiltins(Module &M) {
  bool Changed = false;
  for (const WorkItemBuiltin &WB : WorkItemBuiltins) {
    Function *F = M.getFunction(WB.Name);
    if (!F || !F->isDeclaration() || F->arg_size() != 1 ||
        !F->getArg(0)->getType()->isIntegerTy(32) ||
        !F->getReturnType()->isIntegerTy())
      continue;
    for (User *U : make_early_inc_range(F->users())) {
      auto *CI = dyn_cast<CallInst>(U);
      if (!CI || CI->getCalledFunction() != F)
        continue;
      auto *Dim = dyn_cast<ConstantInt>(CI->getArgOperand(0));
      if (!Dim)
        continue;
      IRBuilder<> B(CI);
      unsigned D = Dim->getZExtValue();
      Value *V;
      if (D > 2)
        V = ConstantInt::get(CI->getType(), WB.OutOfRange);
      else if (WB.ID != Intrinsic::not_intrinsic)
        V = B.CreateIntrinsic(B.getInt32Ty(), WB.ID, {B.getInt32(D)});
      else if (WB.Name == "_Z14get_local_sizej")
        V = localSize(B, CI, D);
      else
        V = globalId(B, CI, D);
      V = B.CreateZExtOrTrunc(V, CI->getType());
      CI->replaceAllUsesWith(V);
      CI->eraseFromParent();
      Changed = true;
    }
  }
  return Changed;
}

// 8- to 64-bit integers, half, float, double, and vectors of up to four of
// them: the types a shader stores at their own size.
static bool isPlainDataTy(Type *T) {
  auto *VT = dyn_cast<VectorType>(T);
  if (VT && (isa<ScalableVectorType>(VT) ||
             cast<FixedVectorType>(VT)->getNumElements() > 4))
    return false;
  Type *S = T->getScalarType();
  if (S->isIntegerTy())
    return S->getIntegerBitWidth() >= 8 && S->getIntegerBitWidth() <= 64 &&
           isPowerOf2_32(S->getIntegerBitWidth());
  return S->isHalfTy() || S->isFloatTy() || S->isDoubleTy();
}

// Structs and arrays of plain data, as passed by value.
static bool isPlainAggregate(Type *T) {
  if (auto *AT = dyn_cast<ArrayType>(T))
    return isPlainDataTy(AT->getElementType()) ||
           isPlainAggregate(AT->getElementType());
  auto *ST = dyn_cast<StructType>(T);
  return ST && !ST->isOpaque() && all_of(ST->elements(), [](Type *E) {
           return isPlainDataTy(E) || isPlainAggregate(E);
         });
}

// The number of plain-data leaves in T.
static uint64_t countLeaves(Type *T) {
  if (auto *AT = dyn_cast<ArrayType>(T))
    return AT->getNumElements() * countLeaves(AT->getElementType());
  if (auto *ST = dyn_cast<StructType>(T))
    return sum_of(map_range(ST->elements(), countLeaves));
  return 1;
}

// Copies the plain-data leaves of a T at Src to Dst.
static void copyLeaves(IRBuilder<> &B, Type *T, Value *Src, Value *Dst,
                       SmallVectorImpl<Value *> &Idx, Type *RootTy) {
  if (isPlainDataTy(T)) {
    B.CreateStore(B.CreateLoad(T, B.CreateInBoundsGEP(RootTy, Src, Idx)),
                  B.CreateInBoundsGEP(RootTy, Dst, Idx));
    return;
  }
  uint64_t Leaves = countLeaves(T);
  if (Leaves == 0)
    return;
  // A long array is copied by a loop: drivers can take minutes on the unroll.
  if (isa<ArrayType>(T) && T->getArrayNumElements() > 1 && Leaves > 128) {
    Instruction *Next = &*B.GetInsertPoint();
    auto [Body, I] = SplitBlockAndInsertSimpleForLoop(
        B.getInt32(T->getArrayNumElements()), Next->getIterator());
    IRBuilder<> LB(Body);
    Idx.push_back(I);
    copyLeaves(LB, T->getArrayElementType(), Src, Dst, Idx, RootTy);
    Idx.pop_back();
    B.SetInsertPoint(Next);
    return;
  }
  unsigned N = isa<ArrayType>(T) ? T->getArrayNumElements()
                                 : T->getStructNumElements();
  for (unsigned I = 0; I < N; ++I) {
    Idx.push_back(B.getInt32(I));
    copyLeaves(B, isa<ArrayType>(T) ? T->getArrayElementType()
                                    : T->getStructElementType(I),
               Src, Dst, Idx, RootTy);
    Idx.pop_back();
  }
}

// The pointer operand of a memory access, or ~0u for other instructions.
static unsigned pointerOperandIndex(const Instruction *I) {
  if (isa<StoreInst>(I))
    return StoreInst::getPointerOperandIndex();
  return isa<LoadInst, AtomicRMWInst, AtomicCmpXchgInst>(I) ? 0 : ~0u;
}

// The alignment of a memory access.
static Align accessAlign(const Instruction *I) {
  if (auto *RMW = dyn_cast<AtomicRMWInst>(I))
    return RMW->getAlign();
  if (auto *CX = dyn_cast<AtomicCmpXchgInst>(I))
    return CX->getAlign();
  return getLoadStoreAlignment(I);
}

// The type a memory access reads or writes.
static Type *accessType(const Instruction *I) {
  if (auto *RMW = dyn_cast<AtomicRMWInst>(I))
    return RMW->getType();
  if (auto *CX = dyn_cast<AtomicCmpXchgInst>(I))
    return CX->getNewValOperand()->getType();
  return getLoadStoreType(I);
}

namespace {
// Rewrites the accesses through one pointer argument as accesses to an
// element of a storage buffer, viewed as an array of each access type.
class BufferArgRewriter {
  const DataLayout &DL;
  Argument &Arg;
  // Byte offset from Arg of every pointer derived from it, in the index type.
  DenseMap<Value *, Value *> Offset;
  SmallVector<Instruction *, 8> Accesses;
  SmallVector<ICmpInst *, 4> Compares;
  SmallSetVector<Instruction *, 8> Derived;

public:
  /// The last user collect() looked at: the one it rejected after a failure.
  Instruction *Unsupported = nullptr;

  /// The byte offset the runtime passes for a null pointer argument.
  static constexpr uint64_t NullArgOffset = 1ull << 63;

private:

  // The byte offset G adds, wrapping as the address does, or null.
  Value *gepByteOffset(GEPOperator *G, Type *IdxTy, Instruction *InsertPt) {
    SmallMapVector<Value *, APInt, 4> VarOffs;
    // Offsets wrap at the index width, which must be the argument's.
    unsigned Width = IdxTy->getIntegerBitWidth();
    APInt ConstOff(Width, 0);
    if (DL.getIndexTypeSizeInBits(G->getType()) != Width ||
        !G->collectOffset(DL, Width, VarOffs, ConstOff))
      return nullptr;
    IRBuilder<> B(InsertPt);
    Value *Off = nullptr;
    auto Add = [&](Value *X) { Off = Off ? B.CreateAdd(Off, X) : X; };
    for (auto &[V, Scale] : VarOffs) {
      Value *X = B.CreateSExtOrTrunc(V, IdxTy);
      Add(Scale.isOne() ? X : B.CreateMul(X, ConstantInt::get(IdxTy, Scale)));
    }
    if (!Off || !ConstOff.isZero())
      Add(ConstantInt::get(IdxTy, ConstOff));
    return Off;
  }

  // Collects the pointers derived from Arg and their accesses and comparisons.
  bool collect(Value *V) {
    for (Use &U : V->uses()) {
      auto *I = dyn_cast<Instruction>(U.getUser());
      if (!I)
        return false;
      Unsupported = I;
      if (isa<AddrSpaceCastInst, PHINode, SelectInst>(I) ||
          (isa<GetElementPtrInst>(I) &&
           cast<GetElementPtrInst>(I)->getPointerOperand() == V)) {
        if (Derived.insert(I) && !collect(I))
          return false;
      } else if (auto *PTI = dyn_cast<PtrToIntInst>(I);
                 PTI && PTI->getType()->getScalarSizeInBits() ==
                            DL.getPointerTypeSizeInBits(V->getType())) {
        // Pointer arguments arrive as inttoptr(ptrtoint(arg)) after coercion.
        if (!Derived.insert(PTI))
          continue;
        for (User *PU : PTI->users()) {
          auto *ITP = dyn_cast<IntToPtrInst>(PU);
          if (!ITP) {
            Unsupported = dyn_cast<Instruction>(PU);
            return false;
          }
          if (Derived.insert(ITP) && !collect(ITP))
            return false;
        }
      } else if (auto *Cmp = dyn_cast<ICmpInst>(I)) {
        if (!is_contained(Compares, Cmp))
          Compares.push_back(Cmp);
      } else if (U.getOperandNo() == pointerOperandIndex(I)) {
        Type *T = accessType(I);
        if (!isPlainDataTy(T) || !isPowerOf2_64(DL.getTypeAllocSize(T)))
          return false;
        Accesses.push_back(I);
      } else {
        return false;
      }
    }
    return true;
  }

  // The byte offset of V from Arg, or null if V is not derived from it.
  Value *offsetOf(Value *V, Type *IdxTy) {
    if (V == &Arg)
      return ConstantInt::get(IdxTy, 0);
    if (auto It = Offset.find(V); It != Offset.end())
      return It->second;
    auto *I = dyn_cast<Instruction>(V);
    if (!I || !Derived.contains(I))
      return nullptr;
    Value *Base = nullptr;
    if (auto *Phi = dyn_cast<PHINode>(I)) {
      auto *NP = PHINode::Create(IdxTy, Phi->getNumIncomingValues(), "",
                                 Phi->getIterator());
      Offset[V] = NP;
      for (auto [In, BB] : zip(Phi->incoming_values(), Phi->blocks())) {
        Value *O = offsetOf(In, IdxTy);
        if (!O)
          return Offset[V] = nullptr;
        NP->addIncoming(O, BB);
      }
      return NP;
    }
    if (auto *Sel = dyn_cast<SelectInst>(I)) {
      Value *T = offsetOf(Sel->getTrueValue(), IdxTy);
      Value *F = T ? offsetOf(Sel->getFalseValue(), IdxTy) : nullptr;
      if (F)
        Base = IRBuilder<>(Sel).CreateSelect(Sel->getCondition(), T, F);
    } else if (auto *ITP = dyn_cast<IntToPtrInst>(I)) {
      Base = offsetOf(cast<PtrToIntInst>(ITP->getOperand(0))->getOperand(0),
                      IdxTy);
    } else if (auto *ASC = dyn_cast<AddrSpaceCastInst>(I)) {
      Base = offsetOf(ASC->getPointerOperand(), IdxTy);
    } else if (auto *G = dyn_cast<GetElementPtrInst>(I)) {
      Value *Off = gepByteOffset(cast<GEPOperator>(G), IdxTy, G);
      Value *PO = Off ? offsetOf(G->getPointerOperand(), IdxTy) : nullptr;
      if (PO)
        Base = match(PO, m_Zero()) ? Off : IRBuilder<>(G).CreateAdd(PO, Off);
    }
    return Offset[V] = Base;
  }

  // The index of an element of ElemSize bytes at byte offset Off, or null if
  // Off is not provably a whole number of them. Wrapped intermediate offsets
  // are fine: an access in the buffer has an offset below the sign bit.
  Value *indexOf(Value *Off, uint64_t ElemSize, Instruction *InsertPt) {
    if (computeKnownBits(Off, DL).countMinTrailingZeros() < Log2_64(ElemSize))
      return nullptr;
    IRBuilder<> B(InsertPt);
    // Undo the scaling of a GEP index rather than shift its product back; at
    // 64 bits the product of an index into a buffer cannot wrap.
    Value *X;
    const APInt *C;
    if (Off->getType()->isIntegerTy(64) &&
        match(Off, m_Mul(m_Value(X), m_APInt(C))) && C->urem(ElemSize) == 0) {
      APInt Q = C->udiv(ElemSize);
      return Q.isOne() ? X : B.CreateMul(X, B.getInt(Q));
    }
    return B.CreateSExt(
        B.CreateAShr(Off, Log2_64(ElemSize), "", /*isExact=*/true),
        B.getInt64Ty());
  }

  // The comparison Cmp makes, on byte offsets, or null.
  Value *compareOffsets(ICmpInst *Cmp, Value *Base, Type *IdxTy) {
    Value *L = Cmp->getOperand(0), *R = Cmp->getOperand(1);
    if (isa<ConstantPointerNull>(L))
      std::swap(L, R);
    IRBuilder<> B(Cmp);
    Value *LO = offsetOf(L, IdxTy);
    if (!LO)
      return nullptr;
    if (isa<ConstantPointerNull>(R)) {
      if (!Cmp->isEquality())
        return nullptr;
      Value *Off = match(LO, m_Zero())
                       ? Base
                       : B.CreateAdd(Base, B.CreateSExtOrTrunc(LO, B.getInt64Ty()));
      return B.CreateICmp(Cmp->getPredicate(), Off, B.getInt64(NullArgOffset));
    }
    Value *RO = offsetOf(R, IdxTy);
    return RO ? B.CreateICmp(Cmp->getPredicate(), LO, RO) : nullptr;
  }

public:
  BufferArgRewriter(const DataLayout &DL, Argument &Arg) : DL(DL), Arg(Arg) {}

  // Replaces the accesses through Arg with accesses to the buffer at Binding,
  // whose byte offset the i64 Base holds. Returns false if they are not all
  // element accesses; the function may then contain dead index computations.
  bool run(Function &NewF, unsigned Binding, Instruction *Base) {
    if (!collect(&Arg))
      return false;
    Type *IdxTy = DL.getIndexType(Arg.getType());
    // Arg is aligned to its elements, so Base is a whole number of them.
    IRBuilder<> BB(Base->getNextNode());
    SmallDenseMap<uint64_t, Value *> BaseIdx;
    SmallVector<std::pair<Type *, Value *>, 8> Idx;
    for (Instruction *I : Accesses) {
      Type *T = accessType(I);
      uint64_t Size = DL.getTypeAllocSize(T);
      Unsupported = I;
      Value *Off = offsetOf(I->getOperand(pointerOperandIndex(I)), IdxTy);
      if (!Off)
        return false;
      if (Value *X = indexOf(Off, Size, I)) {
        Value *&BI = BaseIdx[Size];
        if (!BI)
          BI = BB.CreateLShr(Base, Log2_64(Size));
        Idx.emplace_back(
            T, match(X, m_Zero()) ? BI : IRBuilder<>(I).CreateAdd(BI, X));
      } else if (accessAlign(I) >= Size) {
        // The access is aligned, so a whole number of elements from the
        // buffer's start, if not provably from Arg.
        IRBuilder<> B(I);
        Idx.emplace_back(
            T, B.CreateLShr(B.CreateAdd(Base, B.CreateSExtOrTrunc(
                                                  Off, Base->getType())),
                            Log2_64(Size)));
      } else {
        return false;
      }
    }
    SmallVector<Value *, 4> Cmps;
    for (ICmpInst *Cmp : Compares) {
      Unsupported = Cmp;
      Value *New = compareOffsets(Cmp, Base, IdxTy);
      if (!New)
        return false;
      Cmps.push_back(New);
    }
    LLVMContext &C = Arg.getContext();
    BasicBlock &Entry = NewF.getEntryBlock();
    IRBuilder<> B(&Entry, Entry.getFirstInsertionPt());
    // One view of the buffer per access type, named
    // "<kernel>.<argument number>[~<view>][.<argument name>]"; an unused
    // argument still binds.
    SmallDenseMap<Type *, Value *> Handles;
    auto HandleFor = [&](Type *T) {
      Value *&H = Handles[T];
      if (H)
        return H;
      std::string Name = (NewF.getName() + "." + Twine(Arg.getArgNo())).str();
      if (unsigned View = Handles.size() - 1)
        Name += "~" + std::to_string(View);
      if (Arg.hasName())
        Name += ("." + Arg.getName()).str();
      auto *NameGV = new GlobalVariable(
          *NewF.getParent(), ArrayType::get(B.getInt8Ty(), Name.size() + 1),
          /*isConstant=*/true, GlobalValue::PrivateLinkage,
          ConstantDataArray::getString(C, Name), ".str." + Name, nullptr,
          GlobalValue::NotThreadLocal, /*AddressSpace=*/0);
      return H = B.CreateIntrinsic(
                 TargetExtType::get(C, "spirv.VulkanBuffer",
                                    {ArrayType::get(T, 0)},
                                    {/*StorageBuffer=*/12, /*writable=*/1}),
                 Intrinsic::spv_resource_handlefrombinding,
                 {B.getInt32(0), B.getInt32(Binding), B.getInt32(1),
                  B.getInt32(0), NameGV});
    };
    if (Accesses.empty())
      HandleFor(Type::getInt32Ty(C));
    Type *PtrTy = PointerType::get(C, /*StorageBuffer=*/11);
    for (auto [I, TX] : zip(Accesses, Idx)) {
      Value *P = IRBuilder<>(I).CreateIntrinsic(
          PtrTy, Intrinsic::spv_resource_getpointer,
          {HandleFor(TX.first), TX.second});
      I->setOperand(pointerOperandIndex(I), P);
    }
    for (auto [Cmp, New] : zip(Compares, Cmps)) {
      Cmp->replaceAllUsesWith(New);
      Cmp->eraseFromParent();
    }
    // Only derived pointers, possibly in cycles, use them now.
    for (Instruction *I : Derived)
      I->replaceAllUsesWith(PoisonValue::get(I->getType()));
    for (Instruction *I : Derived)
      I->eraseFromParent();
    return true;
  }
};
} // namespace

// Whether type T, value V, or a constant V uses, holds a pointer to module
// globals or generic memory, which have no Vulkan storage class.
static bool hasUnsupportedPtrTy(Type *T) {
  if (auto *PT = dyn_cast<PointerType>(T))
    return is_contained({1u, 2u, 4u}, PT->getAddressSpace());
  return any_of(T->subtypes(), hasUnsupportedPtrTy);
}

static bool usesUnsupportedPtr(const Value *V) {
  if (hasUnsupportedPtrTy(V->getType()))
    return true;
  if (auto *GA = dyn_cast<GlobalAlias>(V))
    if (const GlobalObject *O = GA->getAliaseeObject())
      V = O;
  // The backend gives each function its own copy of an addrspace(0) global;
  // dynamic shared memory is a declared addrspace(3) array.
  if (auto *GV = dyn_cast<GlobalVariable>(V))
    return hasUnsupportedPtrTy(GV->getValueType()) ||
           (GV->getAddressSpace() == 0 && !GV->isConstant()) ||
           (GV->getAddressSpace() == 3 && GV->isDeclaration());
  if (auto *GEP = dyn_cast<GEPOperator>(V);
      GEP && hasUnsupportedPtrTy(GEP->getSourceElementType()))
    return true;
  auto *C = dyn_cast<Constant>(V);
  return C && !isa<GlobalValue>(C) && any_of(C->operands(), usesUnsupportedPtr);
}

// Returns why a shader cannot express G's static call tree, or "" if it can.
// Path holds the functions on the current call path, Done those checked.
static std::string findUnsupported(Function &G,
                                   SmallPtrSetImpl<Function *> &Path,
                                   SmallPtrSetImpl<Function *> &Done) {
  if (Done.contains(&G))
    return "";
  // Vulkan forbids recursion.
  if (!Path.insert(&G).second)
    return ("recursive call to " + G.getName()).str();
  for (Instruction &I : instructions(G)) {
    auto *AI = dyn_cast<AllocaInst>(&I);
    if (usesUnsupportedPtr(&I) || any_of(I.operands(), usesUnsupportedPtr) ||
        (AI && hasUnsupportedPtrTy(AI->getAllocatedType()))) {
      std::string What = I.getOpcodeName();
      for (Value *Op : I.operands())
        if (auto *GV = dyn_cast<GlobalVariable>(getUnderlyingObject(Op)))
          What = ("@" + GV->getName()).str();
      return "unsupported global or generic pointer in " + What;
    }
    auto *CB = dyn_cast<CallBase>(&I);
    if (!CB)
      continue;
    if (CB->isInlineAsm())
      return "inline assembly";
    auto *Callee = dyn_cast<Function>(
        CB->getCalledOperand()->stripPointerCastsAndAliases());
    if (!Callee)
      return "indirect call";
    if (Callee->isIntrinsic())
      continue;
    // An entry point cannot be called; lowered kernels are entry points.
    if (Callee->getCallingConv() == CallingConv::SPIR_KERNEL ||
        Callee->hasFnAttribute("hlsl.shader"))
      return ("call to kernel " + Callee->getName()).str();
    if (Callee->isDeclaration()) {
      // Builtins the backend lowers to instructions, except extended ones,
      // which it lowers for OpenCL.std only; Vulkan has no linking.
      std::string Demangled =
          getOclOrSpirvBuiltinDemangledName(Callee->getName());
      const auto Set = SPIRV::InstructionSet::GLSL_std_450;
      if (!Demangled.empty() && SPIRV::isBuiltinCall(Demangled, Set) &&
          std::get<0>(SPIRV::mapBuiltinToOpcode(Demangled, Set)) !=
              SPIRV::Extended)
        continue;
      return ("call to undefined function " + Callee->getName()).str();
    }
    if (std::string Msg = findUnsupported(*Callee, Path, Done); !Msg.empty())
      return Msg;
  }
  Path.erase(&G);
  Done.insert(&G);
  return "";
}

static void lowerKernel(Function &F, ArrayRef<unsigned> FPWidths) {
  Module &M = *F.getParent();
  LLVMContext &C = M.getContext();
  const DataLayout &DL = M.getDataLayout();

  Function *NewF =
      Function::Create(FunctionType::get(Type::getVoidTy(C), false),
                       F.getLinkage(), F.getAddressSpace(), "", &M);
  NewF->copyAttributesFrom(&F);
  NewF->setSubprogram(F.getSubprogram());
  if (MDNode *WGSize = F.getMetadata("reqd_work_group_size"))
    NewF->setMetadata("reqd_work_group_size", WGSize);
  NewF->setAttributes(AttributeList::get(C, F.getAttributes().getFnAttrs(),
                                         AttributeSet(), {}));
  NewF->removeFnAttr(Attribute::Memory);
  NewF->removeFnAttr(Attribute::AllocSize);
  NewF->setCallingConv(CallingConv::SPIR_FUNC);
  NewF->addFnAttr("hlsl.shader", "compute");
  // The launch's block size, given as spec constants 0-2.
  if (!NewF->hasFnAttribute("hlsl.numthreads") &&
      !NewF->hasMetadata("reqd_work_group_size"))
    NewF->addFnAttr("spirv.workgroup-size-spec-ids", "0,1,2");
  NewF->takeName(&F);
  NewF->splice(NewF->begin(), &F);

  // Every argument is a push constant field: a scalar its value, a pointer
  // its byte offset into the buffer the runtime binds for it.
  SmallVector<Type *, 8> FieldTys;
  GlobalVariable *PC = nullptr;
  // Leave an empty body so the module, including aliases of the kernel, stays
  // valid after the diagnostic.
  auto Fail = [&](const Twine &Msg) {
    C.diagnose(DiagnosticInfoUnsupported(
        *NewF, Msg, DiagnosticLocation(),
        DeferUnsupported ? DS_Warning : DS_Error));
    if (DeferUnsupported) {
      Function *Err = Function::Create(
          FunctionType::get(Type::getVoidTy(C), false),
          GlobalValue::ExternalLinkage,
          "__hipspv_error: in function " + NewF->getName() + ": " + Msg, M);
      Err->addFnAttr("hlsl.shader", "compute");
      Err->addFnAttr("hlsl.numthreads", "1,1,1");
      ReturnInst::Create(C, BasicBlock::Create(C, "", Err));
    }
    GlobalValue::LinkageTypes Linkage = NewF->getLinkage();
    NewF->deleteBody();
    NewF->setLinkage(Linkage);
    ReturnInst::Create(C, BasicBlock::Create(C, "", NewF));
    if (PC)
      PC->eraseFromParent();
    F.replaceAllUsesWith(NewF);
    F.eraseFromParent();
  };
  // Execution modes requested for a kernel are kernel modes.
  if (NamedMDNode *EM = M.getNamedMetadata("spirv.ExecutionMode"))
    for (const MDNode *N : EM->operands())
      if (N->getNumOperands() &&
          mdconst::dyn_extract_or_null<Function>(N->getOperand(0)) == &F)
        return Fail("unsupported spirv.ExecutionMode");
  auto IsBuffer = [](Argument &A) {
    return A.getType()->isPointerTy() && !A.hasByValAttr() &&
           A.getType()->getPointerAddressSpace() == /*CrossWorkgroup=*/1;
  };
  for (Argument &A : F.args()) {
    if (IsBuffer(A))
      FieldTys.push_back(Type::getInt64Ty(C));
    else if (A.getType()->isIntegerTy(1) || isPlainDataTy(A.getType()))
      // Push constants have no booleans; a bool is passed as one byte.
      FieldTys.push_back(A.getType()->isIntegerTy(1) ? Type::getInt8Ty(C)
                                                     : A.getType());
    else if (A.hasByValAttr() && isPlainAggregate(A.getParamByValType()))
      FieldTys.push_back(A.getParamByValType());
    else
      return Fail("unsupported argument " + Twine(A.getArgNo()));
  }
  // A load of each scalar field, the address of each aggregate one.
  SmallVector<Value *, 8> Fields;
  if (!FieldTys.empty()) {
    auto *PCTy = StructType::create(C, FieldTys, NewF->getName().str() + ".pc");
    BasicBlock &Entry = NewF->getEntryBlock();
    IRBuilder<> B(&Entry, Entry.getFirstInsertionPt());
    Value *Block;
    if (DL.getTypeAllocSize(PCTy) > 128) {
      // Beyond the push constant space Vulkan guarantees, the block is a
      // read-only storage buffer "<kernel>.args" after the pointer arguments'.
      std::string Name = (NewF->getName() + ".args").str();
      auto *NameGV = new GlobalVariable(
          M, ArrayType::get(B.getInt8Ty(), Name.size() + 1),
          /*isConstant=*/true, GlobalValue::PrivateLinkage,
          ConstantDataArray::getString(C, Name), ".str." + Name, nullptr,
          GlobalValue::NotThreadLocal, /*AddressSpace=*/0);
      Value *Handle = B.CreateIntrinsic(
          TargetExtType::get(C, "spirv.VulkanBuffer", {ArrayType::get(PCTy, 0)},
                             {/*StorageBuffer=*/12, /*writable=*/0}),
          Intrinsic::spv_resource_handlefrombinding,
          {B.getInt32(0), B.getInt32(count_if(F.args(), IsBuffer)),
           B.getInt32(1), B.getInt32(0), NameGV});
      Block = B.CreateIntrinsic(PointerType::get(C, /*StorageBuffer=*/11),
                                Intrinsic::spv_resource_getpointer,
                                {Handle, B.getInt64(0)});
    } else {
      PC = new GlobalVariable(M, PCTy, /*isConstant=*/false,
                              GlobalValue::ExternalLinkage,
                              /*Initializer=*/nullptr, NewF->getName() + ".pc",
                              nullptr, GlobalValue::NotThreadLocal,
                              /*PushConstant=*/13,
                              /*isExternallyInitialized=*/true);
      PC->setVisibility(GlobalValue::HiddenVisibility);
      Block = PC;
    }
    for (auto [I, T] : enumerate(FieldTys)) {
      Value *Addr = B.CreateStructGEP(PCTy, Block, I);
      Fields.push_back(T->isAggregateType() ? Addr : B.CreateLoad(T, Addr));
    }
  }
  unsigned Binding = 0;
  for (Argument &A : F.args()) {
    Value *Field = Fields[A.getArgNo()];
    if (A.hasByValAttr()) {
      // The kernel may write its copy; push constants are read-only.
      Type *T = A.getParamByValType();
      IRBuilder<> B(&*NewF->getEntryBlock().getFirstInsertionPt());
      AllocaInst *Copy = B.CreateAlloca(T, DL.getAllocaAddrSpace());
      Copy->setAlignment(DL.getPrefTypeAlign(T));
      if (auto *FieldI = dyn_cast<Instruction>(Field))
        B.SetInsertPoint(FieldI->getNextNode());
      SmallVector<Value *, 4> Idx{B.getInt32(0)};
      copyLeaves(B, T, Field, Copy, Idx, T);
      A.replaceAllUsesWith(Copy);
      continue;
    }
    if (!IsBuffer(A)) {
      auto *Load = cast<Instruction>(Field);
      A.replaceAllUsesWith(
          IRBuilder<>(Load->getNextNode()).CreateTrunc(Load, A.getType()));
      continue;
    }
    BufferArgRewriter Rewriter(DL, A);
    if (!Rewriter.run(*NewF, Binding++, cast<Instruction>(Field))) {
      std::string Use;
      if (Rewriter.Unsupported)
        Use = std::string(" by ") + Rewriter.Unsupported->getOpcodeName();
      if (auto *CB = dyn_cast_or_null<CallBase>(Rewriter.Unsupported);
          CB && CB->getCalledFunction())
        Use += (" to " + CB->getCalledFunction()->getName()).str();
      return Fail("unsupported use of pointer argument " + Twine(A.getArgNo()) +
                  Use);
    }
  }
  SmallPtrSet<Function *, 8> Path, Done;
  if (std::string Msg = findUnsupported(*NewF, Path, Done); !Msg.empty())
    return Fail(Msg);
  // HIP keeps infinities, NaNs and signed zeros, which a Vulkan implementation
  // may otherwise assume absent, and denormals unless the kernel flushes them.
  NamedMDNode *EM = M.getOrInsertNamedMetadata("spirv.ExecutionMode");
  Type *I32 = Type::getInt32Ty(C);
  auto AddMode = [&](unsigned Mode, unsigned Width) {
    EM->addOperand(MDNode::get(
        C, {ValueAsMetadata::get(NewF),
            ConstantAsMetadata::get(ConstantInt::get(I32, Mode)),
            ConstantAsMetadata::get(ConstantInt::get(I32, Width))}));
  };
  for (unsigned Width : FPWidths) {
    AddMode(SPIRV::ExecutionMode::SignedZeroInfNanPreserve, Width);
    bool Flush =
        Width != 16 &&
        F.getDenormalMode(Width == 32 ? APFloat::IEEEsingle()
                                      : APFloat::IEEEdouble())
                .Output != DenormalMode::IEEE;
    AddMode(Flush ? SPIRV::ExecutionMode::DenormFlushToZero
                  : SPIRV::ExecutionMode::DenormPreserve,
            Width);
  }
  F.replaceAllUsesWith(NewF);
  F.eraseFromParent();
}

static bool lowerModule(Module &M) {
  bool Changed = lowerWorkItemBuiltins(M);
  SmallVector<Function *, 8> Kernels;
  for (Function &F : M)
    if (F.getCallingConv() == CallingConv::SPIR_KERNEL && !F.isDeclaration())
      Kernels.push_back(&F);
  // The floating-point widths the module computes with.
  SmallSetVector<unsigned, 3> FPWidths;
  auto AddFP = [&](Type *T) {
    if (T->getScalarType()->isFloatingPointTy())
      FPWidths.insert(T->getScalarSizeInBits());
  };
  for (Function &F : M)
    for (Instruction &I : instructions(F)) {
      AddFP(I.getType());
      for (Value *Op : I.operands())
        AddFP(Op->getType());
    }
  for (Function *F : Kernels)
    lowerKernel(*F, FPWidths.getArrayRef());
  return Changed || !Kernels.empty();
}

PreservedAnalyses HIPSPVLowerToHLSLShapePass::run(Module &M,
                                                  ModuleAnalysisManager &) {
  return lowerModule(M) ? PreservedAnalyses::none() : PreservedAnalyses::all();
}

namespace {
class HIPSPVLowerToHLSLShapeLegacy final : public ModulePass {
public:
  static char ID;
  HIPSPVLowerToHLSLShapeLegacy() : ModulePass(ID) {}
  bool runOnModule(Module &M) override { return lowerModule(M); }
};
} // namespace

char HIPSPVLowerToHLSLShapeLegacy::ID = 0;
INITIALIZE_PASS(HIPSPVLowerToHLSLShapeLegacy, "hipspv-lower-to-hlsl-shape",
                "Lower HIP kernels to Vulkan compute shaders", false, false)

ModulePass *llvm::createHIPSPVLowerToHLSLShapeLegacyPass() {
  return new HIPSPVLowerToHLSLShapeLegacy();
}
