//===- HIPSPVLowerToHLSLShape.cpp - HIP IR -> HLSL-shape rewrite ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Bridges HIP/CUDA-shaped IR (spir_kernel functions, raw addrspace(1) pointer
// args, __cuda_builtin_* fetch chains for threadIdx/blockIdx/blockDim/gridDim)
// into the HLSL-shaped IR that the in-tree SPIR-V backend's Shader/Vulkan path
// expects (hlsl.shader/hlsl.numthreads attributes, llvm.spv.* builtin
// intrinsics, target("spirv.VulkanBuffer") handle types).
//
// Phase 2 implements the entry-point and builtin lowering only; Phase 3 will
// add pointer-arg StorageBuffer wrapping and push-constant scalar args.
//
//===----------------------------------------------------------------------===//

#include "HIPSPVLowerToHLSLShape.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/IntrinsicsSPIRV.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/Utils/Local.h"

#define DEBUG_TYPE "hipspv-lower-to-hlsl-shape"
using namespace llvm;

namespace {

// Identifies position-builtin call sites by a substring that survives Itanium
// mangling. Two flavors are matched:
//   1. Upstream CUDA headers: __cuda_builtin_*::__fetch_builtin_{x,y,z}.
//   2. chipStar HIP headers: OpenCL-style get_local_id / get_group_id etc.
// The dimension index is read from either the trailing x/y/z letter (CUDA) or
// the i32 argument (OpenCL helpers).
struct PositionBuiltinKind {
  StringRef Marker;     // substring required in the mangled callee name
  Intrinsic::ID Intr;   // SPIR-V intrinsic to substitute
  bool DimFromArg;      // false: dim from x/y/z suffix; true: dim from arg 0
};

static const PositionBuiltinKind PositionBuiltins[] = {
    // chipStar / OpenCL-style helpers (dim is the function's i32 argument).
    {"get_local_id",    Intrinsic::spv_thread_id_in_group, true},
    {"get_global_id",   Intrinsic::spv_thread_id,          true},
    {"get_group_id",    Intrinsic::spv_group_id,           true},
    {"get_local_size",  Intrinsic::spv_workgroup_size,     true},
    {"get_num_groups",  Intrinsic::spv_num_workgroups,     true},
    // Upstream CUDA-header style (dim is the x/y/z suffix on the method name).
    {"__cuda_builtin_threadIdx_t", Intrinsic::spv_thread_id_in_group, false},
    {"__cuda_builtin_blockIdx_t",  Intrinsic::spv_group_id,           false},
    {"__cuda_builtin_blockDim_t",  Intrinsic::spv_workgroup_size,     false},
    {"__cuda_builtin_gridDim_t",   Intrinsic::spv_num_workgroups,     false},
};

// Maps the trailing dimension component of __fetch_builtin_x/y/z to 0/1/2.
// Returns -1 if the name does not match the expected pattern.
static int parseFetchDim(StringRef Name) {
  size_t Pos = Name.find("__fetch_builtin_");
  if (Pos == StringRef::npos)
    return -1;
  size_t Idx = Pos + StringRef("__fetch_builtin_").size();
  if (Idx >= Name.size())
    return -1;
  switch (Name[Idx]) {
  case 'x': return 0;
  case 'y': return 1;
  case 'z': return 2;
  default:  return -1;
  }
}

// Lowers i64 bit-count intrinsics (`llvm.ctlz.i64`, `llvm.cttz.i64`,
// `llvm.ctpop.i64`) to a pair of i32 operations. The Vulkan/Shader path of
// the SPIR-V backend can't legalize these on i64 directly — `OpExtInst
// GLSL.std.450 FindUMsb` etc. only operate on 32-bit integers. Splitting
// here produces:
//   ctlz(x) = is_zero_high ? 32 + ctlz(low) : ctlz(high)
//   cttz(x) = is_zero_low  ? 32 + cttz(high) : cttz(low)
//   ctpop(x) = ctpop(low) + ctpop(high)
// where high = trunc(x >> 32, i32) and low = trunc(x, i32).
static bool lowerI64BitcountIntrinsics(Function &F) {
  Module *M = F.getParent();
  bool Changed = false;
  Type *I32 = Type::getInt32Ty(F.getContext());
  Type *I64 = Type::getInt64Ty(F.getContext());
  SmallVector<CallInst *, 8> Worklist;
  for (BasicBlock &BB : F)
    for (Instruction &I : BB)
      if (auto *CI = dyn_cast<CallInst>(&I))
        if (auto *Callee = CI->getCalledFunction())
          if (Callee->isIntrinsic())
            Worklist.push_back(CI);
  for (CallInst *CI : Worklist) {
    Function *Callee = CI->getCalledFunction();
    Intrinsic::ID ID = Callee->getIntrinsicID();
    if (ID != Intrinsic::ctlz && ID != Intrinsic::cttz &&
        ID != Intrinsic::ctpop)
      continue;
    if (CI->getType() != I64)
      continue;
    Value *X = CI->getArgOperand(0);
    if (X->getType() != I64)
      continue;
    IRBuilder<> B(CI);
    Value *Lo = B.CreateTrunc(X, I32);
    Value *Hi = B.CreateTrunc(B.CreateLShr(X, B.getInt64(32)), I32);
    Value *Result = nullptr;
    if (ID == Intrinsic::ctpop) {
      Function *PopI32 =
          Intrinsic::getOrInsertDeclaration(M, Intrinsic::ctpop, {I32});
      Value *PLo = B.CreateCall(PopI32, {Lo});
      Value *PHi = B.CreateCall(PopI32, {Hi});
      Result = B.CreateZExt(B.CreateAdd(PLo, PHi), I64);
    } else {
      // ctlz/cttz both have a (value, i1 is_zero_undef) signature.
      Value *IsZeroUndef = CI->getArgOperand(1);
      Function *F32 = Intrinsic::getOrInsertDeclaration(M, ID, {I32});
      // For a split implementation we pass i1 false as is_zero_undef on each
      // half — we need the well-defined "32 if all zero" answer to combine
      // them safely. The outer `is_zero_undef` flag is preserved by selecting
      // the original value when the input is fully zero.
      Value *False = B.getFalse();
      if (ID == Intrinsic::ctlz) {
        Value *CHi = B.CreateCall(F32, {Hi, False});
        Value *CLo = B.CreateCall(F32, {Lo, False});
        Value *HiZero = B.CreateICmpEQ(Hi, B.getInt32(0));
        Value *Combined =
            B.CreateSelect(HiZero, B.CreateAdd(B.getInt32(32), CLo), CHi);
        Result = B.CreateZExt(Combined, I64);
      } else {
        Value *CLo = B.CreateCall(F32, {Lo, False});
        Value *CHi = B.CreateCall(F32, {Hi, False});
        Value *LoZero = B.CreateICmpEQ(Lo, B.getInt32(0));
        Value *Combined =
            B.CreateSelect(LoZero, B.CreateAdd(B.getInt32(32), CHi), CLo);
        Result = B.CreateZExt(Combined, I64);
      }
      // If is_zero_undef was true and the input was all zeros, behavior is
      // undefined — our select naturally returns 64, which is fine.
      (void)IsZeroUndef;
    }
    CI->replaceAllUsesWith(Result);
    CI->eraseFromParent();
    Changed = true;
  }

  // Second pass: lower remaining i32 ctlz/cttz intrinsics (including the
  // freshly-introduced ones from the i64 split above) to GLSL.std.450
  // FindUMsb / FindILsb via the spv_firstbituhigh / spv_firstbitlow
  // intrinsics. The Vulkan/Shader path of the SPIR-V backend has no
  // legalization rule for G_CTLZ / G_CTLZ_ZERO_UNDEF / G_CTTZ on s32, so we
  // must rewrite at the IR level. For x != 0:
  //   ctlz(x) = 31 - firstbituhigh(x)
  //   cttz(x) = firstbitlow(x)
  // For x == 0 (ctlz/cttz return 32):
  //   firstbituhigh(0) = -1  =>  31 - (-1) = 32  (works naturally)
  //   firstbitlow(0)   = -1  =>  use a select to return 32 explicitly.
  Worklist.clear();
  for (BasicBlock &BB : F)
    for (Instruction &I : BB)
      if (auto *CI = dyn_cast<CallInst>(&I))
        if (auto *Callee = CI->getCalledFunction())
          if (Callee->isIntrinsic())
            Worklist.push_back(CI);
  for (CallInst *CI : Worklist) {
    Function *Callee = CI->getCalledFunction();
    Intrinsic::ID ID = Callee->getIntrinsicID();
    if (ID != Intrinsic::ctlz && ID != Intrinsic::cttz)
      continue;
    if (CI->getType() != I32)
      continue;
    Value *X = CI->getArgOperand(0);
    if (X->getType() != I32)
      continue;
    IRBuilder<> B(CI);
    Value *Result = nullptr;
    if (ID == Intrinsic::ctlz) {
      Function *FBH = Intrinsic::getOrInsertDeclaration(
          M, Intrinsic::spv_firstbituhigh, {I32});
      Value *Hi = B.CreateCall(FBH, {X});
      // 31 - Hi. For x == 0 firstbituhigh returns -1, so this yields 32.
      Result = B.CreateSub(B.getInt32(31), Hi);
    } else {
      Function *FBL = Intrinsic::getOrInsertDeclaration(
          M, Intrinsic::spv_firstbitlow, {I32});
      Value *Lo = B.CreateCall(FBL, {X});
      // For x == 0 firstbitlow returns -1; we need 32. Select on x == 0.
      Value *IsZero = B.CreateICmpEQ(X, B.getInt32(0));
      Result = B.CreateSelect(IsZero, B.getInt32(32), Lo);
    }
    CI->replaceAllUsesWith(Result);
    CI->eraseFromParent();
    Changed = true;
  }
  return Changed;
}

// Replaces calls to chipStar's `_Z6memcpyPvPKvm` / `_Z6memsetPvi*` /
// `_Z7memmovePvPKvm` with the equivalent LLVM intrinsics. Otherwise the
// pointer-arg propagator skips them (because the callee is a `linkonce_odr`
// definition, not a declaration) and the backend's instruction selector
// trips on the resulting cross-addrspace argument types.
static bool remapMemFnsToIntrinsics(Function &F) {
  Module *M = F.getParent();
  bool Changed = false;
  SmallVector<CallInst *, 8> Worklist;
  for (BasicBlock &BB : F)
    for (Instruction &I : BB)
      if (auto *CI = dyn_cast<CallInst>(&I))
        Worklist.push_back(CI);
  for (CallInst *CI : Worklist) {
    Function *Callee = CI->getCalledFunction();
    if (!Callee)
      continue;
    StringRef N = Callee->getName();
    Intrinsic::ID ID = Intrinsic::not_intrinsic;
    bool IsMemset = false;
    bool IsThreeArg = true;
    if (N == "_Z6memcpyPvPKvm" || N == "memcpy") {
      ID = Intrinsic::memcpy;
    } else if (N == "_Z7memmovePvPKvm" || N == "memmove") {
      ID = Intrinsic::memmove;
    } else if (N == "_Z6memsetPvim" || N == "memset" ||
               N == "_Z6memsetPvjm") {
      ID = Intrinsic::memset;
      IsMemset = true;
    } else {
      continue;
    }
    if (CI->arg_size() != 3)
      continue;
    Value *DstArg = CI->getArgOperand(0);
    Value *SrcArg = CI->getArgOperand(1);
    Value *LenArg = CI->getArgOperand(2);
    if (!DstArg->getType()->isPointerTy() || !LenArg->getType()->isIntegerTy())
      continue;
    if (!IsMemset && !SrcArg->getType()->isPointerTy())
      continue;
    IRBuilder<> B(CI);
    if (IsMemset) {
      // memset's "value" is i32 in libc but llvm.memset takes i8.
      Value *I8Val = SrcArg;
      if (I8Val->getType()->isIntegerTy()) {
        unsigned BW = cast<IntegerType>(I8Val->getType())->getBitWidth();
        if (BW != 8)
          I8Val = B.CreateTrunc(I8Val, Type::getInt8Ty(F.getContext()));
      } else {
        continue;
      }
      Function *Decl = Intrinsic::getOrInsertDeclaration(
          M, ID, {DstArg->getType(), LenArg->getType()});
      CallInst *NC =
          B.CreateCall(Decl, {DstArg, I8Val, LenArg, B.getFalse()});
      NC->setCallingConv(Decl->getCallingConv());
      // memcpy/memset/memmove return dest in libc, but llvm.* returns void.
      // Replace the call's uses with the dest arg if it had any.
      if (!CI->getType()->isVoidTy())
        CI->replaceAllUsesWith(DstArg);
      CI->eraseFromParent();
    } else {
      Function *Decl = Intrinsic::getOrInsertDeclaration(
          M, ID, {DstArg->getType(), SrcArg->getType(), LenArg->getType()});
      CallInst *NC =
          B.CreateCall(Decl, {DstArg, SrcArg, LenArg, B.getFalse()});
      NC->setCallingConv(Decl->getCallingConv());
      if (!CI->getType()->isVoidTy())
        CI->replaceAllUsesWith(DstArg);
      CI->eraseFromParent();
    }
    (void)IsThreeArg;
    Changed = true;
  }
  return Changed;
}

// Maps Itanium-mangled OpenCL math builtin calls (e.g. `_Z4log2f`,
// `_Z3sinf`, `_Z3cosd`) to the corresponding LLVM-IR intrinsic
// (`@llvm.log2.f32`, `@llvm.sin.f32`, etc.). The SPIR-V backend's
// selectIntrinsic correctly maps `llvm.log2.f32` to GLSL.std.450::Log2 in
// Shader mode (SPIRVInstructionSelector.cpp:1131); the TableGen-driven
// `lookupBuiltin` path only recognises the OpenCL "log2" name and so
// fails in Shader mode, leaving an unresolved external call that the
// backend later trips over.
//
// Only handles a curated subset of unary/binary single-precision and
// double-precision math fns whose semantics map 1:1 to LLVM intrinsics.
// Anything not in the table is left as-is for the existing path to
// handle (or fail with a clear error).
static bool remapOpenCLMathToIntrinsics(Function &F) {
  struct Entry {
    StringRef Name;        // demangled name, lowercase
    Intrinsic::ID Intr;
    unsigned ArgCount;
  };
  static const Entry Table[] = {
      // unary float/double fns
      {"log",   Intrinsic::log,   1},
      {"log2",  Intrinsic::log2,  1},
      {"log10", Intrinsic::log10, 1},
      {"exp",   Intrinsic::exp,   1},
      {"exp2",  Intrinsic::exp2,  1},
      {"sin",   Intrinsic::sin,   1},
      {"cos",   Intrinsic::cos,   1},
      {"tan",   Intrinsic::tan,   1},
      {"sqrt",  Intrinsic::sqrt,  1},
      {"fabs",  Intrinsic::fabs,  1},
      {"ceil",  Intrinsic::ceil,  1},
      {"floor", Intrinsic::floor, 1},
      {"trunc", Intrinsic::trunc, 1},
      {"round", Intrinsic::round, 1},
      // binary
      {"pow",   Intrinsic::pow,   2},
      {"copysign", Intrinsic::copysign, 2},
      {"fmin",  Intrinsic::minnum, 2},
      {"fmax",  Intrinsic::maxnum, 2},
      // FMA family
      {"fma",   Intrinsic::fma,   3},
  };
  bool Changed = false;
  SmallVector<CallInst *, 16> Worklist;
  for (BasicBlock &BB : F)
    for (Instruction &I : BB)
      if (auto *CI = dyn_cast<CallInst>(&I))
        Worklist.push_back(CI);
  for (CallInst *CI : Worklist) {
    Function *Callee = CI->getCalledFunction();
    if (!Callee || !Callee->isDeclaration())
      continue;
    StringRef MName = Callee->getName();
    if (!MName.starts_with("_Z"))
      continue;
    // Quick demangle by isolating the digit-prefixed name segment after _Z:
    //   "_Z4log2f" -> length=4 -> "log2" (then arg type chars)
    StringRef Rest = MName.drop_front(2);
    unsigned Len = 0;
    while (Len < Rest.size() && isdigit(static_cast<unsigned char>(Rest[Len])))
      ++Len;
    if (Len == 0)
      continue;
    unsigned NameLen = 0;
    if (Rest.substr(0, Len).getAsInteger(10, NameLen) || NameLen == 0)
      continue;
    if (NameLen > Rest.size() - Len)
      continue;
    StringRef Demangled = Rest.substr(Len, NameLen);
    const Entry *Match = nullptr;
    for (const Entry &E : Table) {
      if (E.Name == Demangled && E.ArgCount == CI->arg_size()) {
        Match = &E;
        break;
      }
    }
    if (!Match)
      continue;
    Type *ResTy = CI->getType();
    if (!ResTy->isFloatingPointTy())
      continue;
    SmallVector<Value *, 3> Args(CI->args());
    // Sanity-check arg types match the result type (uniform scalar fp fns).
    bool TypeOk = true;
    for (Value *A : Args)
      if (A->getType() != ResTy) {
        TypeOk = false;
        break;
      }
    if (!TypeOk)
      continue;
    IRBuilder<> B(CI);
    Function *Decl =
        Intrinsic::getOrInsertDeclaration(F.getParent(), Match->Intr, {ResTy});
    CallInst *NC = B.CreateCall(Decl, Args);
    NC->setCallingConv(Decl->getCallingConv());
    if (CI->getFastMathFlags().any())
      NC->setFastMathFlags(CI->getFastMathFlags());
    CI->replaceAllUsesWith(NC);
    CI->eraseFromParent();
    Changed = true;
  }
  return Changed;
}

// Promotes the clang -O0 alloca-shuffle pattern to direct value forwarding.
// Pattern recognised:
//   %slot      = alloca T
//   %slot.cast = addrspacecast ptr %slot to ptr addrspace(N)        ; optional
//   store V, %slot.cast            ; one and only one reaching store
//   %loaded    = load T, %slot.cast ; any number of loads
// For each load, replace uses of `%loaded` with V (bridging via addrspacecast
// when V's type differs from the load's). The store, alloca, and casts go
// dead and get cleaned up by the subsequent DCE pass.
//
// SROA refuses to do this when the stored value's pointer-addrspace differs
// from the alloca's element addrspace (common in HIP -O0 because clang
// allocas in addrspace(0)/Function but loads/stores via the addrspace(4)
// `Generic` cast). Without promotion the pointer-arg rewrite leaves
// addrspace(11) values stored into addrspace(4) slots, which the backend's
// SPIRVLegalizePointerCast can't reconcile.
static bool promoteAllocaShuffles(Function &F) {
  if (F.isDeclaration() || F.getEntryBlock().empty())
    return false;
  bool Changed = false;
  SmallVector<AllocaInst *, 16> Allocas;
  for (Instruction &I : F.getEntryBlock())
    if (auto *A = dyn_cast<AllocaInst>(&I))
      Allocas.push_back(A);
  for (AllocaInst *A : Allocas) {
    if (!A->isStaticAlloca())
      continue;
    // Walk all aliases of A through addrspacecast/bitcast chains, and check
    // that uses are exactly: stores (1 reaching def), loads (any), casts.
    SmallPtrSet<User *, 8> Aliases;
    SmallVector<User *, 8> Worklist{A};
    Aliases.insert(A);
    bool BadUse = false;
    StoreInst *TheStore = nullptr;
    SmallVector<LoadInst *, 4> Loads;
    SmallPtrSet<LoadInst *, 4> SeenLoads;
    while (!Worklist.empty() && !BadUse) {
      User *U = Worklist.pop_back_val();
      for (User *UU : U->users()) {
        if (isa<AddrSpaceCastInst>(UU) || isa<BitCastInst>(UU)) {
          if (Aliases.insert(UU).second)
            Worklist.push_back(UU);
          continue;
        }
        if (auto *SI = dyn_cast<StoreInst>(UU)) {
          // Slot must be the pointer operand (storing INTO the alloca).
          if (SI->getValueOperand() == cast<Value>(U)) {
            BadUse = true;
            break;
          }
          if (TheStore && TheStore != SI) {
            BadUse = true;
            break;
          }
          TheStore = SI;
          continue;
        }
        if (auto *LI = dyn_cast<LoadInst>(UU)) {
          if (SeenLoads.insert(LI).second)
            Loads.push_back(LI);
          continue;
        }
        // Lifetime markers from clang are safe to ignore. Anything else
        // (e.g. GEP, call) means the alloca might escape; bail out.
        if (auto *II = dyn_cast<IntrinsicInst>(UU)) {
          Intrinsic::ID ID = II->getIntrinsicID();
          if (ID == Intrinsic::lifetime_start ||
              ID == Intrinsic::lifetime_end)
            continue;
        }
        BadUse = true;
        break;
      }
    }
    if (BadUse || !TheStore)
      continue;
    Value *V = TheStore->getValueOperand();
    bool Promoted = false;
    for (LoadInst *LI : Loads) {
      Value *Repl = V;
      if (Repl->getType() != LI->getType()) {
        // Bridge differing pointer addrspaces with an addrspacecast.
        if (!V->getType()->isPointerTy() || !LI->getType()->isPointerTy())
          continue; // type-incompatible — bail on this load
        IRBuilder<> B(LI);
        Repl = B.CreateAddrSpaceCast(V, LI->getType());
      }
      LI->replaceAllUsesWith(Repl);
      LI->eraseFromParent();
      Promoted = true;
      Changed = true;
    }
    if (Promoted)
      TheStore->eraseFromParent();
  }
  return Changed;
}

// Stamps the entry-point function with hlsl.shader+hlsl.numthreads and rewires
// the calling convention from spir_kernel to spir_func.
static bool stampShaderAttrs(Function &F) {
  if (F.getCallingConv() != CallingConv::SPIR_KERNEL)
    return false;
  F.addFnAttr("hlsl.shader", "compute");
  // Phase 4 will replace this placeholder with spec-constant ids for runtime
  // block dims. For now any literal value gets us through the backend gate.
  if (!F.hasFnAttribute("hlsl.numthreads"))
    F.addFnAttr("hlsl.numthreads", "1,1,1");
  F.setCallingConv(CallingConv::SPIR_FUNC);
  return true;
}

// Replaces calls to position-builtin helpers (CUDA fetch chains or OpenCL-style
// get_*_id/size helpers) with the matching SPIR-V intrinsic call. The CUDA
// flavor reads the dim index from the x/y/z suffix; the OpenCL flavor reads it
// from the call's i32 argument (must be a constant — always true in practice
// because chipStar's headers pass literal 0/1/2).
static bool replacePositionBuiltinCall(CallInst *CI) {
  Function *Callee = CI->getCalledFunction();
  if (!Callee)
    return false;
  StringRef Name = Callee->getName();
  for (const auto &Kind : PositionBuiltins) {
    if (Name.find(Kind.Marker) == StringRef::npos)
      continue;
    int Dim = -1;
    if (Kind.DimFromArg) {
      if (CI->arg_size() < 1)
        continue;
      auto *DimArg = dyn_cast<ConstantInt>(CI->getArgOperand(0));
      if (!DimArg)
        continue; // Non-literal dim is rare in HIP kernels; bail loudly later.
      Dim = static_cast<int>(DimArg->getZExtValue());
      if (Dim < 0 || Dim > 2)
        continue;
    } else {
      Dim = parseFetchDim(Name);
      if (Dim < 0)
        continue;
    }
    IRBuilder<> B(CI);
    Type *RetTy = CI->getType();
    // Vulkan requires the SPIR-V position builtins (WorkgroupId,
    // LocalInvocationId, etc.) to be 3-component i32 vectors. Even when the
    // OpenCL helper signature returns i64, emit the i32 form and extend.
    Type *IntrRetTy = RetTy;
    bool NeedsExtend = false;
    if (auto *IT = dyn_cast<IntegerType>(RetTy)) {
      if (IT->getBitWidth() != 32) {
        IntrRetTy = Type::getInt32Ty(CI->getContext());
        NeedsExtend = true;
      }
    }
    Function *Intr = Intrinsic::getOrInsertDeclaration(
        CI->getModule(), Kind.Intr, {IntrRetTy});
    Value *Call = B.CreateCall(
        Intr, {ConstantInt::get(B.getInt32Ty(), Dim)});
    if (NeedsExtend)
      Call = B.CreateZExt(Call, RetTy);
    CI->replaceAllUsesWith(Call);
    CI->eraseFromParent();
    return true;
  }
  return false;
}

static bool lowerPositionBuiltinsInFunction(Function &F) {
  bool Changed = false;
  SmallVector<CallInst *, 8> Worklist;
  for (BasicBlock &BB : F)
    for (Instruction &I : BB)
      if (auto *CI = dyn_cast<CallInst>(&I))
        Worklist.push_back(CI);
  for (CallInst *CI : Worklist)
    Changed |= replacePositionBuiltinCall(CI);
  return Changed;
}

// Walks uses of a kernel arg to discover the value type accessed via load/store.
// Looks through: addrspacecast, ptrtoint+inttoptr, GEP, bitcast, and the
// alloca→store→load shuffle clang inserts at -O0 for arg variables. Returns
// the access type (e.g. float, i32) or i8 if no element-typed access is found.
static Type *inferArgAccessType(Argument &A) {
  // Tracked: a set of values that alias (transitively) to A's pointer.
  // We follow GEP/addrspacecast/bitcast/ptrtoint/inttoptr/load through it,
  // and recognise alloca→store→load shuffles by detecting "store TrackedValue
  // into Slot" then enqueuing reads from Slot.
  SmallPtrSet<const Value *, 32> Tracked;
  Tracked.insert(&A);
  SmallVector<Value *, 16> Worklist;
  for (User *U : A.users())
    Worklist.push_back(U);
  SmallPtrSet<User *, 32> SeenUsers;
  // GEP-shape evidence: when the kernel never does a direct load/store of an
  // element type through the arg (e.g. it GEPs by struct then passes the
  // element pointer to a helper function or memcpy), the GEP's source element
  // type tells us the buffer's logical element shape. We collect candidates
  // here and pick one as a fallback after the load/store walk.
  SmallVector<Type *, 4> GEPElemEvidence;

  while (!Worklist.empty()) {
    Value *V = Worklist.pop_back_val();
    auto *U = dyn_cast<User>(V);
    if (!U || !SeenUsers.insert(U).second)
      continue;

    if (auto *LI = dyn_cast<LoadInst>(U)) {
      // If the loaded value is a pointer, treat it as another tracked alias
      // (it came from a slot we previously wrote our pointer into).
      if (LI->getType()->isPointerTy()) {
        Tracked.insert(LI);
        for (User *UU : LI->users())
          Worklist.push_back(UU);
        continue;
      }
      // Element-typed load through a tracked pointer = the access type.
      return LI->getType();
    }

    if (auto *SI = dyn_cast<StoreInst>(U)) {
      Value *Val = SI->getValueOperand();
      Value *Ptr = SI->getPointerOperand();
      bool ValTracked = Tracked.count(Val) > 0;
      bool PtrTracked = Tracked.count(Ptr) > 0 ||
                        Tracked.count(Ptr->stripPointerCasts()) > 0;
      if (ValTracked) {
        // Arg-shuffle: pointer is stored into a slot. Track the slot too,
        // then follow loads from it (which will yield the same pointer).
        Value *Slot = Ptr->stripPointerCasts();
        Tracked.insert(Slot);
        for (User *UU : Slot->users())
          Worklist.push_back(UU);
        continue;
      }
      if (PtrTracked) {
        // Store INTO the buffer: value type is the element type.
        Type *VT = Val->getType();
        if (!VT->isPointerTy())
          return VT;
      }
      continue;
    }

    if (auto *GEP = dyn_cast<GetElementPtrInst>(U)) {
      // Record the GEP's source-element-type as evidence of the buffer's
      // logical element shape. We only use this when no direct load/store
      // through a tracked pointer is found. Skip i8 (raw byte-offset GEPs)
      // since that's our existing fallback and provides no extra information.
      Type *SrcElem = GEP->getSourceElementType();
      if (SrcElem && !SrcElem->isIntegerTy(8))
        GEPElemEvidence.push_back(SrcElem);
      Tracked.insert(U);
      for (User *UU : U->users())
        Worklist.push_back(UU);
      continue;
    }
    if (isa<AddrSpaceCastInst>(U) ||
        isa<PtrToIntInst>(U) || isa<IntToPtrInst>(U) || isa<BitCastInst>(U)) {
      Tracked.insert(U);
      for (User *UU : U->users())
        Worklist.push_back(UU);
      continue;
    }
    // When our pointer is passed to a function, peek at the called function's
    // metadata for the param's pointee type. chipStar's HIP math/atomic
    // helpers carry per-arg `kernel_arg_type` info; even without it, the
    // pointee type is recoverable for the most common cases below.
    if (auto *CI = dyn_cast<CallInst>(U)) {
      Function *Callee = CI->getCalledFunction();
      if (!Callee)
        continue;
      // memcpy/memmove with our tracked pointer as dst or src: the other
      // operand often comes from a byval-struct kernel arg, whose pointee
      // type is the buffer's logical element type. (Tested case:
      // `*Dst = Src;` for a struct-typed kernel arg becomes
      // `memcpy(Dst, &Src, sizeof(T))` with no GEP/load/store on Dst.)
      if (auto *MI = dyn_cast<MemIntrinsic>(CI)) {
        Value *Other = nullptr;
        if (auto *MTI = dyn_cast<MemTransferInst>(MI)) {
          if (Tracked.count(MTI->getRawDest()) ||
              Tracked.count(MTI->getRawDest()->stripPointerCasts()))
            Other = MTI->getRawSource();
          else if (Tracked.count(MTI->getRawSource()) ||
                   Tracked.count(MTI->getRawSource()->stripPointerCasts()))
            Other = MTI->getRawDest();
        }
        if (Other) {
          // Peel addrspacecast / bitcast / ptrtoint+inttoptr chains.
          Value *Cur = Other;
          for (unsigned I = 0; I < 8; ++I) {
            if (auto *ASC = dyn_cast<AddrSpaceCastInst>(Cur)) {
              Cur = ASC->getPointerOperand();
              continue;
            }
            if (auto *BC = dyn_cast<BitCastInst>(Cur)) {
              Cur = BC->getOperand(0);
              continue;
            }
            if (auto *I2P = dyn_cast<IntToPtrInst>(Cur)) {
              if (auto *P2I = dyn_cast<PtrToIntInst>(I2P->getOperand(0))) {
                Cur = P2I->getPointerOperand();
                continue;
              }
            }
            break;
          }
          if (auto *OA = dyn_cast<Argument>(Cur)) {
            if (OA->hasByValAttr())
              return OA->getParamByValType();
          }
        }
        // Fall through to keep walking other uses.
        continue;
      }
      // Special-case common math intrinsics that take int*/uint*/long* and
      // write to it (frexpf, modff, etc.). We grep the demangled name and
      // map the second-arg pointee type by signature.
      StringRef N = Callee->getName();
      auto pointeeFromName = [&](StringRef Mangled) -> Type * {
        // mangled fragments for the integer-output math fns
        // _Z6frexpfPf, _Z5modffPf, _Z6frexpfPi (depends on chipStar header)
        if (Mangled.contains("frexp") || Mangled.contains("remquo") ||
            Mangled.contains("modf") || Mangled.contains("ilogb"))
          return Type::getInt32Ty(A.getContext());
        return nullptr;
      };
      if (Type *T = pointeeFromName(N))
        return T;
      // Fallback: leave the call as a use we can't inspect further.
      continue;
    }
  }
  // No direct load/store found. If the kernel only ever GEPs through the arg
  // (e.g. a struct-stride GEP whose result is handed to a helper function or
  // memcpy), use the most common GEP source-element-type as the buffer's
  // element shape. Picking the largest type when several were observed keeps
  // OpAccessChain stride correct for the dominant access pattern; smaller
  // accesses degrade naturally since rewriteToStorageBuffer only converts
  // stride-matching single-index GEPs.
  if (!GEPElemEvidence.empty()) {
    DenseMap<Type *, unsigned> Counts;
    for (Type *T : GEPElemEvidence)
      ++Counts[T];
    Type *Best = GEPElemEvidence[0];
    unsigned BestCount = 0;
    uint64_t BestSize = 0;
    const DataLayout &DL = A.getParent()->getDataLayout();
    for (auto &KV : Counts) {
      uint64_t Sz = DL.getTypeAllocSize(KV.first).getFixedValue();
      if (KV.second > BestCount ||
          (KV.second == BestCount && Sz > BestSize)) {
        Best = KV.first;
        BestCount = KV.second;
        BestSize = Sz;
      }
    }
    return Best;
  }
  return Type::getInt8Ty(A.getContext());
}

// Phase 3b: rewrites the GEP / Load / Store / AddrSpaceCast use chain rooted
// at an addrspacecast(addrspace(11) -> addrspace(N)) so all reachable pointer
// derivations live in addrspace(11). The bridging cast can then be erased and
// the backend's Vulkan path accepts the result.
//
// Single-index GEPs on the byte-buffer-of-element-type get rewritten to
// `getpointer(handle, elem_idx)` calls so the backend emits OpAccessChain
// (Vulkan-required) instead of OpPtrAccessChain.
static void rewriteToStorageBuffer(Value *Handle, Type *VkBufTy, Type *ElemTy,
                                   Value *NewPtr, Value *Cast) {
  DenseMap<Value *, Value *> Repl;
  Repl[Cast] = NewPtr;

  SmallVector<Value *, 16> Worklist{Cast};
  SmallVector<Instruction *, 16> ToErase;
  Module *M = nullptr;
  if (auto *I = dyn_cast<Instruction>(NewPtr))
    M = I->getModule();

  while (!Worklist.empty()) {
    Value *Old = Worklist.pop_back_val();
    Value *New = Repl[Old];
    SmallVector<User *, 8> Users(Old->user_begin(), Old->user_end());
    for (User *U : Users) {
      auto *I = dyn_cast<Instruction>(U);
      if (!I || !I->getParent())
        continue;
      if (auto *GEP = dyn_cast<GetElementPtrInst>(I)) {
        if (GEP->getPointerOperand() != Old)
          continue;
        IRBuilder<> Local(GEP);
        SmallVector<Value *, 4> Idx(GEP->indices());
        const DataLayout &DL = GEP->getDataLayout();
        Type *SrcElemTy = GEP->getSourceElementType();
        Value *NG = nullptr;
        // Special case: GEP with a single index over a byte-aggregate that's
        // sizeof(ElemTy) bytes wide. That's "advance by idx ElemTy units" —
        // emit `getpointer(handle, idx)` so the backend produces OpAccessChain.
        uint64_t ElemSize =
            ElemTy ? DL.getTypeAllocSize(ElemTy).getFixedValue() : 0;
        if (Idx.size() == 1 && M && Handle && ElemTy &&
            DL.getTypeAllocSize(SrcElemTy) == ElemSize) {
          auto *PtrSB = PointerType::get(GEP->getContext(), 11);
          // getpointer takes i32 idx by HLSL convention.
          Value *I32Idx = Local.CreateTrunc(
              Idx[0], Type::getInt32Ty(GEP->getContext()));
          // Re-materialise the handle locally so it doesn't have cross-BB
          // uses. The Vulkan/Logical backend's reg2mem pass spills cross-BB
          // SSA values to Function-scope allocas, and target ext types
          // (`spirv.VulkanBuffer`) can't be allocated in Function storage
          // class — `mem2reg` then fails to promote them back, leaving an
          // illegal OpVariable in the SPIR-V output.
          Value *LocalHandle = Handle;
          if (auto *HCI = dyn_cast<CallInst>(Handle)) {
            if (HCI->getParent() != GEP->getParent()) {
              SmallVector<Value *, 5> CloneArgs(HCI->args());
              LocalHandle = Local.CreateIntrinsic(
                  HCI->getType(),
                  Intrinsic::spv_resource_handlefrombinding, CloneArgs);
            }
          }
          NG = Local.CreateIntrinsic(
              PtrSB, Intrinsic::spv_resource_getpointer,
              {LocalHandle, I32Idx});
        } else {
          NG = Local.CreateGEP(SrcElemTy, New, Idx, "",
                               GEP->getNoWrapFlags());
        }
        Repl[GEP] = NG;
        Worklist.push_back(GEP);
      } else if (auto *LI = dyn_cast<LoadInst>(I)) {
        if (LI->getPointerOperand() != Old)
          continue;
        IRBuilder<> Local(LI);
        Value *NL = Local.CreateAlignedLoad(LI->getType(), New, LI->getAlign(),
                                            LI->isVolatile());
        LI->replaceAllUsesWith(NL);
        ToErase.push_back(LI);
      } else if (auto *SI = dyn_cast<StoreInst>(I)) {
        if (SI->getPointerOperand() != Old)
          continue;
        IRBuilder<> Local(SI);
        Local.CreateAlignedStore(SI->getValueOperand(), New, SI->getAlign(),
                                 SI->isVolatile());
        ToErase.push_back(SI);
      } else if (auto *AC = dyn_cast<AddrSpaceCastInst>(I)) {
        unsigned DstAS = AC->getDestAddressSpace();
        unsigned NewAS =
            cast<PointerType>(New->getType())->getAddressSpace();
        if (DstAS == NewAS) {
          AC->replaceAllUsesWith(New);
          ToErase.push_back(AC);
        } else {
          // Bypass: forward New into the chain rather than emit a cross-AS cast
          // (Vulkan's Logical addressing has no generic addrspacecast).
          Repl[AC] = New;
          Worklist.push_back(AC);
        }
      } else if (auto *CI = dyn_cast<CallInst>(I)) {
        Function *Callee = CI->getCalledFunction();
        if (!Callee || !Callee->isDeclaration())
          continue;
        // LLVM intrinsics whose name encodes pointer address spaces (e.g.
        // `llvm.memset.p4.i64`) need to be re-emitted at the new addrspace
        // mangling, not just retyped. Use Intrinsic::getOrInsertDeclaration
        // to materialise the correct intrinsic for our pointer types.
        if (Callee->isIntrinsic()) {
          Intrinsic::ID ID = Callee->getIntrinsicID();
          IRBuilder<> Local(CI);
          SmallVector<Value *, 8> NewArgs;
          for (unsigned ArgI = 0, E = CI->arg_size(); ArgI != E; ++ArgI) {
            Value *V = CI->getArgOperand(ArgI);
            if (V == Old)
              V = Repl[Old];
            NewArgs.push_back(V);
          }
          // Mangle the intrinsic with the new arg types. We collect any
          // overload-affecting types from the args; for memset/memcpy/memmove
          // and similar, the pointer-args' types are the overloads.
          SmallVector<Type *, 4> OverloadTys;
          for (Value *V : NewArgs)
            if (V->getType()->isPointerTy())
              OverloadTys.push_back(V->getType());
          // Length-typed overload (i64 for the *.i64 intrinsics):
          for (Value *V : NewArgs)
            if (auto *IT = dyn_cast<IntegerType>(V->getType()))
              if (IT->getBitWidth() == 64)
                OverloadTys.push_back(IT);
          Function *NewIntr = Intrinsic::getOrInsertDeclaration(
              CI->getModule(), ID, OverloadTys);
          if (!NewIntr || NewIntr == Callee)
            continue;
          CallInst *NC = Local.CreateCall(NewIntr, NewArgs);
          NC->setCallingConv(CI->getCallingConv());
          if (!CI->getType()->isVoidTy())
            CI->replaceAllUsesWith(NC);
          ToErase.push_back(CI);
          continue;
        }
        auto *NewPtrPT = cast<PointerType>(NewPtr->getType());
        FunctionType *OldFT = CI->getFunctionType();
        SmallVector<Type *, 8> NewParams(OldFT->params());
        bool Mutated = false;
        for (unsigned ArgI = 0, E = CI->arg_size(); ArgI != E; ++ArgI) {
          if (CI->getArgOperand(ArgI) != Old)
            continue;
          if (!NewParams[ArgI]->isPointerTy())
            continue;
          NewParams[ArgI] = NewPtrPT;
          Mutated = true;
        }
        if (!Mutated)
          continue;
        FunctionType *NewFT = FunctionType::get(
            OldFT->getReturnType(), NewParams, OldFT->isVarArg());
        IRBuilder<> Local(CI);
        SmallVector<Value *, 8> NewArgs;
        for (unsigned ArgI = 0, E = CI->arg_size(); ArgI != E; ++ArgI) {
          Value *V = CI->getArgOperand(ArgI);
          if (V == Old)
            V = Repl[Old];
          NewArgs.push_back(V);
        }
        // CreateCall with explicit FunctionType — the Function it dispatches
        // to keeps its original name + declared type (used by backend
        // builtin matching), but the call passes args of the new types.
        CallInst *NC = Local.CreateCall(NewFT, Callee, NewArgs);
        NC->setCallingConv(CI->getCallingConv());
        if (!CI->getType()->isVoidTy())
          CI->replaceAllUsesWith(NC);
        ToErase.push_back(CI);
      } else if (auto *PTI = dyn_cast<PtrToIntInst>(I)) {
        if (PTI->getPointerOperand() != Old)
          continue;
        // chipStar HIP headers express "addrspace(1) -> addrspace(4) generic"
        // as ptrtoint+inttoptr instead of addrspacecast. Look for matching
        // inttoptr users and forward our addrspace(11) pointer through them.
        SmallVector<User *, 4> PUsers(PTI->user_begin(), PTI->user_end());
        bool Any = false;
        for (User *PU : PUsers) {
          auto *ITP = dyn_cast<IntToPtrInst>(PU);
          if (!ITP)
            continue;
          Repl[ITP] = New;
          Worklist.push_back(ITP);
          Any = true;
        }
        if (Any) {
          // The ptrtoint becomes dead once its inttoptr children are gone.
          ToErase.push_back(PTI);
        }
      }
      // Other users (calls, etc.) intentionally untouched.
    }
  }

  // Stores have side effects that LLVM won't drop on its own, but we know we
  // emitted replacements with identical effect — force-erase them first.
  for (Instruction *I : ToErase)
    if (isa<StoreInst>(I))
      I->eraseFromParent();

  // Everything else (loads, GEPs, casts, ptrtoint/inttoptr) can be erased
  // once it has no remaining uses. Iterate until stable to handle chains.
  SmallSetVector<Instruction *, 16> Pending;
  for (Instruction *I : ToErase)
    if (I->getParent())
      Pending.insert(I);
  for (auto &KV : Repl)
    if (auto *I = dyn_cast<Instruction>(KV.first))
      if (I->getParent())
        Pending.insert(I);

  bool Progress = true;
  while (Progress) {
    Progress = false;
    SmallVector<Instruction *, 8> Erased;
    for (Instruction *I : Pending) {
      if (I->use_empty()) {
        I->eraseFromParent();
        Erased.push_back(I);
        Progress = true;
      }
    }
    for (Instruction *I : Erased)
      Pending.remove(I);
  }
}

// Phase 3a: rewrites a kernel function so that it takes no parameters (Vulkan
// entry-point requirement) and materialises each pointer arg as a
// StorageBuffer descriptor at the start of the body. Returns the new function;
// the original is erased.
//
// Scalar args are replaced with zero placeholders here; Phase 3c will rewrite
// them as push-constant loads.
// Returns true if the module's `__chip_module_has_no_IGBAs` flag is present
// AND set to 0 (meaning: the module HAS Indirect Global Buffer Accesses, so
// pointer args must be lowered via Buffer Device Address rather than Vulkan
// StorageBuffer descriptors). If the flag is absent or non-zero, returns
// false (default: existing Logical / VulkanBuffer path).
static bool moduleHasIGBAs(const Module &M) {
  const GlobalVariable *GV = M.getGlobalVariable("__chip_module_has_no_IGBAs");
  if (!GV || !GV->hasInitializer())
    return false;
  if (auto *CI = dyn_cast<ConstantInt>(GV->getInitializer()))
    return CI->isZero();
  return false;
}

// Phase F (BDA): rewrite a kernel so each pointer arg becomes a uint64
// push-constant field that is converted to a PhysicalStorageBuffer pointer
// (addrspace(5)) at kernel entry via inttoptr. Scalar/byval args are still
// clustered into the push-constant struct alongside the pointer slots.
//
// This is the minimum bridging-pass output for the BDA path; the SPIR-V
// backend is responsible for selecting the inttoptr to OpConvertUToPtr and
// emitting the PhysicalStorageBuffer64 memory model + capability.
static Function *rewriteKernelSignatureBDA(Function &F) {
  LLVMContext &C = F.getContext();
  Module *M = F.getParent();
  auto *VoidTy = Type::getVoidTy(C);
  auto *NewFTy = FunctionType::get(VoidTy, {}, false);
  auto *I32Ty = Type::getInt32Ty(C);
  auto *I64Ty = Type::getInt64Ty(C);

  Function *NewF =
      Function::Create(NewFTy, F.getLinkage(), F.getAddressSpace(), "", M);
  NewF->copyAttributesFrom(&F);
  AttributeList OldAL = NewF->getAttributes();
  NewF->setAttributes(AttributeList::get(C, OldAL.getFnAttrs(),
                                          OldAL.getRetAttrs(), {}));
  NewF->setCallingConv(F.getCallingConv());
  NewF->takeName(&F);
  NewF->splice(NewF->begin(), &F);

  // Build push-constant struct: each kernel arg becomes either a uint64
  // (pointer args) or its native type (scalar / byval-struct).
  SmallVector<Type *, 8> FieldTypes;
  SmallVector<bool, 8> IsPtrField;
  for (Argument &A : F.args()) {
    auto *PT = dyn_cast<PointerType>(A.getType());
    bool IsBufferPtr = PT && !A.hasByValAttr() &&
                       (PT->getAddressSpace() == 1 ||
                        PT->getAddressSpace() == 4 ||
                        PT->getAddressSpace() == 11);
    if (IsBufferPtr) {
      FieldTypes.push_back(I64Ty);
      IsPtrField.push_back(true);
    } else {
      Type *Ty = A.hasByValAttr() ? A.getParamByValType() : A.getType();
      FieldTypes.push_back(Ty);
      IsPtrField.push_back(false);
    }
  }

  StructType *PCS = StructType::create(C, FieldTypes,
                                       (NewF->getName() + ".pcs").str());
  auto *PushConstGV = new GlobalVariable(
      *M, PCS, /*isConstant=*/false, GlobalValue::ExternalLinkage,
      /*Initializer=*/nullptr,
      ".hipspv.pcs." + NewF->getName(),
      /*InsertBefore=*/nullptr, GlobalValue::NotThreadLocal,
      /*AddressSpace=*/13);
  PushConstGV->setExternallyInitialized(true);
  PushConstGV->setVisibility(GlobalValue::HiddenVisibility);

  IRBuilder<> B(&NewF->getEntryBlock(),
                NewF->getEntryBlock().getFirstInsertionPt());

  unsigned Idx = 0;
  for (Argument &A : F.args()) {
    B.SetInsertPoint(&NewF->getEntryBlock(),
                     NewF->getEntryBlock().getFirstInsertionPt());
    Value *FieldPtr = B.CreateGEP(
        PCS, PushConstGV,
        {ConstantInt::get(I32Ty, 0), ConstantInt::get(I32Ty, Idx)});
    if (IsPtrField[Idx]) {
      // Load uint64 BDA, convert to PhysicalStorageBuffer pointer
      // (addrspace(5)).
      Value *Addr = B.CreateAlignedLoad(I64Ty, FieldPtr, MaybeAlign(8));
      auto *PT = cast<PointerType>(A.getType());
      auto *PSBPtrTy = PointerType::get(C, 5);
      Value *PSBPtr = B.CreateIntToPtr(Addr, PSBPtrTy);
      if (PT->getAddressSpace() == 5) {
        A.replaceAllUsesWith(PSBPtr);
      } else {
        Value *Cast = B.CreateAddrSpaceCast(PSBPtr, PT);
        A.replaceAllUsesWith(Cast);
      }
    } else if (A.hasByValAttr()) {
      Value *Cast = (A.getType() == FieldPtr->getType())
                        ? FieldPtr
                        : (Value *)B.CreateAddrSpaceCast(FieldPtr,
                                                          A.getType());
      A.replaceAllUsesWith(Cast);
    } else {
      Value *Loaded = B.CreateAlignedLoad(A.getType(), FieldPtr, MaybeAlign());
      A.replaceAllUsesWith(Loaded);
    }
    ++Idx;
  }

  F.eraseFromParent();
  return NewF;
}

static Function *rewriteKernelSignature(Function &F) {
  if (F.getCallingConv() != CallingConv::SPIR_FUNC ||
      !F.hasFnAttribute("hlsl.shader") || F.arg_empty())
    return &F;

  // Phase F: route to BDA lowering if the module has IGBAs.
  if (moduleHasIGBAs(*F.getParent()))
    return rewriteKernelSignatureBDA(F);

  LLVMContext &C = F.getContext();
  Module *M = F.getParent();
  auto *VoidTy = Type::getVoidTy(C);
  auto *NewFTy = FunctionType::get(VoidTy, {}, false);

  Function *NewF =
      Function::Create(NewFTy, F.getLinkage(), F.getAddressSpace(), "", M);
  NewF->copyAttributesFrom(&F);
  // Drop per-parameter attributes that copyAttributesFrom carried over —
  // they refer to slots that no longer exist in the void() signature.
  AttributeList OldAL = NewF->getAttributes();
  NewF->setAttributes(AttributeList::get(C, OldAL.getFnAttrs(),
                                          OldAL.getRetAttrs(), {}));
  NewF->setCallingConv(F.getCallingConv());
  NewF->takeName(&F);

  NewF->splice(NewF->begin(), &F);

  IRBuilder<> B(&NewF->getEntryBlock(),
                NewF->getEntryBlock().getFirstInsertionPt());

  auto *I8Ty = Type::getInt8Ty(C);
  auto *I32Ty = Type::getInt32Ty(C);
  auto *RuntimeArrTy = ArrayType::get(I8Ty, 0);
  auto *PtrSB = PointerType::get(C, 11); // SPIR-V StorageBuffer storage class

  // Distinguish kernel arg shapes:
  //  * Buffer args   : `ptr addrspace(N)` where N is a global memory space
  //                    (CrossWorkgroup=1, Generic=4) — these are HIP/CUDA
  //                    `T*` arguments, lowered to `target("spirv.VulkanBuffer")`
  //                    StorageBuffer descriptors.
  //  * POD args      : non-pointer types OR `byval(struct)` pointers (the
  //                    Itanium ABI passes structs by hidden pointer with
  //                    `byval` attr — semantically pass-by-value, gets pushed
  //                    into the push-constant block).
  auto isBufferArg = [](Argument &A) -> bool {
    if (A.hasByValAttr())
      return false;
    auto *PT = dyn_cast<PointerType>(A.getType());
    if (!PT)
      return false;
    unsigned AS = PT->getAddressSpace();
    return AS == 1 || AS == 4 || AS == 11;
  };

  // Phase 3c: cluster non-pointer / byval args into a single push-constant
  // struct. Build the struct type, materialise an `addrspace(13)` global of
  // that type, and route each scalar arg's uses to a GEP+load on that global.
  SmallVector<Argument *, 4> ScalarArgs;
  SmallVector<Type *, 4> ScalarTypes;
  for (Argument &A : F.args()) {
    if (!isBufferArg(A)) {
      Type *Ty = A.getType();
      if (A.hasByValAttr())
        Ty = A.getParamByValType();
      ScalarArgs.push_back(&A);
      ScalarTypes.push_back(Ty);
    }
  }
  GlobalVariable *PushConstGV = nullptr;
  StructType *PushConstStruct = nullptr;
  if (!ScalarArgs.empty()) {
    PushConstStruct = StructType::create(C, ScalarTypes,
                                         (NewF->getName() + ".pcs").str());
    PushConstGV = new GlobalVariable(
        *M, PushConstStruct, /*isConstant=*/false,
        GlobalValue::ExternalLinkage, /*Initializer=*/nullptr,
        ".hipspv.pcs." + NewF->getName(),
        /*InsertBefore=*/nullptr,
        GlobalValue::NotThreadLocal,
        /*AddressSpace=*/13); // SPIR-V PushConstant storage class
    PushConstGV->setExternallyInitialized(true);
    PushConstGV->setVisibility(GlobalValue::HiddenVisibility);
  }

  unsigned Binding = 0;
  unsigned ScalarIdx = 0;
  for (Argument &A : F.args()) {
    auto *PT = dyn_cast<PointerType>(A.getType());
    if (PT && isBufferArg(A)) {
      // If the kernel never uses this buffer arg, skip materialising a
      // descriptor for it — the backend's instruction selector chokes on
      // unused Vulkan-buffer handles.
      if (A.use_empty()) {
        ++Binding;
        continue;
      }
      // Infer the buffer's element type from kernel load/store widths so the
      // backend can emit OpAccessChain instead of OpPtrAccessChain (the latter
      // requires Vulkan-forbidden Capability Addresses).
      Type *ElemTy = inferArgAccessType(A);
      Type *RuntimeArr = ArrayType::get(ElemTy, 0);
      // StorageBuffer storage class id 12; writable bit 1 (HIP doesn't yet
      // distinguish const/restrict at this layer).
      Type *VkBufTy = TargetExtType::get(C, "spirv.VulkanBuffer",
                                         {RuntimeArr}, {12, 1});
      // The backend's instruction selector expects a non-null global string
      // for the resource debug-name. Synthesise a unique private global per
      // binding and pass that pointer.
      std::string DbgName =
          (NewF->getName() + "." + Twine(Binding)).str();
      auto *NameInit =
          ConstantDataArray::getString(C, DbgName, /*AddNull=*/true);
      auto *NameGV = new GlobalVariable(
          *M, NameInit->getType(), /*isConstant=*/true,
          GlobalValue::PrivateLinkage, NameInit,
          ".hipspv.argname." + Twine(Binding) + "." + NewF->getName(),
          /*InsertBefore=*/nullptr,
          GlobalValue::NotThreadLocal,
          /*AddressSpace=*/0);
      NameGV->setUnnamedAddr(GlobalValue::UnnamedAddr::Global);
      SmallVector<Value *, 5> HandleArgs{
          ConstantInt::get(I32Ty, 0),       // descriptor set
          ConstantInt::get(I32Ty, Binding), // binding
          ConstantInt::get(I32Ty, 1),       // range
          ConstantInt::get(I32Ty, 0),       // index
          NameGV};
      // Re-anchor the IRBuilder before every CreateIntrinsic. The previous
      // iteration may have called rewriteToStorageBuffer which erases dead
      // addrspacecasts/GEPs in the entry block — including, sometimes, the
      // very instruction B was pointing at. Anchor at first-insertion-pt
      // each time so the iterator stays valid.
      B.SetInsertPoint(&NewF->getEntryBlock(),
                       NewF->getEntryBlock().getFirstInsertionPt());
      Value *Handle = B.CreateIntrinsic(
          VkBufTy, Intrinsic::spv_resource_handlefrombinding, HandleArgs);
      Value *P11 = B.CreateIntrinsic(
          PtrSB, Intrinsic::spv_resource_getpointer,
          {Handle, ConstantInt::get(I32Ty, 0)});
      Value *ReplVal;
      if (PT->getAddressSpace() == 11) {
        ReplVal = P11;
        A.replaceAllUsesWith(ReplVal);
      } else {
        Value *Cast = B.CreateAddrSpaceCast(P11, PT);
        A.replaceAllUsesWith(Cast);
        // Phase 3b: rewrite GEP/Load/Store/AddrSpaceCast users so the bridging
        // cast can vanish; otherwise the backend can't select G_ADDRSPACE_CAST.
        rewriteToStorageBuffer(Handle, VkBufTy, ElemTy, P11, Cast);
      }
      ++Binding;
    } else {
      assert(PushConstGV && "scalar arg without push-constant block");
      // Re-anchor the builder at the entry block before each insert; prior
      // rewrites (rewriteToStorageBuffer) may have erased the instruction B
      // was pointing at.
      IRBuilder<> PCB(&NewF->getEntryBlock(),
                      NewF->getEntryBlock().getFirstInsertionPt());
      Value *FieldPtr = PCB.CreateGEP(
          PushConstStruct, PushConstGV,
          {ConstantInt::get(I32Ty, 0), ConstantInt::get(I32Ty, ScalarIdx)});
      if (A.hasByValAttr()) {
        // The arg is a hidden pointer to a struct; uses access fields via
        // GEPs/loads off the pointer. RAUW requires matching types; A's ptr
        // is in the default addrspace, FieldPtr is in addrspace(13). Bridge
        // via an addrspacecast and let the propagator rewrite GEPs/loads to
        // live in addrspace(13) so the cast can disappear.
        Value *Cast = (A.getType() == FieldPtr->getType())
                          ? FieldPtr
                          : (Value *)PCB.CreateAddrSpaceCast(FieldPtr,
                                                             A.getType());
        A.replaceAllUsesWith(Cast);
        // Inferring an aggregate's element type isn't meaningful for push
        // constants — we pass null Handle/Type to the propagator and let it
        // walk through GEP/Load/Store users with the new addrspace(13) base.
        if (Cast != FieldPtr)
          rewriteToStorageBuffer(/*Handle=*/nullptr, /*VkBufTy=*/nullptr,
                                 /*ElemTy=*/nullptr, FieldPtr, Cast);
      } else {
        Value *Loaded = PCB.CreateAlignedLoad(A.getType(), FieldPtr,
                                              MaybeAlign());
        A.replaceAllUsesWith(Loaded);
      }
      ++ScalarIdx;
    }
  }

  // Post-processing: any memcpy whose source folded to the PushConstant
  // global directly (constant GEP `{0,0}` collapses) will hit the backend
  // as `OpCopyMemory` between mismatched pointee types (PCS-wrapper-struct
  // vs first-field type). Rewrite those to `load PCS-field + store dst` so
  // the backend emits OpLoad/OpStore on matching types.
  if (PushConstGV) {
    SmallVector<MemTransferInst *, 4> ToRewrite;
    for (BasicBlock &BB : *NewF)
      for (Instruction &I : BB)
        if (auto *MTI = dyn_cast<MemTransferInst>(&I))
          if (MTI->getRawSource()->stripPointerCasts() == PushConstGV)
            ToRewrite.push_back(MTI);
    for (MemTransferInst *MTI : ToRewrite) {
      // Use the PCS field type directly (we know which scalar arg this came
      // from by matching size; for the common single-byval-arg case, field 0
      // holds the struct). Find the field index whose alloc size equals the
      // memcpy length.
      auto *LenC = dyn_cast<ConstantInt>(MTI->getLength());
      if (!LenC)
        continue;
      uint64_t Len = LenC->getZExtValue();
      const DataLayout &DL = NewF->getDataLayout();
      int FieldIdx = -1;
      for (unsigned Fi = 0, Fe = PushConstStruct->getNumElements();
           Fi < Fe; ++Fi) {
        Type *FT = PushConstStruct->getElementType(Fi);
        if (DL.getTypeStoreSize(FT).getFixedValue() == Len) {
          FieldIdx = (int)Fi;
          break;
        }
      }
      if (FieldIdx < 0)
        continue;
      Type *FT = PushConstStruct->getElementType(FieldIdx);
      IRBuilder<> RB(MTI);
      // Non-constant-foldable indexing isn't required: even a constant GEP
      // is fine here because we are rewriting the memcpy itself into
      // load+store. The load/store types ARE the field type, so the backend
      // sees matching pointee types.
      // Force a non-folding GEP: build an InBoundsGEP via Builder, but since
      // {0,FieldIdx} on a global folds to a ConstantExpr (and ultimately to
      // the global itself when FieldIdx==0), we instead emit an Instruction
      // GEP with a runtime-zero index. The trick: create a select on a
      // dummy `false` so the GEP's first index isn't a literal Constant, but
      // still simplifies to 0 at runtime. Even simpler: emit a GEP, then if
      // the result is a Constant, fall back to a load-of-wrapper followed by
      // extractvalue of the field.
      Value *FieldPtr = RB.CreateGEP(
          PushConstStruct, PushConstGV,
          {ConstantInt::get(I32Ty, 0), ConstantInt::get(I32Ty, FieldIdx)});
      if (isa<Constant>(FieldPtr)) {
        // Constant-folded to the wrapper (or a wrapper-typed pointer the
        // backend then can't `load` directly without a ptrcast). Emit
        // load(wrapper) + extractvalue + store instead.
        Value *Whole = RB.CreateAlignedLoad(PushConstStruct, PushConstGV,
                                            MaybeAlign());
        Value *Field = RB.CreateExtractValue(Whole, {(unsigned)FieldIdx});
        RB.CreateAlignedStore(Field, MTI->getRawDest(), MaybeAlign());
      } else {
        Value *Loaded = RB.CreateAlignedLoad(FT, FieldPtr, MaybeAlign());
        RB.CreateAlignedStore(Loaded, MTI->getRawDest(), MaybeAlign());
      }
      MTI->eraseFromParent();
    }
  }

  F.eraseFromParent();
  return NewF;
}

// Diagnostic gate: any leftover position-builtin call after the rewrite is a
// missed pattern; surface it loudly so we know to extend the matcher.
static void diagnoseLeftovers(Module &M) {
  for (Function &F : M) {
    for (BasicBlock &BB : F) {
      for (Instruction &I : BB) {
        auto *CI = dyn_cast<CallInst>(&I);
        if (!CI || !CI->getCalledFunction())
          continue;
        StringRef Name = CI->getCalledFunction()->getName();
        if (Name.contains("__cuda_builtin_")) {
          report_fatal_error(
              Twine("hipspv-lower-to-hlsl-shape: unhandled CUDA builtin call '")
                  + Name + "'");
        }
      }
    }
  }
}

} // namespace

// Strips chipStar host-runtime keep-alive globals plus the runtime
// declarations they pin (`@printf`, `@__chipspv_abort`, etc.). Those decls
// would otherwise survive into the SPIR-V via `LinkageAttributes Import/Export`
// decoration, which Vulkan's spec rejects.
//
// Strategy:
//   1. Drop `llvm.compiler.used` / `llvm.used`.
//   2. Replace any call to `__assert_fail` / `__chipspv_abort` / `printf` with
//      a no-op (or `unreachable` for abort), then delete the declaration.
//   3. Delete `__chipspv_*` and `__hip_cuid_*` globals if their only remaining
//      uses are the now-dead keep-alives.
static bool stripChipStarRuntime(Module &M) {
  bool Changed = false;
  for (StringRef Name : {"llvm.compiler.used", "llvm.used"}) {
    if (auto *GV = M.getNamedGlobal(Name)) {
      GV->eraseFromParent();
      Changed = true;
    }
  }

  // Replace calls to assert/abort/printf with no-op. Walk users carefully
  // because erasing changes the use list.
  auto neuter = [&](StringRef FName) {
    Function *F = M.getFunction(FName);
    if (!F)
      return;
    SmallVector<CallInst *, 8> Calls;
    for (User *U : F->users())
      if (auto *CI = dyn_cast<CallInst>(U))
        if (CI->getCalledFunction() == F)
          Calls.push_back(CI);
    for (CallInst *CI : Calls) {
      if (!CI->getType()->isVoidTy())
        CI->replaceAllUsesWith(Constant::getNullValue(CI->getType()));
      CI->eraseFromParent();
      Changed = true;
    }
  };
  neuter("printf");
  neuter("__chipspv_abort");
  neuter("__assert_fail");

  // Now erase the function declarations / weak defs themselves.
  for (StringRef FName : {"printf", "__chipspv_abort", "__assert_fail"}) {
    if (Function *F = M.getFunction(FName)) {
      if (F->use_empty()) {
        F->eraseFromParent();
        Changed = true;
      }
    }
  }

  // Delete chipStar runtime globals if they have no remaining uses.
  // ConstantExprs that referenced these (from llvm.compiler.used) survive in
  // the constant pool because of uniqueness, so kill those off first via
  // removeDeadConstantUsers.
  SmallVector<GlobalVariable *, 4> ToDeleteGV;
  for (GlobalVariable &GV : M.globals()) {
    StringRef N = GV.getName();
    if (N.starts_with("__chipspv_") || N.starts_with("__hip_cuid_") ||
        N == ".str")
      ToDeleteGV.push_back(&GV);
  }
  for (GlobalVariable *GV : ToDeleteGV) {
    GV->removeDeadConstantUsers();
    if (GV->use_empty()) {
      GV->eraseFromParent();
      Changed = true;
    }
  }
  return Changed;
}

// Erases module-scope globals in non-Vulkan-compatible address spaces (e.g.
// `addrspace(1)` CrossWorkgroup) once they have no remaining uses. The
// canonical case is `__chip_clk_counter` left behind after the kernel that
// referenced it (waitKernel) was dropped: the backend would otherwise emit
// `OpTypePointer CrossWorkgroup` for it, which spirv-val rejects in Vulkan.
static bool dropDeadInvalidAddrSpaceGlobals(Module &M) {
  bool Changed = false;
  SmallVector<GlobalVariable *, 8> ToDelete;
  for (GlobalVariable &GV : M.globals()) {
    unsigned AS = GV.getType()->getAddressSpace();
    // Vulkan-legal address spaces for globals: 11 (StorageBuffer),
    // 13 (PushConstant), 0 (Function/private constants like name strings).
    if (AS == 11 || AS == 13 || AS == 0)
      continue;
    GV.removeDeadConstantUsers();
    if (GV.use_empty())
      ToDelete.push_back(&GV);
  }
  for (GlobalVariable *GV : ToDelete) {
    GV->eraseFromParent();
    Changed = true;
  }
  return Changed;
}

// True if `CE` is an AddrSpaceCast or transitively contains one in its
// constant-expr operand tree.
static bool ceContainsAddrSpaceCast(const ConstantExpr *CE) {
  if (CE->getOpcode() == Instruction::AddrSpaceCast)
    return true;
  for (const Use &U : CE->operands())
    if (auto *Sub = dyn_cast<ConstantExpr>(&*U))
      if (ceContainsAddrSpaceCast(Sub))
        return true;
  return false;
}

// Materializes `CE` as a fresh Instruction inserted before `InsertBefore`.
// Recursively does the same for any sub-ConstantExpr operands that contain
// an AddrSpaceCast, so the returned instruction has no CE addrspacecast in
// its operand tree.
static Instruction *materializeCE(ConstantExpr *CE,
                                  Instruction *InsertBefore) {
  Instruction *NewI = CE->getAsInstruction();
  NewI->insertBefore(InsertBefore->getIterator());
  for (unsigned i = 0, e = NewI->getNumOperands(); i != e; ++i) {
    if (auto *OpCE = dyn_cast<ConstantExpr>(NewI->getOperand(i))) {
      if (ceContainsAddrSpaceCast(OpCE)) {
        Instruction *OpI = materializeCE(OpCE, NewI);
        NewI->setOperand(i, OpI);
      }
    }
  }
  return NewI;
}

// Walks every instruction in F and converts each ConstantExpr operand that
// contains an AddrSpaceCast into instruction-form. PHI nodes get their
// materialized instructions placed at the end of the corresponding
// predecessor block (CE-as-instruction must dominate the PHI).
//
// The Vulkan/Logical SPIR-V backend has no ISel rule for ConstantExpr
// addrspacecasts, but instruction-form addrspacecasts can be propagated
// through `rewriteToStorageBuffer` or eliminated as dead code if the
// downstream uses turn out to be unreachable post-inlining.
static bool materializeAddrSpaceConstants(Function &F) {
  bool Changed = false;
  for (BasicBlock &BB : F) {
    SmallVector<Instruction *, 16> Insts;
    for (Instruction &I : BB)
      Insts.push_back(&I);
    for (Instruction *I : Insts) {
      if (auto *PHI = dyn_cast<PHINode>(I)) {
        for (unsigned k = 0, e = PHI->getNumIncomingValues(); k != e; ++k) {
          auto *CE = dyn_cast<ConstantExpr>(PHI->getIncomingValue(k));
          if (!CE || !ceContainsAddrSpaceCast(CE))
            continue;
          BasicBlock *Pred = PHI->getIncomingBlock(k);
          Instruction *NewI = materializeCE(CE, Pred->getTerminator());
          PHI->setIncomingValue(k, NewI);
          Changed = true;
        }
        continue;
      }
      for (unsigned op = 0, e = I->getNumOperands(); op != e; ++op) {
        auto *CE = dyn_cast<ConstantExpr>(I->getOperand(op));
        if (!CE || !ceContainsAddrSpaceCast(CE))
          continue;
        Instruction *NewI = materializeCE(CE, I);
        I->setOperand(op, NewI);
        Changed = true;
      }
    }
  }
  return Changed;
}

// Simple intra-function DCE: erases instructions that are use-empty and
// trivially dead (no side effects, not a terminator). Iterates until stable.
// Used after CE-to-instruction conversion so dead loads of chipStar runtime
// globals (e.g. `__chip_clk_counter` in kernels that don't call `clock()`)
// disappear before the unsupported-addrspace check.
static bool runIntraFnDCE(Function &F) {
  bool AnyChange = false;
  bool Progress = true;
  while (Progress) {
    Progress = false;
    SmallVector<Instruction *, 32> Dead;
    for (BasicBlock &BB : F)
      for (Instruction &I : BB)
        if (isInstructionTriviallyDead(&I))
          Dead.push_back(&I);
    for (Instruction *I : Dead) {
      I->eraseFromParent();
      Progress = true;
      AnyChange = true;
    }
  }
  return AnyChange;
}

// Returns true if the function contains a pattern the Vulkan/Logical backend
// can't lower. Used as a backstop AFTER `materializeAddrSpaceConstants` and
// DCE — at that point most CE addrspacecasts are gone, but two stragglers
// can remain:
//   1. Residual CE addrspacecast we couldn't reach (rare).
//   2. Instruction-form addrspacecast whose source is a GlobalValue in a
//      non-StorageBuffer addrspace. These come from chipStar runtime helper
//      globals (`__chip_clk_counter`, `__chipspv_*`) that have no Vulkan
//      equivalent. The kernel that touches them is always a runtime helper
//      (waitKernel), never user code.
static bool hasUnsupportedAddrSpaceConst(const Function &F) {
  for (const BasicBlock &BB : F) {
    for (const Instruction &I : BB) {
      // (1) ConstantExpr operand check.
      for (const Use &U : I.operands())
        if (auto *CE = dyn_cast<ConstantExpr>(&*U))
          if (ceContainsAddrSpaceCast(CE))
            return true;
      // (2) Instruction addrspacecast on a global value source.
      if (auto *AC = dyn_cast<AddrSpaceCastInst>(&I)) {
        if (isa<GlobalValue>(AC->getPointerOperand())) {
          unsigned SrcAS = AC->getSrcAddressSpace();
          // StorageBuffer (11) and PushConstant (13) are the legitimate
          // Vulkan-side addrspaces; addrspace(1) CrossWorkgroup on a global
          // can't be lowered in Logical mode.
          if (SrcAS != 11 && SrcAS != 13)
            return true;
        }
      }
    }
  }
  return false;
}

// Walk back through ptrtoint / addrspacecast / bitcast to recover the
// originating typed pointer of an i64 value. Returns nullptr if the chain
// does not reach a pointer source. The chipStar Vulkan path produces:
//   %2 = call ptr addrspace(11) @llvm.spv.resource.getpointer...
//   %3 = addrspacecast ptr addrspace(11) %2 to ptr addrspace(1)
//   %4 = ptrtoint ptr addrspace(1) %3 to i64
//   call float @__chip_atomic_add_f32(i64 %4, float %v)
// We want to recover %2 (the StorageBuffer pointer) for atomicrmw.
static Value *tracePtrFromI64(Value *V) {
  Value *Cur = V;
  while (Cur) {
    if (auto *PTI = dyn_cast<PtrToIntInst>(Cur)) {
      Cur = PTI->getPointerOperand();
      continue;
    }
    if (auto *ASC = dyn_cast<AddrSpaceCastInst>(Cur)) {
      Cur = ASC->getPointerOperand();
      continue;
    }
    if (auto *BC = dyn_cast<BitCastInst>(Cur)) {
      Cur = BC->getOperand(0);
      continue;
    }
    if (Cur->getType()->isPointerTy())
      return Cur;
    return nullptr;
  }
  return nullptr;
}

// Replace calls to chipStar's external `__chip_atomic_add_f{32,64}`
// helpers with native `atomicrmw fadd`. The bridging pass runs before the
// SPIR-V backend's instruction selector, which knows how to lower
// `atomicrmw fadd` to `OpAtomicFAddEXT` and request the
// AtomicFloat{32,64}AddEXT capability + SPV_EXT_shader_atomic_float_add
// extension automatically.
static bool lowerChipAtomicFloatAdd(Module &M) {
  bool Changed = false;
  for (StringRef FName :
       {"__chip_atomic_add_f32", "__chip_atomic_add_f64"}) {
    Function *F = M.getFunction(FName);
    if (!F)
      continue;
    SmallVector<CallInst *, 8> Calls;
    for (User *U : F->users())
      if (auto *CI = dyn_cast<CallInst>(U))
        if (CI->getCalledFunction() == F)
          Calls.push_back(CI);
    for (CallInst *CI : Calls) {
      if (CI->arg_size() != 2)
        continue;
      Value *PtrI64 = CI->getArgOperand(0);
      Value *Val = CI->getArgOperand(1);
      Value *Ptr = tracePtrFromI64(PtrI64);
      if (!Ptr) {
        LLVM_DEBUG(dbgs() << "hipspv: cannot trace pointer for "
                          << FName << " call - leaving as-is\n");
        continue;
      }
      // Sanity: pointer's pointee width must match the value type. We use
      // typeless pointers here, so just trust the value type.
      if (!Val->getType()->isFloatingPointTy())
        continue;
      IRBuilder<> B(CI);
      // System-scope, monotonic ordering matches HIP atomicAdd semantics.
      auto *RMW = B.CreateAtomicRMW(
          AtomicRMWInst::FAdd, Ptr, Val, MaybeAlign(),
          AtomicOrdering::Monotonic, SyncScope::System);
      CI->replaceAllUsesWith(RMW);
      CI->eraseFromParent();
      Changed = true;
    }
    if (F->use_empty()) {
      F->eraseFromParent();
      Changed = true;
    }
  }
  return Changed;
}

PreservedAnalyses HIPSPVLowerToHLSLShape::run(Module &M,
                                              ModuleAnalysisManager &) {
  bool Changed = false;
  Changed |= stripChipStarRuntime(M);
  // Phase D: lower chipStar's __chip_atomic_add_f32/f64 helpers to
  // atomicrmw fadd so the SPIR-V backend can emit OpAtomicFAddEXT in
  // Shader mode (SPV_EXT_shader_atomic_float_add).
  Changed |= lowerChipAtomicFloatAdd(M);

  // Phase A: convert ConstantExpr addrspacecasts to instruction form across
  // every defined function, then run a simple intra-function DCE to remove
  // dead loads/stores of chipStar runtime globals (`__chip_clk_counter` etc.)
  // that survived inlining but aren't actually observed.
  for (Function &F : M) {
    if (F.isDeclaration())
      continue;
    Changed |= materializeAddrSpaceConstants(F);
    Changed |= runIntraFnDCE(F);
  }

  // Backstop: any spir_kernel that STILL has a CE addrspacecast (not reached
  // by the conversion above, or with a live use the backend can't lower) is
  // dropped with a debug log. After Phase A the conversion handles the
  // canonical waitKernel/log2f patterns; remaining drops should be rare.
  SmallVector<Function *, 4> ToDrop;
  for (Function &F : M) {
    if (F.isDeclaration())
      continue;
    if (F.getCallingConv() != CallingConv::SPIR_KERNEL)
      continue;
    if (hasUnsupportedAddrSpaceConst(F)) {
      LLVM_DEBUG(dbgs() << "hipspv: dropping kernel '" << F.getName()
                        << "' - residual ConstantExpr addrspacecast after "
                           "Phase A materialization\n");
      ToDrop.push_back(&F);
    }
  }
  for (Function *F : ToDrop) {
    F->dropAllReferences();
    F->eraseFromParent();
    Changed = true;
  }

  // First pass: stamp shader attrs + flip CC. Done before signature rewrite so
  // rewriteKernelSignature can recognise its candidates by attribute.
  for (Function &F : M) {
    if (F.isDeclaration())
      continue;
    Changed |= stampShaderAttrs(F);
    Changed |= remapMemFnsToIntrinsics(F);
    Changed |= remapOpenCLMathToIntrinsics(F);
    Changed |= lowerI64BitcountIntrinsics(F);
    Changed |= lowerPositionBuiltinsInFunction(F);
  }
  // Promote clang -O0 arg-coerce alloca-shuffles to direct value forwarding,
  // run an extra DCE round to drop dead stores/casts, and re-iterate so that
  // chained shuffles (slot1 → slot2 → user) collapse fully. This must run
  // BEFORE signature rewrite because the pointer-arg propagator can't
  // reconcile addrspace(11)-stored-into-addrspace(4)-slot patterns that SROA
  // also refuses (alloca element addrspace differs from stored pointer's).
  for (Function &F : M) {
    if (F.isDeclaration() || F.empty())
      continue;
    bool Local = true;
    while (Local) {
      Local = false;
      Local |= promoteAllocaShuffles(F);
      Local |= runIntraFnDCE(F);
      if (Local)
        Changed = true;
    }
  }

  // Second pass: collect candidates for signature rewrite, then rewrite. Cannot
  // mutate the module while iterating its function list.
  SmallVector<Function *, 4> KernelCandidates;
  for (Function &F : M) {
    if (F.isDeclaration())
      continue;
    if (F.hasFnAttribute("hlsl.shader") && !F.arg_empty())
      KernelCandidates.push_back(&F);
  }
  for (Function *F : KernelCandidates) {
    rewriteKernelSignature(*F);
    Changed = true;
  }

  // After signatures are rewritten the StorageBuffer pointers from
  // llvm.spv.resource.getpointer are visible. Lower chipStar's
  // `__chip_atomic_add_f{32,64}` helpers to native `atomicrmw fadd` so the
  // SPIR-V backend can emit OpAtomicFAddEXT.
  Changed |= lowerChipAtomicFloatAdd(M);

  // After rewriteKernelSignature: IRBuilder constant-folding while emitting
  // GEPs/addrspacecasts on the push-constant global may have produced new
  // ConstantExpr addrspacecasts (especially around byval struct args).
  // Re-materialize them and run DCE again so the backend sees only
  // instruction-form addrspacecasts.
  for (Function &F : M) {
    if (F.isDeclaration() || F.empty())
      continue;
    Changed |= materializeAddrSpaceConstants(F);
    Changed |= runIntraFnDCE(F);
  }

  // Final cleanup: drop any module-scope globals in non-Vulkan addrspaces
  // that became dead after the kernel-drop + bridging-pass rewrites
  // (e.g. `__chip_clk_counter` in CrossWorkgroup after waitKernel removal).
  Changed |= dropDeadInvalidAddrSpaceGlobals(M);

  diagnoseLeftovers(M);
  return Changed ? PreservedAnalyses::none() : PreservedAnalyses::all();
}
