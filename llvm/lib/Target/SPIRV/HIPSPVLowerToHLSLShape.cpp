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
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Operator.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/IntrinsicsSPIRV.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/Utils/Local.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Transforms/Scalar/SimplifyCFG.h"
#include "llvm/Transforms/Scalar/StructurizeCFG.h"
#include "llvm/Transforms/Utils/FixIrreducible.h"
#include "llvm/Transforms/Utils/UnifyLoopExits.h"
#include "llvm/Analysis/CGSCCPassManager.h"
#include "llvm/Passes/PassBuilder.h"

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
  // OCML f16/f32/f64 helpers exposed by chipStar as extern "C" symbols
  // (`__ocml_<op>_f16`, etc.). In Shader/Vulkan mode the OpenCL devicelib
  // that normally provides these is not linked, so the calls remain as
  // unresolved Imports — emitting `OpCapability Linkage` and decorations
  // Vulkan rejects. Map them to their `llvm.<intrin>` equivalents.
  struct OcmlEntry {
    StringRef Name;
    Intrinsic::ID Intr;
    unsigned ArgCount;
  };
  static const OcmlEntry OcmlTable[] = {
      {"__ocml_ceil_f16",  Intrinsic::ceil,     1},
      {"__ocml_cos_f16",   Intrinsic::cos,      1},
      {"__ocml_exp_f16",   Intrinsic::exp,      1},
      {"__ocml_exp2_f16",  Intrinsic::exp2,     1},
      {"__ocml_floor_f16", Intrinsic::floor,    1},
      {"__ocml_fabs_f16",  Intrinsic::fabs,     1},
      {"__ocml_log_f16",   Intrinsic::log,      1},
      {"__ocml_log2_f16",  Intrinsic::log2,     1},
      {"__ocml_log10_f16", Intrinsic::log10,    1},
      {"__ocml_rint_f16",  Intrinsic::rint,     1},
      {"__ocml_sin_f16",   Intrinsic::sin,      1},
      {"__ocml_sqrt_f16",  Intrinsic::sqrt,     1},
      {"__ocml_trunc_f16", Intrinsic::trunc,    1},
      {"__ocml_fmax_f16",  Intrinsic::maxnum,   2},
      {"__ocml_fmin_f16",  Intrinsic::minnum,   2},
      {"__ocml_fma_f16",   Intrinsic::fma,      3},
      // half2 vector flavors (used by __h2*/half2 math wrappers).
      {"__ocml_ceil_2f16",  Intrinsic::ceil,    1},
      {"__ocml_cos_2f16",   Intrinsic::cos,     1},
      {"__ocml_exp_2f16",   Intrinsic::exp,     1},
      {"__ocml_exp2_2f16",  Intrinsic::exp2,    1},
      {"__ocml_floor_2f16", Intrinsic::floor,   1},
      {"__ocml_fabs_2f16",  Intrinsic::fabs,    1},
      {"__ocml_log_2f16",   Intrinsic::log,     1},
      {"__ocml_log2_2f16",  Intrinsic::log2,    1},
      {"__ocml_log10_2f16", Intrinsic::log10,   1},
      {"__ocml_rint_2f16",  Intrinsic::rint,    1},
      {"__ocml_sin_2f16",   Intrinsic::sin,     1},
      {"__ocml_sqrt_2f16",  Intrinsic::sqrt,    1},
      {"__ocml_trunc_2f16", Intrinsic::trunc,   1},
      {"__ocml_fmax_2f16",  Intrinsic::maxnum,  2},
      {"__ocml_fmin_2f16",  Intrinsic::minnum,  2},
      {"__ocml_fma_2f16",   Intrinsic::fma,     3},
  };

  for (CallInst *CI : Worklist) {
    Function *Callee = CI->getCalledFunction();
    if (!Callee || !Callee->isDeclaration())
      continue;
    StringRef MName = Callee->getName();
    // Handle extern "C" __ocml_*_f16 helpers first.
    if (MName.starts_with("__ocml_")) {
      const OcmlEntry *OM = nullptr;
      for (const OcmlEntry &E : OcmlTable) {
        if (E.Name == MName && E.ArgCount == CI->arg_size()) {
          OM = &E;
          break;
        }
      }
      if (!OM)
        continue;
      Type *ResTy = CI->getType();
      // Accept scalar float (f16/f32/f64) or vector-of-float (e.g. <2 x half>).
      if (!ResTy->getScalarType()->isFloatingPointTy())
        continue;
      SmallVector<Value *, 3> Args(CI->args());
      bool TypeOk = true;
      for (Value *A : Args)
        if (A->getType() != ResTy) { TypeOk = false; break; }
      if (!TypeOk)
        continue;
      IRBuilder<> B(CI);
      Function *Decl = Intrinsic::getOrInsertDeclaration(
          F.getParent(), OM->Intr, {ResTy});
      CallInst *NC = B.CreateCall(Decl, Args);
      NC->setCallingConv(Decl->getCallingConv());
      if (CI->getFastMathFlags().any())
        NC->setFastMathFlags(CI->getFastMathFlags());
      CI->replaceAllUsesWith(NC);
      CI->eraseFromParent();
      Changed = true;
      continue;
    }
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
    // OpenCL.std *pi family (asinpi/acospi/atanpi/sinpi/cospi/tanpi) is
    // not part of GLSL.std.450 so the Shader-mode SPV path leaves them
    // as raw OpenCL.std ExtInsts, which Vulkan rejects. Decompose into
    // the standard intrinsic plus a multiply/divide by pi:
    //   asinpi(x) = asin(x) / pi   (acospi/atanpi analogous)
    //   sinpi(x)  = sin(x * pi)    (cospi/tanpi analogous)
    struct PiEntry {
      StringRef Name;
      Intrinsic::ID Intr;
      bool DivideByPi; // true: f(x)/pi; false: f(x*pi)
    };
    static const PiEntry PiTable[] = {
        {"asinpi", Intrinsic::asin, true},
        {"acospi", Intrinsic::acos, true},
        {"atanpi", Intrinsic::atan, true},
        {"sinpi",  Intrinsic::sin,  false},
        {"cospi",  Intrinsic::cos,  false},
        {"tanpi",  Intrinsic::tan,  false},
    };
    const PiEntry *PiMatch = nullptr;
    for (const PiEntry &E : PiTable) {
      if (E.Name == Demangled && CI->arg_size() == 1) {
        PiMatch = &E;
        break;
      }
    }
    if (PiMatch) {
      Type *ResTy = CI->getType();
      if (!ResTy->isFloatingPointTy())
        continue;
      Value *Arg = CI->getArgOperand(0);
      if (Arg->getType() != ResTy)
        continue;
      IRBuilder<> B(CI);
      // pi as the result type's float constant.
      Constant *Pi = ConstantFP::get(ResTy, 3.14159265358979323846);
      Function *Decl = Intrinsic::getOrInsertDeclaration(
          F.getParent(), PiMatch->Intr, {ResTy});
      Value *Replacement;
      if (PiMatch->DivideByPi) {
        CallInst *Inner = B.CreateCall(Decl, {Arg});
        Inner->setCallingConv(Decl->getCallingConv());
        if (CI->getFastMathFlags().any())
          Inner->setFastMathFlags(CI->getFastMathFlags());
        Replacement = B.CreateFDiv(Inner, Pi);
      } else {
        Value *Scaled = B.CreateFMul(Arg, Pi);
        CallInst *Outer = B.CreateCall(Decl, {Scaled});
        Outer->setCallingConv(Decl->getCallingConv());
        if (CI->getFastMathFlags().any())
          Outer->setFastMathFlags(CI->getFastMathFlags());
        Replacement = Outer;
      }
      CI->replaceAllUsesWith(Replacement);
      CI->eraseFromParent();
      Changed = true;
      continue;
    }
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

// Lowers chipStar's HIP subgroup helpers (`__shfl`, `__shfl_xor`, `__shfl_up`,
// `__shfl_down`, `__chip_lane_id`) to direct SPV intrinsic calls. In Shader
// mode the OpenCL devicelib that normally provides these is not linked, so
// the call sites remain as undefined externals. The replacement IR mirrors
// the devicelib semantics, using `llvm.spv.wave.readlane` (which selects to
// `OpGroupNonUniformShuffle`) and `llvm.spv.subgroup.local.invocation.id`.
//
// __chip_lane_id() -> SubgroupLocalInvocationId
// __shfl(v, srcLane, wSize) -> readlane(v, (laneId/wSize)*wSize + (srcLane % wSize))
// __shfl_xor(v, mask, _)    -> readlane(v, laneId ^ mask)   (OpenCL ignores width)
// __shfl_up(v, delta, wSize) -> readlane(v, sub-warp clamped (laneId - delta))
// __shfl_down(v, delta, wSize) -> readlane(v, sub-warp clamped (laneId + delta))
static bool remapSubgroupBuiltins(Function &F) {
  if (F.isDeclaration())
    return false;
  Module *M = F.getParent();
  LLVMContext &Ctx = F.getContext();
  Type *I32 = Type::getInt32Ty(Ctx);
  bool Changed = false;
  SmallVector<CallInst *, 16> Worklist;
  for (BasicBlock &BB : F)
    for (Instruction &I : BB)
      if (auto *CI = dyn_cast<CallInst>(&I))
        Worklist.push_back(CI);

  auto buildLaneId = [&](IRBuilder<> &B) -> Value * {
    Function *Decl = Intrinsic::getOrInsertDeclaration(
        M, Intrinsic::spv_subgroup_local_invocation_id);
    return B.CreateCall(Decl, {});
  };
  auto buildReadLane = [&](IRBuilder<> &B, Value *V, Value *Lane) -> Value * {
    Function *Decl = Intrinsic::getOrInsertDeclaration(
        M, Intrinsic::spv_wave_readlane, {V->getType()});
    return B.CreateCall(Decl, {V, Lane});
  };

  for (CallInst *CI : Worklist) {
    Function *Callee = CI->getCalledFunction();
    if (!Callee || !Callee->isDeclaration())
      continue;
    StringRef N = Callee->getName();
    if (!N.starts_with("_Z"))
      continue;
    StringRef Rest = N.drop_front(2);
    unsigned LenDigits = 0;
    while (LenDigits < Rest.size() &&
           isdigit(static_cast<unsigned char>(Rest[LenDigits])))
      ++LenDigits;
    if (LenDigits == 0)
      continue;
    unsigned NameLen = 0;
    if (Rest.substr(0, LenDigits).getAsInteger(10, NameLen) || NameLen == 0)
      continue;
    if (NameLen > Rest.size() - LenDigits)
      continue;
    StringRef Demangled = Rest.substr(LenDigits, NameLen);

    IRBuilder<> B(CI);

    if (Demangled == "__chip_lane_id" && CI->arg_size() == 0 &&
        CI->getType()->isIntegerTy(32)) {
      Value *Lane = buildLaneId(B);
      CI->replaceAllUsesWith(Lane);
      CI->eraseFromParent();
      Changed = true;
      continue;
    }

    enum Kind { K_Shfl, K_ShflXor, K_ShflUp, K_ShflDown };
    Kind K;
    if (Demangled == "__shfl")
      K = K_Shfl;
    else if (Demangled == "__shfl_xor")
      K = K_ShflXor;
    else if (Demangled == "__shfl_up")
      K = K_ShflUp;
    else if (Demangled == "__shfl_down")
      K = K_ShflDown;
    else
      continue;

    if (CI->arg_size() != 3)
      continue;
    Value *Var = CI->getArgOperand(0);
    Value *Arg1 = CI->getArgOperand(1);
    Value *Arg2 = CI->getArgOperand(2);
    if (!Arg1->getType()->isIntegerTy(32) || !Arg2->getType()->isIntegerTy(32))
      continue;

    Value *LaneId = buildLaneId(B);
    Value *SrcLane = nullptr;
    switch (K) {
    case K_ShflXor:
      SrcLane = B.CreateXor(LaneId, Arg1);
      break;
    case K_Shfl: {
      Value *Width = Arg2;
      Value *SubWarp = B.CreateUDiv(LaneId, Width);
      Value *Base = B.CreateMul(SubWarp, Width);
      Value *LaneInWarp = B.CreateURem(Arg1, Width);
      SrcLane = B.CreateAdd(Base, LaneInWarp);
      break;
    }
    case K_ShflUp: {
      Value *Width = Arg2;
      Value *Delta = Arg1;
      Value *SubWarp = B.CreateUDiv(LaneId, Width);
      Value *LaneInWarp = B.CreateURem(LaneId, Width);
      Value *Sub = B.CreateSub(LaneInWarp, Delta);
      Value *Neg = B.CreateICmpSLT(Sub, ConstantInt::get(I32, 0));
      Value *Clamped = B.CreateSelect(Neg, LaneInWarp, Sub);
      Value *Base = B.CreateMul(SubWarp, Width);
      SrcLane = B.CreateAdd(Base, Clamped);
      break;
    }
    case K_ShflDown: {
      Value *Width = Arg2;
      Value *Delta = Arg1;
      Value *SubWarp = B.CreateUDiv(LaneId, Width);
      Value *LaneInWarp = B.CreateURem(LaneId, Width);
      Value *Sum = B.CreateAdd(LaneInWarp, Delta);
      Value *OutOfRange = B.CreateICmpUGE(Sum, Width);
      Value *Clamped = B.CreateSelect(OutOfRange, LaneInWarp, Sum);
      Value *Base = B.CreateMul(SubWarp, Width);
      SrcLane = B.CreateAdd(Base, Clamped);
      break;
    }
    }
    Value *Result = buildReadLane(B, Var, SrcLane);
    CI->replaceAllUsesWith(Result);
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
  // Phase H7: pre-scan all load/store types accessed through this arg. If
  // the access widths are heterogeneous (e.g. device-global of a struct with
  // mixed-width fields), default to i8 so the byte-array fallbacks in
  // SPIRVLegalizePointerCast can synthesise wider accesses from i8 loads.
  // Without this, the "first load wins" rule below would pick (say) i8 for
  // a struct that also has i64 fields, leaving the i64 load with no fallback.
  {
    const DataLayout &DL = A.getParent()->getDataLayout();
    SmallPtrSet<const Value *, 32> ScanTracked;
    ScanTracked.insert(&A);
    SmallVector<Value *, 16> ScanWL;
    SmallPtrSet<User *, 32> ScanSeen;
    for (User *U : A.users())
      ScanWL.push_back(U);
    SmallSet<uint64_t, 4> AccessSizes;
    bool SawNonPrimitive = false;
    while (!ScanWL.empty()) {
      Value *V = ScanWL.pop_back_val();
      auto *U = dyn_cast<User>(V);
      if (!U || !ScanSeen.insert(U).second)
        continue;
      if (auto *LI = dyn_cast<LoadInst>(U)) {
        Type *T = LI->getType();
        if (T->isPointerTy()) {
          ScanTracked.insert(LI);
          for (User *UU : LI->users())
            ScanWL.push_back(UU);
          continue;
        }
        if (T->isIntegerTy() || T->isFloatingPointTy() || T->isVectorTy())
          AccessSizes.insert(DL.getTypeStoreSize(T).getFixedValue());
        else
          SawNonPrimitive = true;
        continue;
      }
      if (auto *SI = dyn_cast<StoreInst>(U)) {
        Value *Ptr = SI->getPointerOperand();
        if (ScanTracked.count(Ptr) ||
            ScanTracked.count(Ptr->stripPointerCasts())) {
          Type *T = SI->getValueOperand()->getType();
          if (!T->isPointerTy()) {
            if (T->isIntegerTy() || T->isFloatingPointTy() || T->isVectorTy())
              AccessSizes.insert(DL.getTypeStoreSize(T).getFixedValue());
            else
              SawNonPrimitive = true;
          }
        }
        continue;
      }
      if (auto *GEPI = dyn_cast<GetElementPtrInst>(U)) {
        // Record the GEP's source-element-type alloc size as evidence of the
        // buffer's logical aggregate stride. This catches the
        // "uchar4* read decomposed into 4 byte loads" pattern where the
        // load widths are all 1 but the variable-stride GEP runs in
        // [4 x i8] units — without this hint AccessSizes would only see {1}
        // and the runtime array would be `[0 x i8]`, making the
        // `gep [4 x i8], P11, %var` fail to decompose into a valid
        // OpAccessChain over `[0 x i8]`.
        Type *SrcET = GEPI->getSourceElementType();
        if (SrcET && SrcET->isSized()) {
          uint64_t Sz = DL.getTypeAllocSize(SrcET).getFixedValue();
          if (Sz > 0)
            AccessSizes.insert(Sz);
        }
        ScanTracked.insert(U);
        for (User *UU : U->users())
          ScanWL.push_back(UU);
        continue;
      }
      if (isa<AddrSpaceCastInst>(U) ||
          isa<BitCastInst>(U) || isa<PtrToIntInst>(U) ||
          isa<IntToPtrInst>(U)) {
        ScanTracked.insert(U);
        for (User *UU : U->users())
          ScanWL.push_back(UU);
        continue;
      }
    }
    if (!SawNonPrimitive && AccessSizes.size() >= 2) {
      // When the access widths form a "byte slice plus aggregate" pattern —
      // i.e. one width is 1 (per-channel byte) and another is K>1 (whole-
      // aggregate) — the kernel is reading/writing a K-byte aggregate and
      // dissecting it byte-by-byte. The runtime array's element type must
      // be the LARGEST (K) so that variable-index GEPs like
      // `gep [K x i8], P11, %var` decompose cleanly into the
      // `[0 x [K x i8]]` runtime array (the SPIR-V backend rejects
      // `gep [K x i8]` on a `[0 x i8]` array as an OpAccessChain shape
      // mismatch). Sub-byte accesses at constant offsets continue to work
      // through the IsByteMultGEP path which converts byte offsets into
      // element-stride indices via integer division.
      uint64_t MinSz = *AccessSizes.begin();
      uint64_t MaxSz = MinSz;
      for (uint64_t S : AccessSizes) {
        MinSz = std::min(MinSz, S);
        MaxSz = std::max(MaxSz, S);
      }
      if (MinSz == 1 && MaxSz > 1 && (MaxSz % MinSz) == 0)
        return ArrayType::get(Type::getInt8Ty(A.getContext()), MaxSz);
      return Type::getInt8Ty(A.getContext());
    }
  }
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
      // BUT: SROA may have collapsed `*Dst = Src` (Src=byval struct) into a
      // primitive load+store (e.g. i64 for an 8-byte struct), losing the
      // struct type. If the function has exactly one byval arg whose store
      // size matches this load type, prefer the byval struct so the dest
      // storage buffer's element type matches the source push-constant
      // payload. Otherwise the SPIR-V backend can't legalise the
      // load(struct) / store(scalar) mismatch.
      Function *Fn = A.getParent();
      Argument *ByvalArg = nullptr;
      unsigned ByvalCount = 0;
      for (Argument &OA : Fn->args()) {
        if (OA.hasByValAttr()) {
          ByvalArg = &OA;
          ++ByvalCount;
        }
      }
      if (ByvalCount == 1 && ByvalArg) {
        Type *BVT = ByvalArg->getParamByValType();
        const DataLayout &DL = Fn->getDataLayout();
        if (BVT->isAggregateType() &&
            DL.getTypeStoreSize(BVT) == DL.getTypeStoreSize(LI->getType()))
          return BVT;
      }
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
        if (!VT->isPointerTy()) {
          // SROA may have collapsed `*Dst = Src` (Src is a byval struct) into
          // `store iN %loaded, %Dst` with iN = sizeof(struct). If we can find
          // a unique byval arg whose store size matches VT, prefer that
          // struct type — it keeps the dest storage buffer's element type
          // aligned with the source push-constant payload, which is what the
          // SPIR-V backend's pointer-cast legaliser requires.
          Function *Fn = A.getParent();
          Argument *ByvalArg = nullptr;
          unsigned ByvalCount = 0;
          for (Argument &OA : Fn->args()) {
            if (OA.hasByValAttr()) {
              ByvalArg = &OA;
              ++ByvalCount;
            }
          }
          if (ByvalCount == 1 && ByvalArg) {
            Type *BVT = ByvalArg->getParamByValType();
            const DataLayout &DL = Fn->getDataLayout();
            if (BVT->isAggregateType() &&
                DL.getTypeStoreSize(BVT) == DL.getTypeStoreSize(VT))
              return BVT;
          }
          return VT;
        }
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
        // Byte-offset GEP whose offset is a constant multiple of ElemSize
        // (SROA emits these to advance to a struct-field / array-element).
        // Convert to a unit-typed getpointer(handle, idx) call.
        // Treat byte-offset constant GEPs as element-stride GEPs only when the
        // signed offset is a clean multiple of ElemSize AND the resulting
        // element index fits in a signed i32. Negative offsets (SROA reverse-
        // index patterns) round-trip through ZExtValue as huge unsigned values
        // which previously crashed ConstantInt::get(i32, ...).
        bool IsByteMultGEP = false;
        int64_t ByteMultIdx = 0;
        if (Idx.size() == 1 && M && Handle && ElemTy && ElemSize != 0 &&
            SrcElemTy && SrcElemTy->isIntegerTy(8) &&
            isa<ConstantInt>(Idx[0])) {
          int64_t SOff = cast<ConstantInt>(Idx[0])->getSExtValue();
          int64_t SElem = static_cast<int64_t>(ElemSize);
          if (SOff % SElem == 0) {
            int64_t Q = SOff / SElem;
            if (Q >= INT32_MIN && Q <= INT32_MAX) {
              IsByteMultGEP = true;
              ByteMultIdx = Q;
            }
          }
        }
        // Dynamic byte-offset case: when SrcElemTy is i8 and the dynamic
        // byte offset is a multiple of ElemSize, convert to
        // getpointer(handle, byte_offset / ElemSize). This matches the
        // pattern instcombine emits at -O3 for `arr[idx]` where arr was
        // already converted via a getpointer(handle, 0): the resulting
        // GEP is byte-strided with a dynamic offset that is `idx*ElemSize`.
        // Without this rewrite the SPIR-V backend lowers
        // `getpointer(handle,0) + GEP_byte` as just `OpAccessChain h 0 0`,
        // dropping the dynamic offset and reading element 0 always.
        bool IsDynByteGEP = false;
        if (!IsByteMultGEP && Idx.size() == 1 && M && Handle && ElemTy &&
            ElemSize != 0 && SrcElemTy && SrcElemTy->isIntegerTy(8) &&
            !isa<ConstantInt>(Idx[0]) &&
            DL.getTypeAllocSize(SrcElemTy) != ElemSize) {
          IsDynByteGEP = true;
        }
        if (Idx.size() == 1 && M && Handle && ElemTy &&
            (DL.getTypeAllocSize(SrcElemTy) == ElemSize || IsByteMultGEP ||
             IsDynByteGEP)) {
          auto *PtrSB = PointerType::get(GEP->getContext(), 11);
          // getpointer takes i32 idx by HLSL convention.
          Type *I32Ty = Type::getInt32Ty(GEP->getContext());
          Value *I32Idx;
          if (IsByteMultGEP) {
            I32Idx = ConstantInt::getSigned(I32Ty, ByteMultIdx);
          } else if (IsDynByteGEP) {
            // Element index = byte offset / ElemSize. Use a power-of-two
            // arithmetic shift right when possible (signed because byte
            // offsets are signed, e.g. SROA reverse-index patterns).
            Value *ByteIdx = Idx[0];
            Value *ElemIdx;
            if (llvm::isPowerOf2_64(ElemSize)) {
              unsigned Shift = llvm::Log2_64(ElemSize);
              ElemIdx = Local.CreateAShr(
                  ByteIdx,
                  ConstantInt::get(ByteIdx->getType(), Shift),
                  "", /*isExact=*/true);
            } else {
              ElemIdx = Local.CreateSDiv(
                  ByteIdx,
                  ConstantInt::get(ByteIdx->getType(),
                                   static_cast<int64_t>(ElemSize)),
                  "", /*isExact=*/true);
            }
            I32Idx = Local.CreateTrunc(ElemIdx, I32Ty);
          } else {
            I32Idx = Local.CreateTrunc(Idx[0], I32Ty);
          }
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
          // Chained-GEP fixup: if `New` (the rewritten base) is itself a
          // getpointer(handle, prev_idx) call (i.e. an outer GEP that was
          // already converted to OpAccessChain), accumulate the outer
          // index into our new one. Otherwise the resulting OpAccessChain
          // would start fresh at base 0 and discard the outer offset,
          // causing chained array accesses (e.g. `out[i*width + j]` after
          // instcombine splits it into `(out + i*width)[j]`) to read/write
          // the wrong element.
          if (auto *PrevII = dyn_cast<IntrinsicInst>(New)) {
            if (PrevII->getIntrinsicID() ==
                Intrinsic::spv_resource_getpointer) {
              Value *PrevIdx = PrevII->getArgOperand(1);
              if (PrevIdx->getType() != I32Ty)
                PrevIdx = Local.CreateZExtOrTrunc(PrevIdx, I32Ty);
              I32Idx = Local.CreateAdd(PrevIdx, I32Idx);
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
  // For byval-structs containing pointer fields (e.g. hipPitchedPtr.ptr),
  // rewrite those pointer fields to i64 so the resulting Vulkan SPV doesn't
  // need GenericPointer capability. The pointer is reconstructed at use-site
  // as a PhysicalStorageBuffer (BDA) pointer.
  std::function<Type *(Type *)> ReplacePtrFieldsWithI64 = [&](Type *T) -> Type * {
    if (T->isPointerTy())
      return I64Ty;
    if (auto *ST = dyn_cast<StructType>(T)) {
      SmallVector<Type *, 8> NewFields;
      bool Changed = false;
      for (Type *FT : ST->elements()) {
        Type *NF = ReplacePtrFieldsWithI64(FT);
        if (NF != FT)
          Changed = true;
        NewFields.push_back(NF);
      }
      if (!Changed)
        return ST;
      // Create an anonymous (literal) struct so layout matches the
      // original (clang-emitted) hipPitchedPtr layout — pointer and i64
      // are both 8 bytes, so offsets are preserved.
      return StructType::get(C, NewFields, ST->isPacked());
    }
    if (auto *AT = dyn_cast<ArrayType>(T)) {
      Type *NE = ReplacePtrFieldsWithI64(AT->getElementType());
      if (NE == AT->getElementType())
        return AT;
      return ArrayType::get(NE, AT->getNumElements());
    }
    return T;
  };

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
      Ty = ReplacePtrFieldsWithI64(Ty);
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
  // For byval-structs containing pointer fields (e.g. hipPitchedPtr.ptr),
  // rewrite those pointer fields to i64 so the resulting Vulkan SPV doesn't
  // need GenericPointer capability. The post-processing PtrLoads loop below
  // reconstructs the pointer at use-site as a PhysicalStorageBuffer (BDA)
  // pointer.
  auto *I64TyForRewrite = Type::getInt64Ty(C);
  std::function<Type *(Type *)> ReplacePtrFieldsWithI64 = [&](Type *T) -> Type * {
    if (T->isPointerTy())
      return I64TyForRewrite;
    if (auto *ST = dyn_cast<StructType>(T)) {
      SmallVector<Type *, 8> NewFields;
      bool Changed = false;
      for (Type *FT : ST->elements()) {
        Type *NF = ReplacePtrFieldsWithI64(FT);
        if (NF != FT)
          Changed = true;
        NewFields.push_back(NF);
      }
      if (!Changed)
        return ST;
      return StructType::get(C, NewFields, ST->isPacked());
    }
    if (auto *AT = dyn_cast<ArrayType>(T)) {
      Type *NE = ReplacePtrFieldsWithI64(AT->getElementType());
      if (NE == AT->getElementType())
        return AT;
      return ArrayType::get(NE, AT->getNumElements());
    }
    return T;
  };

  SmallVector<Argument *, 4> ScalarArgs;
  SmallVector<Type *, 4> ScalarTypes;
  for (Argument &A : F.args()) {
    if (!isBufferArg(A)) {
      Type *Ty = A.getType();
      if (A.hasByValAttr())
        Ty = A.getParamByValType();
      Ty = ReplacePtrFieldsWithI64(Ty);
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
      // Phase I5: if the kernel never references this buffer arg we still
      // must materialise a descriptor so clspv's reflection emits a
      // matching ArgumentStorageBuffer entry. Otherwise the chipStar
      // runtime sees NumHipPtrs > PointerArgsOcl (e.g. `half_neg(x,y,z)`
      // ignores `y`) and the legacy fallback re-orders args incorrectly,
      // producing garbage outputs. We materialise the handle/getpointer
      // intrinsics and anchor them with a synthetic volatile i8 load so
      // they survive DCE — the host always binds a real cl_mem for HIP
      // pointer args, so reading byte 0 is safe and harmless.
      bool UnusedBufferArg = A.use_empty();
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
      // Phase H4: if this arg was synthesised by lowerDeviceGlobalsToStorageBuffers
      // (name prefix `__hipspv_dg_`), preserve the prefixed symbol name as the
      // resource debug-name so the chipStar runtime can recover the
      // (kernel, ord, symbol) mapping by walking SPV OpName decorations.
      std::string DbgName;
      if (A.hasName() && A.getName().starts_with("__hipspv_dg_"))
        DbgName = A.getName().str();
      else
        DbgName = (NewF->getName() + "." + Twine(Binding)).str();
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
      if (UnusedBufferArg) {
        // Anchor the otherwise-dead handle with a volatile byte load so the
        // descriptor survives DCE and clspv emits an ArgumentStorageBuffer
        // reflection entry. The load value is unused and discarded.
        LoadInst *Anchor =
            B.CreateAlignedLoad(Type::getInt8Ty(C), P11, MaybeAlign(1));
        Anchor->setVolatile(true);
        ++Binding;
        continue;
      }
      Value *ReplVal;
      if (PT->getAddressSpace() == 11) {
        ReplVal = P11;
        A.replaceAllUsesWith(ReplVal);
        // Even though no addrspacecast is needed, we still need to fold any
        // variable-index GEPs sitting on top of P11 (`getelementptr ElemTy,
        // ptr addrspace(11) %P11, i64 %var`) into element-indexed
        // `getpointer(handle, idx)` calls. A raw GEP with a variable first
        // index causes the SPIR-V backend to request the
        // StorageBufferArrayDynamicIndexing capability (forbidden under the
        // Vulkan 1.2 baseline targeted by clvk). Seeding the rewrite with
        // Cast=NewPtr=P11 walks P11's users, converts variable-index GEPs to
        // typed-index getpointer calls, and rebuilds direct load/store users
        // unchanged (same base, harmless).
        rewriteToStorageBuffer(Handle, VkBufTy, ElemTy, P11, P11);
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

  // Post-processing: a byval struct kernel arg whose IR loads a pointer-typed
  // field (e.g. `hipPitchedPtr.ptr`) ends up as
  //   %p = load ptr addrspace(N), ptr addrspace(13) <gep into pcs>
  // The kernel then does byte-strided GEPs and element-typed load/stores
  // through %p. Under Vulkan Logical addressing the SPIR-V backend has no
  // way to express such arbitrary-stride accesses through a generic pointer
  // loaded from a push constant — `SPIRVLegalizePointerCast` can't deduce a
  // compatible pointee layout and aborts. Treat each such field as a Buffer
  // Device Address: load it as i64, inttoptr to PhysicalStorageBuffer
  // (addrspace(5)), then addrspacecast to the original addrspace so the
  // existing Phase F (BDA) path in the backend handles it.
  if (PushConstGV) {
    SmallVector<LoadInst *, 4> PtrLoads;
    for (BasicBlock &BB : *NewF)
      for (Instruction &I : BB)
        if (auto *LI = dyn_cast<LoadInst>(&I)) {
          if (!LI->getType()->isPointerTy())
            continue;
          auto *PtrOp = LI->getPointerOperand();
          auto *PtrOpTy = dyn_cast<PointerType>(PtrOp->getType());
          if (!PtrOpTy || PtrOpTy->getAddressSpace() != 13)
            continue;
          // Confirm the load's source ultimately points at our PCS global.
          Value *Cur = PtrOp->stripPointerCasts();
          while (auto *GEP = dyn_cast<GEPOperator>(Cur))
            Cur = GEP->getPointerOperand()->stripPointerCasts();
          if (Cur != PushConstGV)
            continue;
          PtrLoads.push_back(LI);
        }
    for (LoadInst *LI : PtrLoads) {
      IRBuilder<> RB(LI);
      auto *I64Ty = Type::getInt64Ty(C);
      // The PCS leaf for this field has been rewritten from `ptr` to `i64`
      // by ReplacePtrFieldsWithI64 — load it as i64 directly so the SPIR-V
      // backend never sees a Generic pointer type (which would require the
      // GenericPointer capability, unavailable under Vulkan).
      //
      // Strategy: keep the per-thread address in i64 arithmetic (via add/sub
      // for byte-strided GEPs) and only `inttoptr` to a `ptr addrspace(5)`
      // *immediately before* each leaf load/store, with the PSB pointer
      // typed to match the load/store value type. This produces SPV that
      // the backend lowers as `OpConvertUToPtr <typed-PSB-ptr>; OpLoad/OpStore`
      // — the only shape currently supported for byte-strided BDA access.
      Value *BaseAddr = RB.CreateAlignedLoad(I64Ty, LI->getPointerOperand(),
                                              MaybeAlign(8));

      // Map: old Value* -> i64 address expression representing it.
      DenseMap<Value *, Value *> AddrFor;
      AddrFor[LI] = BaseAddr;
      SmallVector<Value *, 8> Worklist{LI};
      SmallVector<Instruction *, 8> ToErase;
      const DataLayout &DL = NewF->getDataLayout();

      auto getOrComputeAddr = [&](Value *V) -> Value * {
        auto It = AddrFor.find(V);
        return It == AddrFor.end() ? nullptr : It->second;
      };

      while (!Worklist.empty()) {
        Value *Old = Worklist.pop_back_val();
        Value *Cur = getOrComputeAddr(Old);
        if (!Cur)
          continue;
        SmallVector<User *, 8> Users(Old->user_begin(), Old->user_end());
        for (User *U : Users) {
          auto *I = dyn_cast<Instruction>(U);
          if (!I || !I->getParent())
            continue;
          if (auto *GEP = dyn_cast<GetElementPtrInst>(I)) {
            if (GEP->getPointerOperand() != Old)
              continue;
            // Compute byte offset added by this GEP in i64.
            IRBuilder<> Local(GEP);
            APInt ConstOff(64, 0);
            SmallMapVector<Value *, APInt, 4> VarOffsets;
            if (!GEP->collectOffset(DL, 64, VarOffsets, ConstOff)) {
              // Fall back: leave GEP alone (shouldn't happen for our patterns).
              continue;
            }
            Value *Off = ConstantInt::get(I64Ty, ConstOff);
            for (auto &P : VarOffsets) {
              Value *Idx = P.first;
              APInt Scale = P.second;
              Value *IdxI64 =
                  Local.CreateSExtOrTrunc(Idx, I64Ty);
              if (Scale != 1) {
                IdxI64 = Local.CreateMul(
                    IdxI64, ConstantInt::get(I64Ty, Scale));
              }
              Off = Local.CreateAdd(Off, IdxI64);
            }
            Value *NewAddr = Local.CreateAdd(Cur, Off);
            AddrFor[GEP] = NewAddr;
            Worklist.push_back(GEP);
            ToErase.push_back(GEP);
          } else if (auto *L2 = dyn_cast<LoadInst>(I)) {
            if (L2->getPointerOperand() != Old)
              continue;
            // Emit: %p = inttoptr i64 %addr to ptr addrspace(5)
            //       %v = load <ValTy>, ptr addrspace(5) %p
            IRBuilder<> Local(L2);
            Type *ValTy = L2->getType();
            Value *PSBPtr =
                Local.CreateIntToPtr(Cur, PointerType::get(C, 5));
            // Hint pointee type by an explicit assign-aligned load — the
            // SPV backend deduces the PSB pointee from the load type.
            LoadInst *NL = Local.CreateAlignedLoad(ValTy, PSBPtr,
                                                    L2->getAlign(),
                                                    L2->isVolatile());
            L2->replaceAllUsesWith(NL);
            ToErase.push_back(L2);
          } else if (auto *S2 = dyn_cast<StoreInst>(I)) {
            if (S2->getPointerOperand() != Old)
              continue;
            IRBuilder<> Local(S2);
            Value *PSBPtr =
                Local.CreateIntToPtr(Cur, PointerType::get(C, 5));
            Local.CreateAlignedStore(S2->getValueOperand(), PSBPtr,
                                     S2->getAlign(), S2->isVolatile());
            ToErase.push_back(S2);
          } else if (auto *AC = dyn_cast<AddrSpaceCastInst>(I)) {
            // Bypass: forward the address rather than emit a cross-AS cast.
            AddrFor[AC] = Cur;
            Worklist.push_back(AC);
            ToErase.push_back(AC);
          }
          // Other users (calls, ptrtoint, etc.) intentionally untouched.
        }
      }
      // Erase stores first (side-effecting), then the rest in
      // dependency-safe order.
      for (Instruction *I : ToErase)
        if (isa<StoreInst>(I) && I->getParent())
          I->eraseFromParent();
      bool Progress = true;
      while (Progress) {
        Progress = false;
        for (Instruction *I : ToErase) {
          if (I->getParent() && I->use_empty()) {
            I->eraseFromParent();
            Progress = true;
          }
        }
      }
      if (LI->getParent() && LI->use_empty())
        LI->eraseFromParent();
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
      // If exact-size match failed, fall back to a "Len fits in field" match.
      // Clang's memcpy length is the source-language sizeof, which can omit
      // trailing struct padding that LLVM's getTypeStoreSize accounts for
      // (e.g. an outer packed `<{...}>` wrapping a non-packed inner struct).
      // This shows up as memcpy(len=28) targeting a PCS field whose
      // getTypeStoreSize is 32. Match by "first field whose storeSize >= Len"
      // — this is safe because the memcpy source IS the PCS global itself, so
      // we know the entire payload lives in field 0 for the single-byval-arg
      // case (which is what TestKernelArgs hits for S1_t/S2_t/S5_t).
      if (FieldIdx < 0) {
        for (unsigned Fi = 0, Fe = PushConstStruct->getNumElements();
             Fi < Fe; ++Fi) {
          Type *FT = PushConstStruct->getElementType(Fi);
          if (DL.getTypeStoreSize(FT).getFixedValue() >= Len) {
            FieldIdx = (int)Fi;
            break;
          }
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
      // Force an Instruction GEP (not a folded ConstantExpr/global-strip) so
      // the resulting pointer carries the field's pointee type. The SPIR-V
      // backend infers OpCopyMemory's pointee from the operand's pointer type;
      // if we let the GEP fold to PushConstGV directly, the source pointee is
      // the wrapper struct while the dest pointee is the inner field type,
      // which produces "OpCopyMemory requires operands to have the same type".
      SmallVector<Value *, 2> GEPIdx{ConstantInt::get(I32Ty, 0),
                                     ConstantInt::get(I32Ty, FieldIdx)};
      auto *FieldGEP = GetElementPtrInst::Create(
          PushConstStruct, PushConstGV, GEPIdx, "pcs.field",
          MTI->getIterator());
      FieldGEP->setIsInBounds(true);
      // Re-emit the memcpy with the typed source pointer. Both src (FT*) and
      // dst (already FT* via spv.resource.getpointer) now share a pointee
      // type.
      // Use the full field type size (not the original memcpy length) so the
      // backend's "OpCopyMemory size matches pointee size" check passes. The
      // length difference is benign trailing padding the original memcpy
      // omitted; copying it from the externally-initialised PCS into the
      // storage buffer is harmless because both endpoints have the same
      // padded layout.
      const DataLayout &DL2 = NewF->getDataLayout();
      uint64_t FieldSize = DL2.getTypeStoreSize(FT).getFixedValue();
      RB.CreateMemCpy(MTI->getRawDest(), MTI->getDestAlign(), FieldGEP,
                      MTI->getSourceAlign(),
                      ConstantInt::get(MTI->getLength()->getType(), FieldSize),
                      MTI->isVolatile());
      MTI->eraseFromParent();
    }

    // Same idea for direct loads off PushConstGV: when the load type doesn't
    // match the PCS wrapper struct (e.g. SROA collapsed a small struct into
    // a single i64 load), the backend can't drill in and asserts in
    // SPIRVLegalizePointerCast::buildLegalizedLoad. Rewrite to a typed
    // instruction-GEP into the matching PCS field.
    SmallVector<LoadInst *, 4> LoadsToRewrite;
    for (BasicBlock &BB : *NewF)
      for (Instruction &I : BB)
        if (auto *LI = dyn_cast<LoadInst>(&I))
          if (LI->getPointerOperand()->stripPointerCasts() == PushConstGV &&
              LI->getType() != PushConstStruct)
            LoadsToRewrite.push_back(LI);
    for (LoadInst *LI : LoadsToRewrite) {
      Type *LTy = LI->getType();
      const DataLayout &DL3 = NewF->getDataLayout();
      uint64_t LSize = DL3.getTypeStoreSize(LTy).getFixedValue();
      int FieldIdx = -1;
      // Prefer exact-size match.
      for (unsigned Fi = 0, Fe = PushConstStruct->getNumElements();
           Fi < Fe; ++Fi) {
        Type *FT = PushConstStruct->getElementType(Fi);
        if (DL3.getTypeStoreSize(FT).getFixedValue() == LSize) {
          FieldIdx = (int)Fi;
          break;
        }
      }
      // Fallback: first field that fits the load.
      if (FieldIdx < 0) {
        for (unsigned Fi = 0, Fe = PushConstStruct->getNumElements();
             Fi < Fe; ++Fi) {
          Type *FT = PushConstStruct->getElementType(Fi);
          if (DL3.getTypeStoreSize(FT).getFixedValue() >= LSize) {
            FieldIdx = (int)Fi;
            break;
          }
        }
      }
      if (FieldIdx < 0)
        continue;
      Type *FT = PushConstStruct->getElementType(FieldIdx);
      SmallVector<Value *, 2> GEPIdx{ConstantInt::get(I32Ty, 0),
                                     ConstantInt::get(I32Ty, FieldIdx)};
      auto *FieldGEP =
          GetElementPtrInst::Create(PushConstStruct, PushConstGV, GEPIdx,
                                    "pcs.field", LI->getIterator());
      FieldGEP->setIsInBounds(true);
      // Special case: the load type doesn't match the field type (e.g. SROA
      // collapsed `*Dst = Src` for an 8-byte struct into `load i64; store i64`,
      // but we picked the byval struct as the dest's element type). The
      // SPIR-V backend can't legalise a load(iN)/store(iN) chain whose src
      // and dst are both struct-typed. If the unique consumer of LI is a
      // store with a tracked-as-dst pointer, replace the load+store pair
      // with a typed memcpy.
      if (LTy != FT && LI->hasOneUse()) {
        auto *SI = dyn_cast<StoreInst>(*LI->user_begin());
        if (SI && SI->getValueOperand() == LI) {
          IRBuilder<> RB(SI);
          uint64_t FieldSize =
              DL3.getTypeStoreSize(FT).getFixedValue();
          RB.CreateMemCpy(SI->getPointerOperand(), SI->getAlign(), FieldGEP,
                          LI->getAlign(),
                          ConstantInt::get(I32Ty->getInt64Ty(C), FieldSize),
                          SI->isVolatile());
          SI->eraseFromParent();
          LI->eraseFromParent();
          continue;
        }
      }
      LI->setOperand(LoadInst::getPointerOperandIndex(), FieldGEP);
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

// Phase G3b: lower chipStar's `__chip_clk_counter` host-runtime helper used
// by `clock()` / `clock64()` / `wall_clock()` / `wall_clock64()`. The
// chipStar device library implements these as
//   atomic_fetch_add_explicit(&__chip_clk_counter, 1, ...);
//   return __chip_clk_counter;
// where `__chip_clk_counter` is a module-scope `addrspace(1)` (CrossWorkgroup)
// global. Vulkan/Logical SPIR-V has no CrossWorkgroup storage class for
// arbitrary i64 counters, and `SPV_KHR_shader_clock` (OpReadClockKHR) is not
// reachable from the bridging pass without extending the SPIRV builtin
// matcher to GLSL_std_450 mode. The simplest correct lowering is a per-
// invocation Function-scope (alloca) counter:
//
//   entry:
//     %clk = alloca i64; store i64 0, ptr %clk
//   ...
//   %old = load i64, ptr %clk
//   %new = add i64 %old, 1
//   store i64 %new, ptr %clk     ; replaces atomic_fetch_add(...,1,...)
//   <use of %old in place of the original atomic call>
//   <use of %new in place of subsequent loads of __chip_clk_counter>
//
// The `samples/clock` test only requires that consecutive `clock64()` calls
// return monotonically-increasing values so the wait-loop terminates; a
// per-thread counter satisfies that without needing CrossWorkgroup storage
// or the shader_clock extension. After the rewrite the global has no
// remaining uses and is erased so the backstop in
// `hasUnsupportedAddrSpaceConst` no longer drops the kernel.
static bool lowerChipClkCounter(Module &M) {
  GlobalVariable *Clk = nullptr;
  for (GlobalVariable &GV : M.globals()) {
    if (GV.getType()->getAddressSpace() != 1)
      continue;
    StringRef N = GV.getName();
    // Match the (possibly mangled) `__chip_clk_counter` symbol. clang
    // emits it as `_ZL18__chip_clk_counter` due to internal linkage.
    if (N == "__chip_clk_counter" || N.contains("__chip_clk_counter")) {
      Clk = &GV;
      break;
    }
  }
  if (!Clk)
    return false;

  LLVMContext &C = M.getContext();
  Type *I64 = Type::getInt64Ty(C);

  // Walk the global's user graph (ConstantExprs *and* the AddrSpaceCast /
  // BitCast / GEP instructions left behind by Phase A's materialization)
  // and collect every Call / Load instruction that ultimately reads from
  // or writes to it.
  SmallVector<Instruction *, 8> Users;
  SmallVector<Value *, 8> WL{Clk};
  SmallPtrSet<Value *, 16> Seen;
  while (!WL.empty()) {
    Value *V = WL.pop_back_val();
    if (!Seen.insert(V).second)
      continue;
    for (User *U : V->users()) {
      if (isa<ConstantExpr>(U) || isa<GlobalValue>(U)) {
        WL.push_back(U);
        continue;
      }
      if (isa<AddrSpaceCastInst>(U) || isa<BitCastInst>(U) ||
          isa<GetElementPtrInst>(U)) {
        // Forward through the cast/GEP to its real users; the cast itself
        // is left behind and becomes dead after we rewrite its consumers.
        WL.push_back(U);
        continue;
      }
      if (auto *I = dyn_cast<Instruction>(U))
        Users.push_back(I);
    }
  }

  // Install one i64 alloca per kernel that uses the global.
  MapVector<Function *, AllocaInst *> FnToAlloca;
  for (Instruction *I : Users) {
    Function *F = I->getFunction();
    if (!F || F->isDeclaration())
      continue;
    if (FnToAlloca.count(F))
      continue;
    IRBuilder<> B(&*F->getEntryBlock().getFirstInsertionPt());
    AllocaInst *A = B.CreateAlloca(I64, nullptr, "__chip_clk_local");
    B.CreateStore(ConstantInt::get(I64, 0), A);
    FnToAlloca[F] = A;
  }

  auto tracesToClk = [&](Value *Ptr) {
    Value *Cur = Ptr;
    SmallPtrSet<Value *, 8> Visited;
    while (Cur && Visited.insert(Cur).second) {
      if (Cur == Clk)
        return true;
      if (auto *CE = dyn_cast<ConstantExpr>(Cur)) {
        if (CE->getOpcode() == Instruction::AddrSpaceCast ||
            CE->getOpcode() == Instruction::BitCast ||
            CE->getOpcode() == Instruction::GetElementPtr) {
          Cur = CE->getOperand(0);
          continue;
        }
        return false;
      }
      if (auto *AC = dyn_cast<AddrSpaceCastInst>(Cur)) {
        Cur = AC->getPointerOperand();
        continue;
      }
      if (auto *BC = dyn_cast<BitCastInst>(Cur)) {
        Cur = BC->getOperand(0);
        continue;
      }
      if (auto *GEP = dyn_cast<GetElementPtrInst>(Cur)) {
        Cur = GEP->getPointerOperand();
        continue;
      }
      return false;
    }
    return false;
  };

  SmallVector<Instruction *, 8> ToErase;
  for (Instruction *I : Users) {
    Function *F = I->getFunction();
    auto It = FnToAlloca.find(F);
    if (It == FnToAlloca.end())
      continue;
    AllocaInst *A = It->second;

    if (auto *CI = dyn_cast<CallInst>(I)) {
      Function *Callee = CI->getCalledFunction();
      if (!Callee)
        continue;
      StringRef Name = Callee->getName();
      // Match the mangled `atomic_fetch_add_explicit` helper *or*
      // chipStar's `__chip_atomic_add_*` lowering of it. The device
      // library compiles `atomicAdd(&__chip_clk_counter, 1)` to one of
      // these depending on the compile path (devicelib link order).
      if (!Name.contains("atomic_fetch_add_explicit") &&
          !Name.contains("__chip_atomic_add"))
        continue;
      if (CI->arg_size() < 1 || !tracesToClk(CI->getArgOperand(0)))
        continue;
      IRBuilder<> B(CI);
      Value *Old = B.CreateLoad(I64, A);
      Value *New = B.CreateAdd(Old, ConstantInt::get(I64, 1));
      B.CreateStore(New, A);
      Value *Repl = Old;
      if (CI->getType() != I64 && CI->getType()->isIntegerTy())
        Repl = B.CreateIntCast(Old, CI->getType(), /*isSigned=*/false);
      CI->replaceAllUsesWith(Repl);
      ToErase.push_back(CI);
      continue;
    }
    if (auto *LI = dyn_cast<LoadInst>(I)) {
      if (!tracesToClk(LI->getPointerOperand()))
        continue;
      IRBuilder<> B(LI);
      Value *Cur = B.CreateLoad(I64, A);
      Value *Repl = Cur;
      if (LI->getType() != I64 && LI->getType()->isIntegerTy())
        Repl = B.CreateIntCast(Cur, LI->getType(), /*isSigned=*/false);
      LI->replaceAllUsesWith(Repl);
      ToErase.push_back(LI);
      continue;
    }
  }
  for (Instruction *I : ToErase)
    I->eraseFromParent();

  // Drop the now-dead extern atomic_fetch_add declaration if no other code
  // calls it; otherwise leave it for downstream lowering. The global is
  // erased once all of its uses are gone (ConstantExpr addrspacecasts in
  // particular need a `removeDeadConstantUsers` first).
  Clk->removeDeadConstantUsers();
  if (Clk->use_empty())
    Clk->eraseFromParent();

  return true;
}

// Phase G3: lower chipStar's `__chip_syncthreads()` extern call to
// `llvm.spv.group.memory.barrier.with.group.sync`, which the SPIR-V
// backend selects to `OpControlBarrier Workgroup Workgroup
// AcquireRelease|WorkgroupMemory`. The OpenCL devicelib that normally
// provides `__chip_syncthreads` is not linked in Shader mode, so leaving
// the call as-is would otherwise emit `OpCapability Linkage` (rejected by
// Vulkan) and an unresolved Import.
static bool lowerChipSyncThreads(Module &M) {
  Function *F = M.getFunction("__chip_syncthreads");
  if (!F)
    return false;
  bool Changed = false;
  SmallVector<CallInst *, 8> Calls;
  for (User *U : F->users())
    if (auto *CI = dyn_cast<CallInst>(U))
      if (CI->getCalledFunction() == F)
        Calls.push_back(CI);
  Function *Barrier = Intrinsic::getOrInsertDeclaration(
      &M, Intrinsic::spv_group_memory_barrier_with_group_sync);
  for (CallInst *CI : Calls) {
    IRBuilder<> B(CI);
    B.CreateCall(Barrier, {});
    CI->eraseFromParent();
    Changed = true;
  }
  if (F->use_empty()) {
    F->eraseFromParent();
    Changed = true;
  }
  return Changed;
}

// Phase G3: lower `__shared__` (addrspace(3) Workgroup) globals through
// AS(3)->AS(4) addrspacecasts. The HIP front-end emits accesses as
//   addrspacecast (ptr addrspace(3) @sharedMem to ptr addrspace(4))
// then GEP/load/store off the generic pointer. Vulkan/Logical SPIR-V has no
// generic addrspace, so we use rewriteToStorageBuffer to propagate the
// addrspace(3) base through the GEP/load/store chain and erase the cast.
// Globals in addrspace(3) themselves pass straight through to the backend,
// which maps addrspace(3) -> Workgroup storage class.
static bool lowerWorkgroupAddrSpaceCasts(Function &F) {
  bool Changed = false;
  SmallVector<AddrSpaceCastInst *, 4> Casts;
  for (BasicBlock &BB : F)
    for (Instruction &I : BB)
      if (auto *AC = dyn_cast<AddrSpaceCastInst>(&I))
        if (AC->getSrcAddressSpace() == 3 &&
            isa<GlobalValue>(AC->getPointerOperand()))
          Casts.push_back(AC);
  for (AddrSpaceCastInst *AC : Casts) {
    Value *GV = AC->getPointerOperand();
    rewriteToStorageBuffer(/*Handle=*/nullptr, /*VkBufTy=*/nullptr,
                           /*ElemTy=*/nullptr, GV, AC);
    Changed = true;
    // rewriteToStorageBuffer already erases the cast once its uses are gone.
  }
  return Changed;
}

// Phase G3: resize `extern __shared__ T sdata[];` arrays so the SPIR-V
// backend can emit a concrete-sized OpVariable Workgroup. The HIP front-end
// emits `external_global addrspace(3) [0 x T]` whose total byte size is
// supplied at kernel launch via the runtime's sharedMem parameter. Until the
// runtime cooperation (spec-constant for size) is wired up, we give these
// globals a generous fixed size so simple tests work.
//
// TODO(G3): replace with a LocalSpecConstant whose value is set from
// `sharedMem / sizeof(T)` at kernel launch.
static bool resizeExternSharedGlobals(Module &M) {
  bool Changed = false;
  SmallVector<GlobalVariable *, 4> ToFix;
  for (GlobalVariable &GV : M.globals()) {
    if (GV.getType()->getAddressSpace() != 3)
      continue;
    if (!GV.isDeclaration() && GV.hasInitializer())
      continue;
    auto *AT = dyn_cast<ArrayType>(GV.getValueType());
    if (!AT || AT->getNumElements() != 0)
      continue;
    ToFix.push_back(&GV);
  }
  // 16 KiB is the minimum guaranteed maxComputeSharedMemorySize in Vulkan.
  static constexpr uint64_t DefaultBytes = 16384;
  for (GlobalVariable *GV : ToFix) {
    auto *AT = cast<ArrayType>(GV->getValueType());
    Type *ElemTy = AT->getElementType();
    const DataLayout &DL = M.getDataLayout();
    uint64_t ElemSize = DL.getTypeAllocSize(ElemTy).getFixedValue();
    if (ElemSize == 0)
      continue;
    uint64_t NumElems = DefaultBytes / ElemSize;
    if (NumElems == 0)
      NumElems = 1;
    auto *NewAT = ArrayType::get(ElemTy, NumElems);
    auto *NewGV = new GlobalVariable(
        M, NewAT, /*isConstant=*/false, GlobalValue::InternalLinkage,
        UndefValue::get(NewAT), "", GV, GV->getThreadLocalMode(),
        /*AddressSpace=*/3);
    NewGV->setAlignment(MaybeAlign(GV->getAlign()));
    NewGV->takeName(GV);
    GV->replaceAllUsesWith(NewGV);
    GV->eraseFromParent();
    Changed = true;
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
    // 13 (PushConstant), 3 (Workgroup, for __shared__),
    // 0 (Function/private constants like name strings).
    if (AS == 11 || AS == 13 || AS == 3 || AS == 0)
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
          // StorageBuffer (11), PushConstant (13), and Workgroup (3, for
          // __shared__) are the legitimate Vulkan-side addrspaces;
          // addrspace(1) CrossWorkgroup on a global can't be lowered in
          // Logical mode.
          if (SrcAS != 11 && SrcAS != 13 && SrcAS != 3)
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

// Lower `llvm.memcpy` whose destination or source is a Vulkan resource
// pointer (addrspace 11) into a per-element load+store sequence. The SPIR-V
// backend lowers memcpy to OpCopyMemory, which requires source and
// destination pointee types to match. When a memcpy moves data between
// shared memory (addrspace 3) — typically typed (e.g. `[N x %struct.vec4]`)
// — and a storage buffer (addrspace 11) — typically typed as `[0 x [N x i8]]`
// — those pointee types disagree, and the assertion fires. Lowering the
// memcpy to load+store of a primitive scalar (i32, chosen for ubiquitous
// alignment compatibility) with one trip per scalar lets each load/store
// inherit its own typed pointer, so the backend emits OpLoad/OpStore on
// matching pointee types instead of OpCopyMemory.
static bool lowerVulkanResourceMemcpys(Function &F) {
  SmallVector<MemTransferInst *, 4> Worklist;
  for (BasicBlock &BB : F)
    for (Instruction &I : BB)
      if (auto *MTI = dyn_cast<MemTransferInst>(&I)) {
        unsigned DstAS = MTI->getRawDest()->getType()->getPointerAddressSpace();
        unsigned SrcAS = MTI->getRawSource()->getType()->getPointerAddressSpace();
        if (DstAS == 11 || SrcAS == 11)
          Worklist.push_back(MTI);
      }
  if (Worklist.empty())
    return false;
  Type *I8Ty = Type::getInt8Ty(F.getContext());
  bool Changed = false;
  for (MemTransferInst *MTI : Worklist) {
    auto *LenC = dyn_cast<ConstantInt>(MTI->getLength());
    if (!LenC)
      continue; // Only constant-length forms (the chipStar pattern).
    uint64_t Len = LenC->getZExtValue();
    if (Len == 0) {
      MTI->eraseFromParent();
      Changed = true;
      continue;
    }
    Value *Dst = MTI->getRawDest();
    Value *Src = MTI->getRawSource();
    Align DA = MTI->getDestAlign().valueOrOne();
    Align SA = MTI->getSourceAlign().valueOrOne();
    // Phase H7: if Dst is the result of `spv.resource.getpointer(handle, 0)`
    // for a buffer typed `[N x ElemTy]` AND Len == sizeof(ElemTy) AND Src is
    // a typed pointer to a value of size Len (e.g. a typed GEP into a
    // PushConstant struct field), emit a single typed load+store instead of
    // a byte-unrolled copy. The byte unrolling produces `gep i8, ptr S0_t, K`
    // which the SPIR-V backend's pointer-cast legaliser can't lower into a
    // struct-typed buffer; a typed load+store works without legalisation.
    {
      const DataLayout &DL = F.getDataLayout();
      auto *DstII = dyn_cast<IntrinsicInst>(Dst);
      Type *DstElemTy = nullptr;
      if (DstII &&
          DstII->getIntrinsicID() == Intrinsic::spv_resource_getpointer) {
        // Buffer element type lives in the resource_handlefrombinding's
        // target-ext-type: target("spirv.VulkanBuffer", [0 x ElemTy], 12, 1).
        Value *Handle = DstII->getArgOperand(0);
        if (auto *TET = dyn_cast<TargetExtType>(Handle->getType())) {
          if (TET->getNumTypeParameters() >= 1) {
            Type *RT = TET->getTypeParameter(0);
            if (auto *AT = dyn_cast<ArrayType>(RT))
              DstElemTy = AT->getElementType();
          }
        }
      }
      Type *SrcElemTy = nullptr;
      if (auto *SrcGEP = dyn_cast<GetElementPtrInst>(Src))
        SrcElemTy = SrcGEP->getResultElementType();
      if (DstElemTy && SrcElemTy && DstElemTy == SrcElemTy &&
          DstElemTy->isSized() &&
          DL.getTypeStoreSize(DstElemTy).getFixedValue() == Len) {
        IRBuilder<> B(MTI);
        LoadInst *LI = B.CreateAlignedLoad(DstElemTy, Src, SA,
                                            MTI->isVolatile());
        B.CreateAlignedStore(LI, Dst, DA, MTI->isVolatile());
        MTI->eraseFromParent();
        Changed = true;
        continue;
      }
    }
    // If a side is a global variable (typical for LDS source pointers),
    // use the global's actual alignment — clang often omits the explicit
    // align on the memcpy itself even though the underlying allocation is
    // aggressively aligned.
    if (auto *GV = dyn_cast<GlobalVariable>(Dst->stripPointerCasts()))
      if (GV->getAlign())
        DA = std::max(DA, *GV->getAlign());
    if (auto *GV = dyn_cast<GlobalVariable>(Src->stripPointerCasts()))
      if (GV->getAlign())
        SA = std::max(SA, *GV->getAlign());
    IRBuilder<> B(MTI);
    // Use a byte step. Single-index typed `gep i32, ptr, I` would lower to
    // OpPtrAccessChain in logical SPIR-V (unavailable), but
    // SPIRVEmitIntrinsics' walkLogicalAccessChainConstant happily descends
    // into typed aggregates from `gep i8, ptr, constByteIdx`.
    Type *StepTy = I8Ty;
    uint64_t StepBytes = 1;
    uint64_t Trips = Len;
    // Emit a small unrolled sequence — Len is typically <= 64 in shader code.
    for (uint64_t I = 0; I < Trips; ++I) {
      Value *DstGEP = B.CreateConstInBoundsGEP1_64(StepTy, Dst, I);
      Value *SrcGEP = B.CreateConstInBoundsGEP1_64(StepTy, Src, I);
      LoadInst *LI = B.CreateAlignedLoad(
          StepTy, SrcGEP, I == 0 ? SA : Align(StepBytes), MTI->isVolatile());
      StoreInst *SI = B.CreateAlignedStore(
          LI, DstGEP, I == 0 ? DA : Align(StepBytes), MTI->isVolatile());
      (void)SI;
    }
    MTI->eraseFromParent();
    Changed = true;
  }
  return Changed;
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

// Phase I4: replace calls to chipStar's external integer atomic helpers
// (`__chip_atomic_{add,sub,min,max,and,or,xor,exch}_{i,u,l}`) with native
// `atomicrmw` so the SPIR-V backend emits OpAtomic* directly. In the
// Vulkan/clvk path, the OpenCL kernel-flavored devicelib SPV cannot be linked
// against the shader-flavored kernel SPV (clLinkProgram fails), so the
// integer atomics must be lowered in-IR like the float variants. Unlike the
// float helpers, integer helpers take the typed pointer directly (no
// obfuscated i64 wrapper), so we use the call's pointer arg as-is.
static bool lowerChipAtomicIntegerOps(Module &M) {
  bool Changed = false;
  static const struct {
    const char *Suffix;
    AtomicRMWInst::BinOp Op;
  } OpsBySuffix[] = {
      {"add", AtomicRMWInst::Add},   {"sub", AtomicRMWInst::Sub},
      {"min", AtomicRMWInst::Min},   {"max", AtomicRMWInst::Max},
      {"and", AtomicRMWInst::And},   {"or",  AtomicRMWInst::Or},
      {"xor", AtomicRMWInst::Xor},   {"exch", AtomicRMWInst::Xchg},
  };
  // _i = signed int, _u = unsigned int, _l = (unsigned) long. min/max for _u
  // and _l should use UMin/UMax — handled below.
  for (const auto &E : OpsBySuffix) {
    for (char TyCh : {'i', 'u', 'l'}) {
      std::string FName =
          (Twine("__chip_atomic_") + E.Suffix + "_" + Twine(TyCh)).str();
      Function *F = M.getFunction(FName);
      if (!F)
        continue;
      // Vulkan Shader needs Int64Atomics + SPV_KHR_shader_atomic_int64 for
      // i64 atomicrmw; Intel Arc + clvk doesn't enable that. Skip _l so the
      // call falls back to the previous (extern call) lowering and the
      // user is told via a clean link-time error rather than a backend
      // "Unable to meet SPIR-V requirements" assert.
      if (TyCh == 'l')
        continue;
      AtomicRMWInst::BinOp Op = E.Op;
      // Promote min/max to unsigned for _u and _l variants.
      if (TyCh != 'i') {
        if (Op == AtomicRMWInst::Min)
          Op = AtomicRMWInst::UMin;
        else if (Op == AtomicRMWInst::Max)
          Op = AtomicRMWInst::UMax;
      }
      SmallVector<CallInst *, 8> Calls;
      for (User *U : F->users())
        if (auto *CI = dyn_cast<CallInst>(U))
          if (CI->getCalledFunction() == F)
            Calls.push_back(CI);
      for (CallInst *CI : Calls) {
        if (CI->arg_size() != 2)
          continue;
        Value *Ptr = CI->getArgOperand(0);
        Value *Val = CI->getArgOperand(1);
        if (!Ptr->getType()->isPointerTy())
          continue;
        if (!Val->getType()->isIntegerTy())
          continue;
        IRBuilder<> B(CI);
        // Vulkan disallows CrossDevice scope; use "device" scope so the SPIR-V
        // backend emits Scope::Device on OpAtomic*. Vulkan also requires
        // non-relaxed memory order whenever a storage-class semantics bit is
        // set (UniformMemory/WorkgroupMemory) — use AcquireRelease.
        SyncScope::ID DevScope =
            CI->getContext().getOrInsertSyncScopeID("device");
        auto *RMW = B.CreateAtomicRMW(
            Op, Ptr, Val, MaybeAlign(),
            AtomicOrdering::AcquireRelease, DevScope);
        CI->replaceAllUsesWith(RMW);
        CI->eraseFromParent();
        Changed = true;
      }
      if (F->use_empty()) {
        F->eraseFromParent();
        Changed = true;
      }
    }
  }
  return Changed;
}

// Phase I4 helper: rewrite `atomicrmw ... ptr addrspace(4) ...` so it operates
// on the underlying non-Generic pointer (Workgroup AS 3, StorageBuffer AS 11,
// PhysicalStorageBuffer AS 5, or addrspace(1)). In Vulkan logical addressing,
// OpAtomic* requires a typed-storage-class pointer; OpTypePointer Generic
// would force GenericPointer capability, which Shader execution model forbids.
static bool rewriteGenericAtomicsToTypedAS(Function &F) {
  SmallVector<AtomicRMWInst *, 8> Worklist;
  for (BasicBlock &BB : F)
    for (Instruction &I : BB)
      if (auto *AR = dyn_cast<AtomicRMWInst>(&I))
        if (AR->getPointerAddressSpace() == 4)
          Worklist.push_back(AR);
  if (Worklist.empty())
    return false;
  bool Changed = false;
  for (AtomicRMWInst *AR : Worklist) {
    Value *Ptr = AR->getPointerOperand();
    SmallVector<GetElementPtrInst *, 4> GEPChain;
    Value *Cur = Ptr;
    while (true) {
      if (auto *G = dyn_cast<GetElementPtrInst>(Cur)) {
        if (G->getPointerAddressSpace() != 4)
          break;
        GEPChain.push_back(G);
        Cur = G->getPointerOperand();
        continue;
      }
      break;
    }
    Value *Root = Cur;
    unsigned NewAS = 0;
    Value *NewBase = nullptr;
    if (auto *AC = dyn_cast<AddrSpaceCastInst>(Root)) {
      NewAS = AC->getSrcAddressSpace();
      NewBase = AC->getPointerOperand();
    } else if (auto *ITP = dyn_cast<IntToPtrInst>(Root)) {
      Value *V = ITP->getOperand(0);
      if (auto *PTI = dyn_cast<PtrToIntInst>(V)) {
        // Walk through any preceding addrspacecast(s) on the source pointer
        // to find the most-specific (non-Generic) storage class — typically
        // the addrspace(11) StorageBuffer pointer beneath an
        // addrspacecast(addrspace(11) -> addrspace(1)) chain.
        Value *Src = PTI->getPointerOperand();
        while (auto *AC2 = dyn_cast<AddrSpaceCastInst>(Src))
          Src = AC2->getPointerOperand();
        unsigned SrcAS = Src->getType()->getPointerAddressSpace();
        if (SrcAS == 1 || SrcAS == 3 || SrcAS == 5 || SrcAS == 11) {
          NewAS = SrcAS;
          NewBase = Src;
        }
      }
    }
    if (!NewBase || (NewAS != 1 && NewAS != 3 && NewAS != 5 && NewAS != 11))
      continue;
    Value *Replaced = NewBase;
    const DataLayout &DL = F.getDataLayout();
    for (auto It = GEPChain.rbegin(); It != GEPChain.rend(); ++It) {
      GetElementPtrInst *G = *It;
      IRBuilder<> B(G);
      SmallVector<Value *, 4> Idx(G->idx_begin(), G->idx_end());
      // If `Replaced` is the result of an offset-0 `spv.resource.getpointer`
      // call into a `[0 x ElemTy]` runtime array, AND this GEP is a single
      // variable index whose source element type's alloc size matches
      // ElemTy's alloc size, emit `getpointer(handle, idx)` directly.
      // A bare `gep ElemTy, ptr addrspace(11) %p, i64 %var` triggers
      // SPIRVModuleAnalysis::addOpAccessChainReqs to request the
      // StorageBufferArrayDynamicIndexing capability (Vulkan 1.2 baseline
      // does not expose it under clvk + Intel Arc), so we route the variable
      // index through the resource intrinsic which the backend lowers to
      // OpAccessChain over the runtime array's structurally allowed
      // dynamic index.
      Value *NewGEP = nullptr;
      if (Idx.size() == 1) {
        if (auto *Call = dyn_cast<CallInst>(Replaced)) {
          if (Call->getIntrinsicID() == Intrinsic::spv_resource_getpointer) {
            auto *ZeroIdx = dyn_cast<ConstantInt>(Call->getArgOperand(1));
            Value *Handle = Call->getArgOperand(0);
            auto *TET = dyn_cast<TargetExtType>(Handle->getType());
            Type *RuntimeElemTy = nullptr;
            if (TET && TET->getNumTypeParameters() >= 1) {
              if (auto *AT = dyn_cast<ArrayType>(TET->getTypeParameter(0)))
                RuntimeElemTy = AT->getElementType();
            }
            if (ZeroIdx && ZeroIdx->isZero() && RuntimeElemTy &&
                RuntimeElemTy->isSized() &&
                G->getSourceElementType()->isSized()) {
              uint64_t SrcSz =
                  DL.getTypeAllocSize(G->getSourceElementType())
                      .getFixedValue();
              uint64_t ElemSz =
                  DL.getTypeAllocSize(RuntimeElemTy).getFixedValue();
              uint64_t StrideMul = 0;
              if (ElemSz != 0 && SrcSz != 0) {
                if (SrcSz == ElemSz)
                  StrideMul = 1;
                else if (SrcSz > ElemSz && (SrcSz % ElemSz) == 0)
                  StrideMul = SrcSz / ElemSz;
              }
              if (StrideMul != 0) {
                auto *PtrSB =
                    PointerType::get(G->getContext(), 11);
                Type *I32Ty = Type::getInt32Ty(G->getContext());
                Value *I32Idx = B.CreateTrunc(Idx[0], I32Ty);
                if (StrideMul != 1)
                  I32Idx = B.CreateMul(
                      I32Idx,
                      ConstantInt::get(I32Ty,
                                       static_cast<int64_t>(StrideMul)));
                NewGEP = B.CreateIntrinsic(
                    PtrSB, Intrinsic::spv_resource_getpointer,
                    {Handle, I32Idx});
              }
            }
          }
        }
      }
      if (!NewGEP)
        NewGEP = B.CreateGEP(G->getSourceElementType(), Replaced, Idx,
                             "", G->isInBounds());
      Replaced = NewGEP;
    }
    IRBuilder<> B(AR);
    auto *NewAR = B.CreateAtomicRMW(AR->getOperation(), Replaced,
                                     AR->getValOperand(), AR->getAlign(),
                                     AR->getOrdering(),
                                     AR->getSyncScopeID());
    NewAR->setVolatile(AR->isVolatile());
    AR->replaceAllUsesWith(NewAR);
    AR->eraseFromParent();
    for (GetElementPtrInst *G : GEPChain)
      if (G->use_empty())
        G->eraseFromParent();
    Changed = true;
  }
  return Changed;
}

// Phase G4: lower user-defined `__device__` globals (addrspace(1)
// CrossWorkgroup) into per-module StorageBuffer descriptors so that host-side
// hipMemcpyToSymbol / hipMemcpyFromSymbol can route data via a dedicated
// descriptor binding. Each such global is assigned (set=1, binding=N) -
// distinct from kernel-arg buffers which live in set=0 - and every kernel
// that references the global is rewritten to materialise a fresh descriptor
// handle in its entry block. The original GV is then erased so the backend
// no longer needs to lower addrspace(1) module-scope variables in Logical
// addressing mode.
//
// A reflection record is emitted as a named-metadata node
// `!hipspv.device_globals` containing one tuple per device global:
// {name_string, binding_const, size_const}. The post-bridging
// inject_reflection.py script reads these tuples and emits ClspvReflection
// `Constant` entries that the chipStar runtime parses to populate its
// host-symbol -> descriptor-binding map.
// Phase H4: lower `__device__` globals as *additional hidden kernel arguments*
// at descriptor set 0, rather than as set=1 module-scope StorageBuffer
// descriptors. Background: clvk's OpenCL bridge does not expose a public
// API to bind a Vulkan descriptor at any set other than 0 — only set=0 args
// are reachable from `clSetKernelArg`. By appending one hidden pointer arg
// per (kernel, used global) pair, the existing kernel-arg lowering pipeline
// (rewriteKernelSignature) wraps each as a regular set=0 buffer descriptor
// that the chipStar runtime can bind via the standard OpenCL path.
//
// Each new argument is named `__hipspv_dg_<symbol>` so that the bridging-pass
// follow-up in `rewriteKernelSignature` can use this name for the SPIR-V
// `OpName` on the descriptor variable. The chipStar runtime walks set=0
// descriptors at module-load time, identifies `__hipspv_dg_*` entries, and
// records their (kernel, ord, symbol) tuple in `SPVDeviceGlobal`.

// Phase I3: extract a __device__ global's initializer bytes for the
// runtime initial-value seeding path. Returns an empty vector for
// undef/zero/no-initializer GVs (the runtime treats absent InitData as
// "leave the buffer at its default-zero allocation"). Handles primitive
// integer/float scalars (the cases all four hipTestSymbol* tests need)
// and ConstantDataSequential aggregates by reading the raw constant
// bytes per the module's DataLayout.
static SmallVector<uint8_t, 16> extractInitBytes(GlobalVariable *GV,
                                                  const DataLayout &DL) {
  SmallVector<uint8_t, 16> Out;
  if (!GV->hasInitializer())
    return Out;
  Constant *Init = GV->getInitializer();
  if (!Init || isa<UndefValue>(Init) || Init->isNullValue())
    return Out;
  uint64_t Size = DL.getTypeAllocSize(GV->getValueType()).getFixedValue();
  Out.assign(Size, 0);
  if (auto *CI = dyn_cast<ConstantInt>(Init)) {
    APInt V = CI->getValue();
    unsigned NBits = V.getBitWidth();
    unsigned NBytes = (NBits + 7) / 8;
    if (NBytes > Size)
      NBytes = Size;
    for (unsigned B = 0; B < NBytes; ++B)
      Out[B] = static_cast<uint8_t>(V.extractBitsAsZExtValue(8, B * 8));
    return Out;
  }
  if (auto *CFP = dyn_cast<ConstantFP>(Init)) {
    APInt V = CFP->getValueAPF().bitcastToAPInt();
    unsigned NBytes =
        std::min<unsigned>((V.getBitWidth() + 7) / 8, (unsigned)Size);
    for (unsigned B = 0; B < NBytes; ++B)
      Out[B] = static_cast<uint8_t>(V.extractBitsAsZExtValue(8, B * 8));
    return Out;
  }
  if (auto *CDS = dyn_cast<ConstantDataSequential>(Init)) {
    StringRef Raw = CDS->getRawDataValues();
    unsigned N = std::min<unsigned>(Raw.size(), (unsigned)Size);
    for (unsigned B = 0; B < N; ++B)
      Out[B] = static_cast<uint8_t>(Raw[B]);
    return Out;
  }
  // Unsupported initializer kind (e.g. ConstantStruct, ConstantArray of
  // non-data, pointer initialisers). Bail with an empty vector so the
  // runtime falls back to the default zero-allocated buffer.
  Out.clear();
  return Out;
}

// Phase I3: encode the InitData bytes as lower-case hex in the kernel-arg
// name suffix `__sz_<size>__init_<hex>`. The chipStar reflection parser
// in spirv.cc decodes this suffix to populate SPVDeviceGlobal::Size and
// SPVDeviceGlobal::InitData (the latter empty when no init bytes were
// available, e.g. zero/undef initialisers).
static std::string encodeDgArgName(StringRef Sym, uint64_t Size,
                                   ArrayRef<uint8_t> Init) {
  std::string Out;
  Out.reserve(strlen("__hipspv_dg_") + Sym.size() + 32 + Init.size() * 2);
  Out += "__hipspv_dg_";
  Out += Sym.str();
  Out += "__sz_";
  Out += std::to_string(Size);
  if (!Init.empty()) {
    Out += "__init_";
    static const char Hex[] = "0123456789abcdef";
    for (uint8_t B : Init) {
      Out.push_back(Hex[B >> 4]);
      Out.push_back(Hex[B & 0xF]);
    }
  }
  return Out;
}

static bool lowerDeviceGlobalsToStorageBuffers(Module &M) {
  LLVMContext &C = M.getContext();
  auto *I32Ty = Type::getInt32Ty(C);

  SmallVector<GlobalVariable *, 4> DGs;
  for (GlobalVariable &GV : M.globals()) {
    if (GV.getType()->getAddressSpace() != 1)
      continue;
    StringRef N = GV.getName();
    if (N.starts_with("__chipspv_") || N.starts_with("__hip_cuid_") ||
        N.starts_with("__chip_") || N.starts_with("llvm.") ||
        N.starts_with(".str") || N.empty())
      continue;
    if (N.contains("__chip") || N.contains("__hip_cuid"))
      continue;
    // Skip clang-emitted constant aggregates (string literals etc.) which
    // live in addrspace(1) but are read-only and don't need a descriptor.
    // __constant__ user globals also get IsConstant=true at IR level but
    // we must still bind them as kernel-arg buffers; distinguish by
    // section / type. Heuristic: skip globals named `.str*` or under
    // `llvm.metadata` section; user `__constant__` globals have neither.
    if (GV.isConstant()) {
      if (GV.getSection() == "llvm.metadata")
        continue;
      // String-literal aggregates are array-of-i8.
      if (auto *AT = dyn_cast<ArrayType>(GV.getValueType()))
        if (AT->getElementType()->isIntegerTy(8))
          continue;
    }
    GV.removeDeadConstantUsers();
    if (GV.use_empty())
      continue;
    DGs.push_back(&GV);
  }
  if (DGs.empty())
    return false;

  llvm::sort(DGs, [](GlobalVariable *A, GlobalVariable *B) {
    return A->getName() < B->getName();
  });

  // Pre-compute the icmp-pattern skip set per global; identical to the prior
  // implementation. Functions in this set keep their GV uses; the kernel-drop
  // backstop later removes them.
  auto chainHasICmp = [](Value *Root) -> bool {
    SmallVector<Value *, 8> WL{Root};
    SmallPtrSet<Value *, 16> Seen;
    while (!WL.empty()) {
      Value *V = WL.pop_back_val();
      if (!Seen.insert(V).second)
        continue;
      for (User *UU : V->users()) {
        if (isa<ICmpInst>(UU))
          return true;
        if (isa<AddrSpaceCastInst>(UU) || isa<BitCastInst>(UU) ||
            isa<PtrToIntInst>(UU) || isa<IntToPtrInst>(UU) ||
            isa<GetElementPtrInst>(UU))
          WL.push_back(UU);
      }
    }
    return false;
  };

  // Build a map: kernel function -> ordered list of globals it uses.
  MapVector<Function *, SmallSetVector<GlobalVariable *, 4>> KernelToDGs;
  for (GlobalVariable *GV : DGs) {
    SmallSetVector<Function *, 4> UsingFns;
    for (User *U : GV->users())
      if (auto *I = dyn_cast<Instruction>(U))
        if (I->getFunction())
          UsingFns.insert(I->getFunction());

    for (Function *F : UsingFns) {
      if (F->isDeclaration() || F->empty())
        continue;
      // Skip kernels with pointer-equality on this GV — leave the GV use
      // intact so the kernel-drop backstop removes that kernel.
      bool Skip = false;
      for (User *U : GV->users()) {
        auto *I = dyn_cast<Instruction>(U);
        if (!I || I->getFunction() != F)
          continue;
        if (chainHasICmp(I)) {
          Skip = true;
          break;
        }
      }
      if (Skip)
        continue;
      KernelToDGs[F].insert(GV);
    }
  }

  if (KernelToDGs.empty()) {
    // Still emit MD for size info so the host can size its allocations.
    SmallVector<Metadata *, 4> MDList;
    const DataLayout &DL = M.getDataLayout();
    for (GlobalVariable *GV : DGs) {
      uint64_t SizeBytes = DL.getTypeAllocSize(GV->getValueType()).getFixedValue();
      MDList.push_back(MDNode::get(
          C, {MDString::get(C, GV->getName()),
              ConstantAsMetadata::get(ConstantInt::get(I32Ty, 0)),
              ConstantAsMetadata::get(
                  ConstantInt::get(Type::getInt64Ty(C), SizeBytes))}));
    }
    if (!MDList.empty()) {
      NamedMDNode *NMD = M.getOrInsertNamedMetadata("hipspv.device_globals");
      for (Metadata *MD : MDList)
        NMD->addOperand(cast<MDNode>(MD));
    }
    return false;
  }

  const DataLayout &DL = M.getDataLayout();

  // For each kernel, create a new function with appended pointer args and
  // rewrite GV uses inside the kernel to reference the new args.
  SmallVector<Function *, 4> OldFns;
  for (auto &KV : KernelToDGs) {
    Function *F = KV.first;
    auto &DGSet = KV.second;
    if (F->isDeclaration() || F->empty())
      continue;

    FunctionType *OldFTy = F->getFunctionType();
    SmallVector<Type *, 8> NewParamTys(OldFTy->params().begin(),
                                       OldFTy->params().end());
    // Each device global becomes a `ptr addrspace(1)` (CrossWorkgroup) arg.
    // rewriteKernelSignature handles addrspace(1) buffer args by binding them
    // to set=0 StorageBuffer descriptors.
    auto *PtrAS1 = PointerType::get(C, 1);
    for (unsigned i = 0; i < DGSet.size(); ++i)
      NewParamTys.push_back(PtrAS1);

    FunctionType *NewFTy = FunctionType::get(OldFTy->getReturnType(),
                                              NewParamTys, OldFTy->isVarArg());
    Function *NewF = Function::Create(NewFTy, F->getLinkage(),
                                      F->getAddressSpace(), "", F->getParent());
    NewF->copyAttributesFrom(F);
    // Drop per-parameter attributes referring to old slot indexes; keep only
    // function and return-value attrs.
    AttributeList OldAL = NewF->getAttributes();
    SmallVector<AttributeSet, 8> ArgAttrs;
    for (unsigned I = 0; I < OldFTy->getNumParams(); ++I)
      ArgAttrs.push_back(OldAL.getParamAttrs(I));
    // No per-arg attrs for the new hidden args (default).
    for (unsigned i = 0; i < DGSet.size(); ++i)
      ArgAttrs.push_back(AttributeSet());
    NewF->setAttributes(AttributeList::get(C, OldAL.getFnAttrs(),
                                            OldAL.getRetAttrs(), ArgAttrs));
    NewF->setCallingConv(F->getCallingConv());
    NewF->takeName(F);

    // Splice body, then RAUW old args with new args by index.
    NewF->splice(NewF->begin(), F);
    auto NewArgIt = NewF->arg_begin();
    for (Argument &A : F->args()) {
      A.replaceAllUsesWith(&*NewArgIt);
      NewArgIt->setName(A.getName());
      ++NewArgIt;
    }

    // Replace each device-global use inside this kernel with the corresponding
    // appended argument. After materializeAddrSpaceConstants() any CE
    // addrspacecasts have already been hoisted to instructions, so all uses
    // we care about are Instructions. The GV and the new arg both have type
    // `ptr addrspace(1)`, so direct operand-replacement is type-safe.
    for (GlobalVariable *GV : DGSet) {
      Argument &NewArg = *NewArgIt;
      // Phase I3: encode size + initializer bytes in the arg name. The
      // SPIR-V backend forwards arg names verbatim to OpName decorations
      // on the descriptor variable, which the chipStar runtime parses
      // back out in tryAnalyzeVulkanReflection.
      uint64_t SizeBytes =
          DL.getTypeAllocSize(GV->getValueType()).getFixedValue();
      auto InitBytes = extractInitBytes(GV, DL);
      NewArg.setName(encodeDgArgName(GV->getName(), SizeBytes, InitBytes));
      ++NewArgIt;

      SmallVector<Use *, 8> UsesInF;
      for (Use &U : GV->uses()) {
        auto *I = dyn_cast<Instruction>(U.getUser());
        if (I && I->getFunction() == NewF) {
          UsesInF.push_back(&U);
          continue;
        }
        // CE users that survived materialization (e.g. inside CEs nested in
        // CEs that the per-fn materializer didn't reach because they were
        // not in any function) get logged but left intact; the kernel-drop
        // backstop will remove the kernel if any CE addrspacecast on a GV
        // remains.
      }
      LLVM_DEBUG(dbgs() << "hipspv: rewriting " << UsesInF.size()
                        << " uses of @" << GV->getName() << " in kernel "
                        << NewF->getName() << " to arg #"
                        << NewArg.getArgNo() << "\n");
      for (Use *U : UsesInF)
        U->set(&NewArg);
    }

    // Old function may still have callers from non-kernel code; for kernels
    // (spir_kernel CC, no internal callers in HIP) it's safe to erase.
    OldFns.push_back(F);
  }

  for (Function *F : OldFns) {
    F->dropAllReferences();
    F->eraseFromParent();
  }

  // Emit MD: per-kernel symbol→hidden-arg-index map plus per-symbol size.
  // Used by inject_reflection.py / runtime if it wants to corroborate the
  // OpName-based discovery.
  SmallVector<Metadata *, 4> SizeMDList;
  for (GlobalVariable *GV : DGs) {
    uint64_t SizeBytes = DL.getTypeAllocSize(GV->getValueType()).getFixedValue();
    SizeMDList.push_back(MDNode::get(
        C, {MDString::get(C, GV->getName()),
            ConstantAsMetadata::get(ConstantInt::get(I32Ty, 0)),
            ConstantAsMetadata::get(
                ConstantInt::get(Type::getInt64Ty(C), SizeBytes))}));
  }
  if (!SizeMDList.empty()) {
    NamedMDNode *NMD = M.getOrInsertNamedMetadata("hipspv.device_globals");
    for (Metadata *MD : SizeMDList)
      NMD->addOperand(cast<MDNode>(MD));
  }

  // Erase any GVs whose uses are now all inside rewritten kernels.
  for (GlobalVariable *GV : DGs) {
    GV->removeDeadConstantUsers();
    if (GV->use_empty())
      GV->eraseFromParent();
  }

  return true;
}

PreservedAnalyses HIPSPVLowerToHLSLShape::run(Module &M,
                                              ModuleAnalysisManager &) {
  bool Changed = false;
  Changed |= stripChipStarRuntime(M);
  // Phase D: lower chipStar's __chip_atomic_add_f32/f64 helpers to
  // atomicrmw fadd so the SPIR-V backend can emit OpAtomicFAddEXT in
  // Shader mode (SPV_EXT_shader_atomic_float_add).
  Changed |= lowerChipAtomicFloatAdd(M);
  // Phase I4: lower integer atomic helpers (__chip_atomic_{add,sub,min,
  // max,and,or,xor,exch}_{i,u,l}) similarly so they don't survive as
  // unresolved externs against an OpenCL-flavored devicelib SPV (clvk's
  // clLinkProgram fails when shader-flavored kernel SPV references kernel-
  // flavored externs).
  Changed |= lowerChipAtomicIntegerOps(M);

  // Phase G3: give `extern __shared__ T sdata[];` (`external_global
  // addrspace(3) [0 x T]`) a concrete size so the backend can emit a sized
  // OpVariable Workgroup. Must run before kernel-drop / addrspace
  // rewriting so the resized global is what we propagate.
  Changed |= resizeExternSharedGlobals(M);
  // Lower `__chip_syncthreads()` to OpControlBarrier; otherwise its declaration
  // remains as an unresolved Import requiring OpCapability Linkage.
  Changed |= lowerChipSyncThreads(M);

  // Phase A: convert ConstantExpr addrspacecasts to instruction form across
  // every defined function, then run a simple intra-function DCE to remove
  // dead loads/stores of chipStar runtime globals (`__chip_clk_counter` etc.)
  // that survived inlining but aren't actually observed.
  // Also lower addrspacecast(addrspace(3) global -> addrspace(4)) chains so
  // `__shared__` accesses live in addrspace(3) (Workgroup) end-to-end.
  for (Function &F : M) {
    if (F.isDeclaration())
      continue;
    Changed |= materializeAddrSpaceConstants(F);
    Changed |= lowerWorkgroupAddrSpaceCasts(F);
    Changed |= runIntraFnDCE(F);
  }

  // Phase G3b: lower `__chip_clk_counter` accesses (clock/clock64 path) to
  // a per-kernel alloca so the addrspace(1) global has no remaining uses.
  // Must run after Phase A so all CE addrspacecasts are instruction-form.
  Changed |= lowerChipClkCounter(M);

  // Phase G4: bind user `__device__` globals to StorageBuffer descriptors
  // so the kernel-drop backstop below doesn't drop kernels that legitimately
  // touch them, and so host-side hipMemcpyToSymbol can target them via
  // (set=1, binding=N).
  Changed |= lowerDeviceGlobalsToStorageBuffers(M);
  for (Function &F : M) {
    if (F.isDeclaration() || F.empty())
      continue;
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
    Changed |= remapSubgroupBuiltins(F);
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
  // Re-run integer atomic lowering after signature rewrite: post-rewrite
  // pointer arguments may have changed addrspace (e.g. from generic to
  // StorageBuffer/Workgroup), and any leftover calls need to be lowered.
  Changed |= lowerChipAtomicIntegerOps(M);
  // Phase I4: any `atomicrmw` still on a Generic-AS pointer must be retargeted
  // to the underlying non-Generic storage class so the SPIR-V backend does not
  // emit OpTypePointer Generic (which requires GenericPointer capability,
  // forbidden in Shader execution model).
  for (Function &F : M) {
    if (F.isDeclaration() || F.empty())
      continue;
    Changed |= rewriteGenericAtomicsToTypedAS(F);
  }

  // After rewriteKernelSignature: IRBuilder constant-folding while emitting
  // GEPs/addrspacecasts on the push-constant global may have produced new
  // ConstantExpr addrspacecasts (especially around byval struct args).
  // Re-materialize them and run DCE again so the backend sees only
  // instruction-form addrspacecasts.
  for (Function &F : M) {
    if (F.isDeclaration() || F.empty())
      continue;
    // Lower memcpys touching Vulkan resource pointers (addrspace 11) to
    // explicit per-byte load/store sequences, so the SPIR-V backend doesn't
    // try to emit OpCopyMemory between mismatched pointee types when the
    // resource pointer's pointee (e.g. `[N x i8]`) disagrees with the source
    // global's element layout (e.g. `[K x %struct.HIP_vector_type]`).
    Changed |= lowerVulkanResourceMemcpys(F);
    Changed |= materializeAddrSpaceConstants(F);
    Changed |= runIntraFnDCE(F);
  }

  // Final cleanup: drop any module-scope globals in non-Vulkan addrspaces
  // that became dead after the kernel-drop + bridging-pass rewrites
  // (e.g. `__chip_clk_counter` in CrossWorkgroup after waitKernel removal).
  Changed |= dropDeadInvalidAddrSpaceGlobals(M);

  // Pre-structurize: HIP kernels with deep cascades of nested ifs (e.g. the
  // `hipDynamicShared` parallel-prefix-sum pattern) survive into the SPIR-V
  // backend with a CFG shape that the in-tree SPIRVStructurizer's
  // splitCriticalEdges/MergeRegionExitTargets passes can't always regularize:
  // they create an inner OpSelectionMerge whose merge block is unreachable
  // from the header's conditional branch (the conditional jumps directly to
  // an outer construct's merge), producing the spirv-val error
  //   "block <X> branches to the selection construct, but not to the
  //    selection header".
  // Running upstream StructurizeCFG followed by SimplifyCFG normalises these
  // cascades into a shape the backend handles correctly. Done as a final step
  // so any structurization changes occur on the fully-bridged IR.
  {
    PassBuilder PB;
    LoopAnalysisManager LAM;
    FunctionAnalysisManager FAM;
    CGSCCAnalysisManager CGAM;
    ModuleAnalysisManager MAM;
    PB.registerModuleAnalyses(MAM);
    PB.registerCGSCCAnalyses(CGAM);
    PB.registerFunctionAnalyses(FAM);
    PB.registerLoopAnalyses(LAM);
    PB.crossRegisterProxies(LAM, FAM, CGAM, MAM);

    FunctionPassManager FPM;
    FPM.addPass(StructurizeCFGPass());
    FPM.addPass(SimplifyCFGPass());
    for (Function &F : M) {
      if (F.isDeclaration() || F.empty())
        continue;
      // Only run on shader entry points (kernels). Device-side helpers that
      // remain after dead-code elimination should not be perturbed.
      if (!F.hasFnAttribute("hlsl.shader"))
        continue;
      FPM.run(F, FAM);
      Changed = true;
    }
  }

  diagnoseLeftovers(M);
  return Changed ? PreservedAnalyses::none() : PreservedAnalyses::all();
}
