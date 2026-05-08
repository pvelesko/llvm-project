//===- HIPSPVLowerToHLSLShape.h - HIP IR -> HLSL-shape rewrite --*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//

#ifndef LLVM_LIB_TARGET_SPIRV_HIPSPVLOWERTOHLSLSHAPE_H
#define LLVM_LIB_TARGET_SPIRV_HIPSPVLOWERTOHLSLSHAPE_H

#include "llvm/IR/PassManager.h"

namespace llvm {

class HIPSPVLowerToHLSLShape
    : public PassInfoMixin<HIPSPVLowerToHLSLShape> {
public:
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &AM);
};

} // namespace llvm

#endif // LLVM_LIB_TARGET_SPIRV_HIPSPVLOWERTOHLSLSHAPE_H
