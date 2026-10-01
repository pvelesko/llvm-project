// The ShaderVulkan environment is only accepted with the chipstar OS and no vendor.
// RUN: %clang_cc1 -triple spirv64-unknown-chipstar-shadervulkan -fsyntax-only -verify %s
// RUN: not %clang_cc1 -triple spirv64-unknown-vulkan-shadervulkan -fsyntax-only %s 2>&1 | FileCheck %s
// RUN: not %clang_cc1 -triple spirv64-unknown-unknown-shadervulkan -fsyntax-only %s 2>&1 | FileCheck %s
// RUN: not %clang_cc1 -triple spirv64-intel-chipstar-shadervulkan -fsyntax-only %s 2>&1 | FileCheck %s

// expected-no-diagnostics
// CHECK: error: unknown target triple
