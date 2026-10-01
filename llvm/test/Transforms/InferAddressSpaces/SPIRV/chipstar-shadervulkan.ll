; ShaderVulkan keeps clang's address space map in which 4 is generic.
; RUN: opt -mtriple=spirv64-unknown-chipstar-shadervulkan -passes=infer-address-spaces -S %s | FileCheck %s

; CHECK-LABEL: @f(
; CHECK: load float, ptr addrspace(3) %p
define float @f(ptr addrspace(3) %p) {
  %g = addrspacecast ptr addrspace(3) %p to ptr addrspace(4)
  %v = load float, ptr addrspace(4) %g, align 4
  ret float %v
}
