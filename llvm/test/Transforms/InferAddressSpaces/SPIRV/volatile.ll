; SPIR-V's Volatile memory operand applies in every storage class, so volatile
; accesses through a generic pointer move to the pointer's address space.
; RUN: opt -mtriple=spirv64-unknown-unknown -S -passes=infer-address-spaces %s | FileCheck %s

; CHECK-LABEL: @volatile_global(
; CHECK: %v = load volatile i32, ptr addrspace(1) %p, align 4
; CHECK: store volatile i32 %v, ptr addrspace(1) %p, align 4
define spir_func void @volatile_global(ptr addrspace(1) %p) {
  %g = addrspacecast ptr addrspace(1) %p to ptr addrspace(4)
  %v = load volatile i32, ptr addrspace(4) %g, align 4
  store volatile i32 %v, ptr addrspace(4) %g, align 4
  ret void
}

; CHECK-LABEL: @volatile_private(
; CHECK: %a = alloca i32, align 4
; CHECK: store volatile i32 1, ptr %a, align 4
define spir_func void @volatile_private() {
  %a = alloca i32, align 4
  %g = addrspacecast ptr %a to ptr addrspace(4)
  store volatile i32 1, ptr addrspace(4) %g, align 4
  ret void
}
