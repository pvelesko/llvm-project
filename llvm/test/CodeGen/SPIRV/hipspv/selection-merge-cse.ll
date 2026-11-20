; MachineCSE hoists an address computation common to both arms of a branch
; into the header, before its terminators; it must not land between
; OpSelectionMerge and the OpBranchConditional it precedes.
; RUN: llc -O3 -mtriple=spirv64-unknown-chipstar-shadervulkan %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O3 -mtriple=spirv64-unknown-chipstar-shadervulkan %s -o - -filetype=obj | spirv-val --target-env vulkan1.3 %}

; CHECK: OpSelectionMerge
; CHECK-NEXT: OpBranchConditional

define spir_kernel void @k(ptr addrspace(1) %out, i64 %i, i1 %c) {
entry:
  %p = getelementptr [4 x i8], ptr addrspace(1) %out, i64 %i
  br i1 %c, label %then, label %join

then:
  store float 0.000000e+00, ptr addrspace(1) %p, align 4
  br label %join

join:
  store float 1.000000e+00, ptr addrspace(1) %p, align 4
  ret void
}
