; llc lowers HIP kernels for the ShaderVulkan environment without a separate
; opt step; either pass manager schedules it after infer-address-spaces and
; before loop strength reduction, which would make the loop's pointer a phi,
; and runs SROA after it, so that a by-value argument read as another type
; needs no Function variable.
; RUN: llc -O0 -mtriple=spirv64-unknown-chipstar-shadervulkan %s -o - | FileCheck %s
; RUN: llc -O2 -mtriple=spirv64-unknown-chipstar-shadervulkan %s -o - | FileCheck %s
; RUN: llc -O0 -mtriple=spirv64-unknown-chipstar-shadervulkan -enable-new-pm -print-pipeline-passes %s -o /dev/null | FileCheck %s --check-prefix=NPM
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-chipstar-shadervulkan %s -o - -filetype=obj | spirv-val --target-env vulkan1.3 %}
; RUN: %if spirv-tools %{ llc -O2 -mtriple=spirv64-unknown-chipstar-shadervulkan %s -o - -filetype=obj | spirv-val --target-env vulkan1.3 %}

; CHECK: OpEntryPoint GLCompute %[[#]] "store_one"
; CHECK: OpEntryPoint GLCompute %[[#]] "fill"
; CHECK: OpEntryPoint GLCompute %[[#]] "pun"
; CHECK: OpExecutionModeId %[[#]] LocalSizeId
; CHECK-LABEL: Begin function pun
; CHECK: OpBitcast
; NPM: infer-address-spaces){{.*}}hipspv-lower-to-hlsl-shape,function(sroa

define spir_kernel void @store_one(ptr addrspace(1) %p) {
  store i32 1, ptr addrspace(1) %p, align 4
  ret void
}

define spir_kernel void @fill(ptr addrspace(1) %p, i64 %n) {
entry:
  %empty = icmp eq i64 %n, 0
  br i1 %empty, label %exit, label %loop

loop:
  %i = phi i64 [ 0, %entry ], [ %next, %loop ]
  %gep = getelementptr inbounds i32, ptr addrspace(1) %p, i64 %i
  store i32 0, ptr addrspace(1) %gep, align 4
  %next = add nuw i64 %i, 1
  %done = icmp eq i64 %next, %n
  br i1 %done, label %exit, label %loop

exit:
  ret void
}

%struct.C = type { <2 x float> }

define spir_kernel void @pun(ptr addrspace(1) %out, ptr byval(%struct.C) align 8 %c) {
  %v = load i64, ptr %c, align 8
  store i64 %v, ptr addrspace(1) %out, align 8
  ret void
}
