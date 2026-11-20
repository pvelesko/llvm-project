; A byte GEP on another byte GEP folds into one access chain on the array,
; instead of stepping past the element the inner GEP selects.
; RUN: llc -O0 -mtriple=spirv1.6-unknown-vulkan1.3-compute %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv1.6-unknown-vulkan1.3-compute %s -o - -filetype=obj | spirv-val --target-env vulkan1.3 %}

; CHECK-DAG: %[[#T:]] = OpVariable %[[#]] Private
; CHECK-DAG: %[[#R:]] = OpVariable %[[#]] Private
; CHECK: %[[#Next:]] = OpIAdd %[[#]] %[[#]] %[[#]]
; CHECK: %[[#P:]] = OpAccessChain %[[#]] %[[#T]] %[[#Next]]
; CHECK: OpLoad %[[#]] %[[#P]]

; A step past the element splits into whole elements and a byte offset within.
; CHECK: %[[#RNext:]] = OpIAdd %[[#]] %[[#]] %[[#]]
; CHECK: %[[#RE:]] = OpAccessChain %[[#]] %[[#R]] %[[#RNext]]
; CHECK: %[[#RP:]] = OpAccessChain %[[#]] %[[#RE]] %[[#]]
; CHECK: OpLoad %[[#]] %[[#RP]]

%rec = type { float, float, float }
@t = internal addrspace(10) constant [4 x float] [float 1.0, float 2.0, float 3.0, float 4.0]
@r = internal addrspace(10) constant [2 x %rec] zeroinitializer
@out = internal addrspace(10) global float 0.0

define void @main() #0 {
  %tid = call i32 @llvm.spv.thread.id.in.group.i32(i32 0)
  %i = zext i32 %tid to i64
  %p = getelementptr [4 x i8], ptr addrspace(10) @t, i64 %i
  %q = getelementptr i8, ptr addrspace(10) %p, i64 4
  %v = load float, ptr addrspace(10) %q, align 4
  store float %v, ptr addrspace(10) @out, align 4
  %rp = getelementptr [12 x i8], ptr addrspace(10) @r, i64 %i
  %rq = getelementptr i8, ptr addrspace(10) %rp, i64 16
  %rv = load float, ptr addrspace(10) %rq, align 4
  store float %rv, ptr addrspace(10) @out, align 4
  ret void
}

declare i32 @llvm.spv.thread.id.in.group.i32(i32)

attributes #0 = { "hlsl.numthreads"="1,1,1" "hlsl.shader"="compute" }
