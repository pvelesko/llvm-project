; spirv.workgroup-size-spec-ids gives a compute entry point a LocalSizeId of
; specialization constants, defaulting to 1, in place of its other work-group
; size execution modes.
; RUN: llc -O0 -mtriple=spirv1.6-unknown-vulkan1.3-compute %s -o - | FileCheck %s
; RUN: llc -O0 -mtriple=spirv1.6-unknown-vulkan1.3-compute %s -o - | FileCheck %s --check-prefix=DEDUP
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv1.6-unknown-vulkan1.3-compute %s -o - -filetype=obj | spirv-val --target-env vulkan1.3 %}
; RUN: sed 's/"0,1,2"/"0,1"/' %s | not llc -O0 -mtriple=spirv1.6-unknown-vulkan1.3-compute -o /dev/null 2>&1 | FileCheck %s --check-prefix=BAD
; RUN: not llc -O0 -mtriple=spirv64v1.0-unknown-unknown %s -o /dev/null 2>&1 | FileCheck %s --check-prefix=V10

; Spec constants with the same SpecId and value are one constant: @d and its
; __spirv_SpecConstant(0, 1) share @a's.
; DEDUP: OpExecutionModeId %[[#]] LocalSizeId %[[#X:]] %[[#Y:]] %[[#Z:]]
; DEDUP: OpExecutionModeId %[[#]] LocalSizeId %[[#X]] %[[#Y]] %[[#Z]]
; DEDUP: OpDecorate %[[#X]] SpecId 0
; DEDUP-NOT: SpecId 0

; BAD: invalid spirv.workgroup-size-spec-ids on a
; V10: spirv.workgroup-size-spec-ids requires SPIR-V 1.2

; CHECK-DAG: OpEntryPoint GLCompute %[[#A:]] "a"
; CHECK-DAG: OpEntryPoint GLCompute %[[#B:]] "b"
; CHECK-DAG: OpEntryPoint GLCompute %[[#C:]] "c"
; CHECK-DAG: OpExecutionModeId %[[#A]] LocalSizeId %[[#AX:]] %[[#AY:]] %[[#AZ:]]
; CHECK-DAG: OpExecutionModeId %[[#B]] LocalSizeId %[[#BX:]] %[[#BY:]] %[[#BZ:]]
; CHECK-DAG: OpExecutionModeId %[[#C]] LocalSizeId %[[#CX:]] %[[#CY:]] %[[#CZ:]]
; CHECK-NOT: OpExecutionMode {{.*}} LocalSize
; CHECK-DAG: OpDecorate %[[#AX]] SpecId 0
; CHECK-DAG: OpDecorate %[[#AY]] SpecId 1
; CHECK-DAG: OpDecorate %[[#AZ]] SpecId 2
; CHECK-DAG: OpDecorate %[[#BX]] SpecId 3
; CHECK-DAG: OpDecorate %[[#BY]] SpecId 4
; CHECK-DAG: OpDecorate %[[#BZ]] SpecId 5
; CHECK-DAG: %[[#AX]] = OpSpecConstant %[[#]] 1
; CHECK-DAG: %[[#BX]] = OpSpecConstant %[[#]] 1
; CHECK-DAG: %[[#CX]] = OpSpecConstant %[[#]] 1

define void @a() #0 {
  ret void
}

define void @b() #1 {
  ret void
}

define void @d() #0 {
  %x = call i32 @_Z20__spirv_SpecConstantii(i32 0, i32 1)
  ret void
}

declare i32 @_Z20__spirv_SpecConstantii(i32, i32)

define void @c() #2 !reqd_work_group_size !0 !work_group_size_hint !0 {
  ret void
}

attributes #0 = { "hlsl.shader"="compute" "spirv.workgroup-size-spec-ids"="0,1,2" }
attributes #1 = { "hlsl.shader"="compute" "hlsl.numthreads"="8,4,1" "spirv.workgroup-size-spec-ids"="3,4,5" }
attributes #2 = { "hlsl.shader"="compute" "spirv.workgroup-size-spec-ids"="6,7,8" }

!0 = !{i32 4, i32 2, i32 1}

; Also replaces LocalSize requested through spirv.ExecutionMode.
!spirv.ExecutionMode = !{!1}
!1 = !{ptr @a, i32 17, i32 8, i32 1, i32 1}
