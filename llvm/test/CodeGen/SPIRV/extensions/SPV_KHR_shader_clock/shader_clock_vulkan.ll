; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv-unknown-vulkan1.3-compute --spirv-ext=+SPV_KHR_shader_clock %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv-unknown-vulkan1.3-compute --spirv-ext=+SPV_KHR_shader_clock %s -o - -filetype=obj | spirv-val --target-env vulkan1.3 %}

; CHECK: OpCapability ShaderClockKHR
; CHECK: OpExtension "SPV_KHR_shader_clock"
; CHECK-DAG: [[uint:%[0-9]+]] = OpTypeInt 32 0
; CHECK-DAG: [[ulong:%[0-9]+]] = OpTypeInt 64 0
; CHECK-DAG: [[subgroup:%[0-9]+]] = OpConstant [[uint]] 3
; CHECK: OpReadClockKHR [[ulong]] [[subgroup]]

@out = internal addrspace(10) global i64 0

define void @main() #0 {
entry:
  %t = call spir_func i64 @_Z20__spirv_ReadClockKHRi(i32 3)
  store i64 %t, ptr addrspace(10) @out, align 8
  ret void
}

declare spir_func i64 @_Z20__spirv_ReadClockKHRi(i32)

attributes #0 = { "hlsl.numthreads"="1,1,1" "hlsl.shader"="compute" }
