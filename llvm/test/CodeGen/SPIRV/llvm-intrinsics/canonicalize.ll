; llvm.canonicalize has no SPIR-V instruction; it becomes a multiplication by
; one, which canonicalizes its operand.
; RUN: llc -O0 -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s
; RUN: llc -O0 -mtriple=spirv1.6-unknown-vulkan1.3-compute %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; CHECK-DAG: %[[#F:]] = OpTypeFloat 32
; CHECK-DAG: %[[#ONE:]] = OpConstant %[[#F]] 1
; CHECK: OpFMul %[[#F]] %[[#]] %[[#ONE]]

define spir_func float @f(float %x) {
  %c = call float @llvm.canonicalize.f32(float %x)
  ret float %c
}

define void @main() #0 {
  ret void
}

attributes #0 = { "hlsl.numthreads"="1,1,1" "hlsl.shader"="compute" }
