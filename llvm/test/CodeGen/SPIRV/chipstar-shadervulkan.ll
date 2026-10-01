; spirv64-unknown-chipstar-shadervulkan targets the Vulkan (Shader) environment
; with logical addressing, unlike plain spirv64.
; RUN: llc -O0 -mtriple=spirv64-unknown-chipstar-shadervulkan %s -o - | FileCheck %s --implicit-check-not=Addresses
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-chipstar-shadervulkan %s -o - -filetype=obj | spirv-val --target-env vulkan1.3 %}
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-chipstar-shadervulkan %s -o - -filetype=obj | spirv-dis | FileCheck %s --check-prefix=VER %}

; RUN: not llc -O0 -mtriple=spirv64v1.2-unknown-chipstar-shadervulkan %s -o /dev/null 2>&1 | FileCheck %s --check-prefix=V12

; VER: Version: 1.6
; V12: the ShaderVulkan environment requires SPIR-V 1.3

; CHECK-DAG: OpCapability Shader
; CHECK-DAG: OpMemoryModel Logical GLSL450
; CHECK-DAG: OpEntryPoint GLCompute %[[#]] "main"
; CHECK-DAG: OpExecutionMode %[[#]] LocalSize 64 1 1
; CHECK: OpInBoundsAccessChain
; CHECK-NOT: OpInBoundsPtrAccessChain

define void @main() #0 {
entry:
  %a = alloca [4 x i32], align 4
  %id = call i32 @llvm.spv.thread.id.in.group.i32(i32 0)
  %i = and i32 %id, 3
  %p = getelementptr inbounds [4 x i32], ptr %a, i32 0, i32 %i
  store i32 %id, ptr %p, align 4
  ret void
}

attributes #0 = { "hlsl.numthreads"="64,1,1" "hlsl.shader"="compute" }
