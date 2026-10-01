; The helper a funnel shift is lowered to is not exported from a shader:
; Vulkan has no linking.
; RUN: llc -O0 -mtriple=spirv1.6-unknown-vulkan1.3-compute %s -o - | FileCheck %s --implicit-check-not=Linkage
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv1.6-unknown-vulkan1.3-compute %s -o - -filetype=obj | spirv-val --target-env vulkan1.3 %}

; CHECK: OpFunctionCall %[[#]] %[[#]]

@.str = private constant [4 x i8] c"buf\00"

define void @main() #0 {
  %h = call target("spirv.VulkanBuffer", [0 x i32], 12, 1) @llvm.spv.resource.handlefrombinding(i32 0, i32 0, i32 1, i32 0, ptr @.str)
  %p = call ptr addrspace(11) @llvm.spv.resource.getpointer(target("spirv.VulkanBuffer", [0 x i32], 12, 1) %h, i32 0)
  %x = load i32, ptr addrspace(11) %p
  %r = call i32 @llvm.fshl.i32(i32 %x, i32 %x, i32 %x)
  store i32 %r, ptr addrspace(11) %p
  ret void
}

attributes #0 = { "hlsl.numthreads"="1,1,1" "hlsl.shader"="compute" }
