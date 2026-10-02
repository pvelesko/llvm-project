; GLSL.std.450 Round rounds halfway cases either way, but llvm.round and
; llvm.lround round them away from zero, so shaders expand them.
; RUN: llc -O0 -mtriple=spirv1.6-unknown-vulkan1.3-compute %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv1.6-unknown-vulkan1.3-compute %s -o - -filetype=obj | spirv-val --target-env vulkan1.3 %}

; CHECK-NOT: {{ Round }}
; CHECK: OpExtInst %[[#]] %[[#]] Trunc
; CHECK: OpExtInst %[[#]] %[[#]] Trunc
; CHECK: OpConvertFToS

@.str = private constant [4 x i8] c"buf\00"
define void @main() #0 {
  %h = call target("spirv.VulkanBuffer", [0 x float], 12, 1) @llvm.spv.resource.handlefrombinding(i32 0, i32 0, i32 1, i32 0, ptr @.str)
  %p = call ptr addrspace(11) @llvm.spv.resource.getpointer(target("spirv.VulkanBuffer", [0 x float], 12, 1) %h, i32 0)
  %x = load float, ptr addrspace(11) %p
  %r = call float @llvm.round.f32(float %x)
  %l = call i32 @llvm.lround.i32.f32(float %x)
  %lf = sitofp i32 %l to float
  %s = fadd float %r, %lf
  store float %s, ptr addrspace(11) %p
  ret void
}
attributes #0 = { "hlsl.numthreads"="1,1,1" "hlsl.shader"="compute" }
