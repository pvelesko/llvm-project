; GLSL.std.450 has no saturating arithmetic; shaders expand it to min/max
; and overflow checks instead of OpenCL.std's u_add_sat and friends.
; RUN: llc -O0 -mtriple=spirv1.6-unknown-vulkan1.3-compute %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv1.6-unknown-vulkan1.3-compute %s -o - -filetype=obj | spirv-val --target-env vulkan1.3 %}

; CHECK-NOT: _sat
; CHECK: OpExtInst %[[#]] %[[#]] UMin
; CHECK: OpISub

@.str = private constant [4 x i8] c"buf\00"
define void @main() #0 {
  %h = call target("spirv.VulkanBuffer", [0 x i32], 12, 1) @llvm.spv.resource.handlefrombinding(i32 0, i32 0, i32 1, i32 0, ptr @.str)
  %p = call ptr addrspace(11) @llvm.spv.resource.getpointer(target("spirv.VulkanBuffer", [0 x i32], 12, 1) %h, i32 0)
  %x = load i32, ptr addrspace(11) %p
  %a = call i32 @llvm.usub.sat.i32(i32 %x, i32 7)
  %b = call i32 @llvm.uadd.sat.i32(i32 %a, i32 %x)
  %c = call i32 @llvm.ssub.sat.i32(i32 %b, i32 %x)
  %d = call i32 @llvm.sadd.sat.i32(i32 %c, i32 %x)
  store i32 %d, ptr addrspace(11) %p
  ret void
}
attributes #0 = { "hlsl.numthreads"="1,1,1" "hlsl.shader"="compute" }
