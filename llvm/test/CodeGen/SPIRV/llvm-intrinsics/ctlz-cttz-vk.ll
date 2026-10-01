; GLSL.std.450 has no count of leading or trailing zeros; shaders expand
; llvm.ctlz and llvm.cttz to shifts and OpBitCount.
; RUN: llc -O0 -mtriple=spirv1.6-unknown-vulkan1.3-compute %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv1.6-unknown-vulkan1.3-compute %s -o - -filetype=obj | spirv-val --target-env vulkan1.3 %}

; CHECK-NOT: OpExtInst
; CHECK: OpBitCount
; CHECK: OpBitCount

@.str = private constant [4 x i8] c"buf\00"
define void @main() #0 {
  %h = call target("spirv.VulkanBuffer", [0 x i32], 12, 1) @llvm.spv.resource.handlefrombinding(i32 0, i32 0, i32 1, i32 0, ptr @.str)
  %p = call ptr addrspace(11) @llvm.spv.resource.getpointer(target("spirv.VulkanBuffer", [0 x i32], 12, 1) %h, i32 0)
  %x = load i32, ptr addrspace(11) %p
  %a = call i32 @llvm.ctlz.i32(i32 %x, i1 false)
  %b = call i32 @llvm.cttz.i32(i32 %x, i1 true)
  %x64 = zext i32 %x to i64
  %c = call i64 @llvm.ctlz.i64(i64 %x64, i1 false)
  %c32 = trunc i64 %c to i32
  %s = add i32 %a, %b
  %t = add i32 %s, %c32
  store i32 %t, ptr addrspace(11) %p
  ret void
}
attributes #0 = { "hlsl.numthreads"="1,1,1" "hlsl.shader"="compute" }
