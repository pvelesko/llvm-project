; OpOrdered and OpUnordered need the Kernel capability; shaders compare each
; operand with itself instead.
; RUN: llc -O0 -mtriple=spirv1.6-unknown-vulkan1.3-compute %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv1.6-unknown-vulkan1.3-compute %s -o - -filetype=obj | spirv-val --target-env vulkan1.3 %}

; CHECK-NOT: OpOrdered
; CHECK-NOT: OpUnordered
; CHECK: [[A:%[0-9]+]] = OpFUnordNotEqual %{{[0-9]+}} [[X:%[0-9]+]] [[X]]
; CHECK: [[B:%[0-9]+]] = OpFUnordNotEqual %{{[0-9]+}} [[Y:%[0-9]+]] [[Y]]
; CHECK: OpLogicalOr %{{[0-9]+}} [[A]] [[B]]
; CHECK: [[C:%[0-9]+]] = OpFOrdEqual %{{[0-9]+}} [[X]] [[X]]
; CHECK: [[D:%[0-9]+]] = OpFOrdEqual %{{[0-9]+}} [[Y]] [[Y]]
; CHECK: OpLogicalAnd %{{[0-9]+}} [[C]] [[D]]

@.str = private constant [4 x i8] c"buf\00"
define void @main() #0 {
  %h = call target("spirv.VulkanBuffer", [0 x float], 12, 1) @llvm.spv.resource.handlefrombinding(i32 0, i32 0, i32 1, i32 0, ptr @.str)
  %p = call ptr addrspace(11) @llvm.spv.resource.getpointer(target("spirv.VulkanBuffer", [0 x float], 12, 1) %h, i32 0)
  %q = call ptr addrspace(11) @llvm.spv.resource.getpointer(target("spirv.VulkanBuffer", [0 x float], 12, 1) %h, i32 1)
  %x = load float, ptr addrspace(11) %p
  %y = load float, ptr addrspace(11) %q
  %uno = fcmp uno float %x, %y
  %ord = fcmp ord float %x, %y
  %both = and i1 %uno, %ord
  %r = select i1 %both, float 1.0, float 0.0
  store float %r, ptr addrspace(11) %p
  ret void
}
attributes #0 = { "hlsl.numthreads"="1,1,1" "hlsl.shader"="compute" }
