; Storage buffers bound for HIP kernels may alias; HLSL resources do not.
; RUN: llc -O0 -mtriple=spirv64-unknown-chipstar-shadervulkan %s -o - | FileCheck %s
; RUN: llc -O0 -mtriple=spirv1.6-unknown-vulkan1.3-compute %s -o - | FileCheck %s --check-prefix=VULKAN
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-chipstar-shadervulkan %s -o - -filetype=obj | spirv-val --target-env vulkan1.3 %}

; CHECK-DAG: OpDecorate %[[#A:]] Binding 0
; CHECK-DAG: OpDecorate %[[#B:]] Binding 1
; CHECK-DAG: OpDecorate %[[#A]] Aliased
; CHECK-DAG: OpDecorate %[[#B]] Aliased

; VULKAN-NOT: Aliased

@.str.a = private constant [2 x i8] c"a\00"
@.str.b = private constant [2 x i8] c"b\00"

define void @k() #0 {
  %a = call target("spirv.VulkanBuffer", [0 x i32], 12, 1) @llvm.spv.resource.handlefrombinding(i32 0, i32 0, i32 1, i32 0, ptr @.str.a)
  %b = call target("spirv.VulkanBuffer", [0 x i32], 12, 1) @llvm.spv.resource.handlefrombinding(i32 0, i32 1, i32 1, i32 0, ptr @.str.b)
  %pa = call ptr addrspace(11) @llvm.spv.resource.getpointer(target("spirv.VulkanBuffer", [0 x i32], 12, 1) %a, i32 0)
  %pb = call ptr addrspace(11) @llvm.spv.resource.getpointer(target("spirv.VulkanBuffer", [0 x i32], 12, 1) %b, i32 0)
  store i32 1, ptr addrspace(11) %pb
  %v = load i32, ptr addrspace(11) %pa
  store i32 %v, ptr addrspace(11) %pb
  ret void
}

attributes #0 = { "hlsl.numthreads"="1,1,1" "hlsl.shader"="compute" }
