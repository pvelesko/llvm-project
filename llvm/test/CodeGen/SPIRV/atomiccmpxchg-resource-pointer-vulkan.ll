; cmpxchg on a storage buffer element: the pointee type is the value type, not
; the {value, success} result pair, so no pointer cast is needed.
; RUN: llc -O0 -mtriple=spirv1.6-unknown-vulkan1.3-compute %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv1.6-unknown-vulkan1.3-compute %s -o - -filetype=obj | spirv-val --target-env vulkan1.3 %}

; CHECK: OpAtomicCompareExchange %[[#]] %[[#]]

@.str = private constant [4 x i8] c"buf\00"

define void @main() #0 {
  %h = call target("spirv.VulkanBuffer", [0 x i32], 12, 1) @llvm.spv.resource.handlefrombinding(i32 0, i32 0, i32 1, i32 0, ptr @.str)
  %p = call ptr addrspace(11) @llvm.spv.resource.getpointer(target("spirv.VulkanBuffer", [0 x i32], 12, 1) %h, i32 0)
  %x = cmpxchg ptr addrspace(11) %p, i32 0, i32 1 syncscope("device") monotonic monotonic, align 4
  %v = extractvalue { i32, i1 } %x, 0
  store i32 %v, ptr addrspace(11) %p
  ret void
}

attributes #0 = { "hlsl.numthreads"="1,1,1" "hlsl.shader"="compute" }
