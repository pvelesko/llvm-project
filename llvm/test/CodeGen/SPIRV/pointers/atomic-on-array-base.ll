; An atomic on a workgroup array's address accesses its first element, and one
; on a constant byte offset into it accesses the element at that offset.
; RUN: llc -O0 -mtriple=spirv1.6-unknown-vulkan1.3-compute %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv1.6-unknown-vulkan1.3-compute %s -o - -filetype=obj | spirv-val --target-env vulkan1.3 %}

; CHECK-DAG: %[[#H:]] = OpVariable %[[#]] Workgroup
; CHECK-DAG: %[[#Zero:]] = OpConstantNull %[[#]]
; CHECK-DAG: %[[#Two:]] = OpConstant %[[#]] 2{{$}}
; CHECK: %[[#P0:]] = OpInBoundsAccessChain %[[#]] %[[#H]] %[[#Zero]]
; CHECK: OpAtomicIAdd %[[#]] %[[#P0]]
; CHECK: %[[#P2:]] = OpInBoundsAccessChain %[[#]] %[[#H]] %[[#Two]]
; CHECK: OpAtomicIAdd %[[#]] %[[#P2]]

@h = internal addrspace(3) global [14 x i32] poison, align 4

define void @main() #0 {
  %a = atomicrmw add ptr addrspace(3) @h, i32 1 syncscope("workgroup") monotonic, align 4
  %b = atomicrmw add ptr addrspace(3) getelementptr inbounds (i8, ptr addrspace(3) @h, i64 8), i32 1 syncscope("workgroup") monotonic, align 4
  ret void
}

attributes #0 = { "hlsl.numthreads"="1,1,1" "hlsl.shader"="compute" }
