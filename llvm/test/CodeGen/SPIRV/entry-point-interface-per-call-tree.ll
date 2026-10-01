; Each Vulkan entry point lists only the variables its static call tree uses;
; listing the other entry point's push constant would give it two. Calls
; through an alias count too, and a global no function uses belongs to no
; entry point.
; RUN: llc -O0 -mtriple=spirv1.6-unknown-vulkan1.3-compute %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv1.6-unknown-vulkan1.3-compute %s -o - -filetype=obj | spirv-val --target-env vulkan1.3 %}

; CHECK: OpEntryPoint GLCompute %[[#]] "a" %[[#APC:]] %[[#BUF:]]{{$}}
; CHECK: OpEntryPoint GLCompute %[[#]] "b" %[[#BPC:]] %[[#BUF]]{{$}}
; CHECK-DAG: OpName %[[#APC]] "a.pc{{.*}}"
; CHECK-DAG: OpName %[[#BPC]] "b.pc{{.*}}"
; CHECK-DAG: OpName %[[#BUF]] "buf"

%pc = type { i32 }

@a.pc = external hidden addrspace(13) externally_initialized global %pc
@b.pc = external hidden addrspace(13) externally_initialized global %pc
@unused.pc = external hidden addrspace(13) externally_initialized global %pc
@.str = private constant [4 x i8] c"buf\00"
@store_alias = internal alias void (i32), ptr @store

define void @a() #0 {
  %v = load i32, ptr addrspace(13) @a.pc
  call void @store(i32 %v)
  ret void
}

define void @b() #0 {
  %v = load i32, ptr addrspace(13) @b.pc
  call void @store_alias(i32 %v)
  ret void
}

define internal void @store(i32 %v) {
  %h = call target("spirv.VulkanBuffer", [0 x i32], 12, 1) @llvm.spv.resource.handlefrombinding(i32 0, i32 0, i32 1, i32 0, ptr @.str)
  %p = call ptr addrspace(11) @llvm.spv.resource.getpointer(target("spirv.VulkanBuffer", [0 x i32], 12, 1) %h, i32 0)
  store i32 %v, ptr addrspace(11) %p
  ret void
}

attributes #0 = { "hlsl.numthreads"="1,1,1" "hlsl.shader"="compute" }
