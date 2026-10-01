; The storage capability also covers 16-bit data that only appears nested in
; a struct and a vector of the interface type, including a long vector.
; RUN: llc -O0 -mtriple=spirv1.6-unknown-vulkan1.3-compute %s -o - | FileCheck %s
; RUN: sed 's/<2 x i16>/<1 x i16>/' %s | llc -O0 -mtriple=spirv1.6-unknown-vulkan1.3-compute -spirv-ext=+SPV_EXT_long_vector -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv1.6-unknown-vulkan1.3-compute %s -o - -filetype=obj | spirv-val --target-env vulkan1.3 %}

; CHECK: OpCapability StoragePushConstant16

@pc = external hidden addrspace(13) externally_initialized global { i32, <2 x i16> }
@.str = private constant [4 x i8] c"buf\00"

define void @main() #0 {
  %v = load i32, ptr addrspace(13) @pc
  %h = call target("spirv.VulkanBuffer", [0 x i32], 12, 1) @llvm.spv.resource.handlefrombinding(i32 0, i32 0, i32 1, i32 0, ptr @.str)
  %p = call ptr addrspace(11) @llvm.spv.resource.getpointer(target("spirv.VulkanBuffer", [0 x i32], 12, 1) %h, i32 0)
  store i32 %v, ptr addrspace(11) %p
  ret void
}

attributes #0 = { "hlsl.numthreads"="1,1,1" "hlsl.shader"="compute" }
