; 8- and 16-bit data in StorageBuffer, Uniform or PushConstant storage needs
; the matching storage capability where the target makes it available: core
; from SPIR-V 1.3 (16-bit) and 1.5 (8-bit), or through the extension.
; RUN: llc -O0 -mtriple=spirv1.6-unknown-vulkan1.3-compute %s -o - | FileCheck %s --check-prefixes=CHECK,ALL8,CORE
; RUN: llc -O0 -mtriple=spirv-unknown-vulkan-compute -spirv-ext=+SPV_KHR_8bit_storage %s -o - | FileCheck %s --check-prefixes=CHECK,ALL8,EXT
; RUN: llc -O0 -mtriple=spirv-unknown-vulkan-compute %s -o - | FileCheck %s --check-prefix=NO8
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv1.6-unknown-vulkan1.3-compute %s -o - -filetype=obj | spirv-val --target-env vulkan1.3 %}

; CHECK-DAG: OpCapability StorageBuffer16BitAccess
; CHECK-DAG: OpCapability StorageUniform16
; CHECK-DAG: OpCapability StoragePushConstant16
; ALL8-DAG: OpCapability StorageBuffer8BitAccess
; ALL8-DAG: OpCapability UniformAndStorageBuffer8BitAccess
; ALL8-DAG: OpCapability StoragePushConstant8
; CORE-NOT: OpExtension
; EXT: OpExtension "SPV_KHR_8bit_storage"
; NO8-NOT: 8BitAccess
; NO8-NOT: StoragePushConstant8

%inner = type { half }
%S = type { i32, %inner, <4 x i16> }

@pc = external hidden addrspace(13) externally_initialized global { i16, i8 }
%U = type { half, i8, i8, i32, i32, i32 }
@.bytes = private constant [6 x i8] c"bytes\00"
@.structs = private constant [8 x i8] c"structs\00"
@.ubo = private constant [4 x i8] c"ubo\00"

define void @main() #0 {
  %w = load i16, ptr addrspace(13) @pc
  %pc8p = getelementptr inbounds i8, ptr addrspace(13) @pc, i32 2
  %v = load i8, ptr addrspace(13) %pc8p
  %hu = call target("spirv.VulkanBuffer", [1 x %U], 2, 0) @llvm.spv.resource.handlefrombinding(i32 0, i32 2, i32 1, i32 0, ptr @.ubo)
  %pu = call ptr addrspace(12) @llvm.spv.resource.getpointer(target("spirv.VulkanBuffer", [1 x %U], 2, 0) %hu, i32 0)
  %c = load half, ptr addrspace(12) %pu
  %cb8p = getelementptr inbounds i8, ptr addrspace(12) %pu, i32 2
  %c8 = load i8, ptr addrspace(12) %cb8p
  %hb = call target("spirv.VulkanBuffer", [0 x i8], 12, 1) @llvm.spv.resource.handlefrombinding(i32 0, i32 0, i32 1, i32 0, ptr @.bytes)
  %pb = call ptr addrspace(11) @llvm.spv.resource.getpointer(target("spirv.VulkanBuffer", [0 x i8], 12, 1) %hb, i32 0)
  %sum = add i8 %v, %c8
  store i8 %sum, ptr addrspace(11) %pb
  %hs = call target("spirv.VulkanBuffer", [0 x %S], 12, 1) @llvm.spv.resource.handlefrombinding(i32 0, i32 1, i32 1, i32 0, ptr @.structs)
  %ps = call ptr addrspace(11) @llvm.spv.resource.getpointer(target("spirv.VulkanBuffer", [0 x %S], 12, 1) %hs, i32 0)
  %wi = zext i16 %w to i32
  store i32 %wi, ptr addrspace(11) %ps
  %ph = getelementptr inbounds %S, ptr addrspace(11) %ps, i32 0, i32 1, i32 0
  store half %c, ptr addrspace(11) %ph
  ret void
}

attributes #0 = { "hlsl.numthreads"="1,1,1" "hlsl.shader"="compute" }
