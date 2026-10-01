; Shaders lower llvm.lrint and llvm.llrint to RoundEven and a conversion.
; RUN: llc -O0 -mtriple=spirv1.6-unknown-vulkan1.3-compute %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv1.6-unknown-vulkan1.3-compute %s -o - -filetype=obj | spirv-val --target-env vulkan1.3 %}

; CHECK: OpExtInst %[[#]] %[[#]] RoundEven
; CHECK: OpConvertFToS
; CHECK: OpExtInst %[[#]] %[[#]] RoundEven
; CHECK: OpConvertFToS

@.str = private constant [4 x i8] c"buf\00"
define void @main() #0 {
  %h = call target("spirv.VulkanBuffer", [0 x i64], 12, 1) @llvm.spv.resource.handlefrombinding(i32 0, i32 0, i32 1, i32 0, ptr @.str)
  %p = call ptr addrspace(11) @llvm.spv.resource.getpointer(target("spirv.VulkanBuffer", [0 x i64], 12, 1) %h, i32 0)
  %x = load i64, ptr addrspace(11) %p
  %f = sitofp i64 %x to float
  %d = sitofp i64 %x to double
  %a = call i32 @llvm.lrint.i32.f32(float %f)
  %b = call i64 @llvm.llrint.i64.f64(double %d)
  %a64 = sext i32 %a to i64
  %s = add i64 %a64, %b
  store i64 %s, ptr addrspace(11) %p
  ret void
}
attributes #0 = { "hlsl.numthreads"="1,1,1" "hlsl.shader"="compute" }
