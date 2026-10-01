; A GEP into an array of bytes with more than one index is an element access,
; not byte addressing.
; RUN: llc -O0 -mtriple=spirv1.6-unknown-vulkan1.3-compute %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv1.6-unknown-vulkan1.3-compute %s -o - -filetype=obj | spirv-val --target-env vulkan1.3 %}

; CHECK: %[[#S:]] = OpInBoundsAccessChain %[[#]] %[[#]] %[[#]]
; CHECK: OpInBoundsAccessChain %[[#]] %[[#S]] %[[#]]

%S = type { i32, [3 x i8] }

define void @main() #0 {
  %s = alloca %S, align 4
  %d = getelementptr inbounds %S, ptr %s, i32 0, i32 1
  %t = call i32 @llvm.spv.thread.id.in.group.i32(i32 0)
  %i = zext i32 %t to i64
  %a = getelementptr inbounds [3 x i8], ptr %d, i64 0, i64 %i
  store i8 1, ptr %a, align 1
  ret void
}

declare i32 @llvm.spv.thread.id.in.group.i32(i32)

attributes #0 = { "hlsl.numthreads"="1,1,1" "hlsl.shader"="compute" }
