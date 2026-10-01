; Identical OpSpecConstants, SpecId included, are one constant; ones that
; differ in SpecId or value stay separate.
; RUN: llc -O0 -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s
; RUN: llc -O0 -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s --check-prefix=COUNT
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; CHECK-DAG: OpDecorate %[[#A:]] SpecId 5
; CHECK-DAG: OpDecorate %[[#B:]] SpecId 6
; CHECK-DAG: OpDecorate %[[#D:]] SpecId 8
; CHECK-DAG: OpDecorate %[[#E:]] SpecId 8
; CHECK-DAG: %[[#A]] = OpSpecConstant %[[#]] 3
; CHECK-DAG: %[[#B]] = OpSpecConstant %[[#]] 3
; CHECK-DAG: %[[#D]] = OpSpecConstant %[[#]] 1
; CHECK-DAG: %[[#E]] = OpSpecConstant %[[#]] 2
; COUNT-COUNT-4: SpecId
; COUNT-NOT: SpecId

define spir_func i32 @f() {
  %a = call i32 @_Z20__spirv_SpecConstantii(i32 5, i32 3)
  %b = call i32 @_Z20__spirv_SpecConstantii(i32 6, i32 3)
  %d = call i32 @_Z20__spirv_SpecConstantii(i32 8, i32 1)
  %x = add i32 %a, %b
  %y = add i32 %x, %d
  ret i32 %y
}

define spir_func i32 @g() {
  %a = call i32 @_Z20__spirv_SpecConstantii(i32 5, i32 3)
  %e = call i32 @_Z20__spirv_SpecConstantii(i32 8, i32 2)
  %y = add i32 %a, %e
  ret i32 %y
}

declare i32 @_Z20__spirv_SpecConstantii(i32, i32)
