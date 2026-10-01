; llvm.lrint and llvm.llrint round in the current rounding mode, then convert.
; RUN: llc -O0 -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; CHECK: OpExtInst %[[#]] %[[#]] rint
; CHECK: OpConvertFToS
; CHECK: OpExtInst %[[#]] %[[#]] rint
; CHECK: OpConvertFToS

define i32 @lrint(float %x) {
  %r = call i32 @llvm.lrint.i32.f32(float %x)
  ret i32 %r
}

define i64 @llrint(double %x) {
  %r = call i64 @llvm.llrint.i64.f64(double %x)
  ret i64 %r
}
