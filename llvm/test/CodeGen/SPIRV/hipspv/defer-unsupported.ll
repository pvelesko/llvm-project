; With -hipspv-defer-unsupported, a kernel the pass cannot lower is a warning
; and an entry point carrying the message, for the runtime to report.
; RUN: llc -O0 -mtriple=spirv64-unknown-chipstar-shadervulkan -hipspv-defer-unsupported %s -o - 2>&1 | FileCheck %s
; RUN: not llc -O0 -mtriple=spirv64-unknown-chipstar-shadervulkan %s -o /dev/null 2>&1 | FileCheck %s --check-prefix=ERR

; CHECK: warning: {{.*}} in function k {{.*}}: call to undefined function f
; CHECK-DAG: OpEntryPoint GLCompute %[[#]] "k"
; CHECK-DAG: OpEntryPoint GLCompute %[[#]] "__hipspv_error: in function k: call to undefined function f"
; ERR: error: {{.*}} in function k {{.*}}: call to undefined function f

declare spir_func void @f()

define spir_kernel void @k() {
  call spir_func void @f()
  ret void
}
