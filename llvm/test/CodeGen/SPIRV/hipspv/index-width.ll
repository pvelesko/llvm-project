; Offsets are collected at the index width of the pointer, here 32 bits.
; RUN: opt -mtriple=spirv32-unknown-unknown -passes=hipspv-lower-to-hlsl-shape -S %s | FileCheck %s
; A derived pointer with another index width wraps differently.
; RUN: not opt -mtriple=spirv64-unknown-chipstar-shadervulkan --data-layout=e-p1:64:64:64:64-p4:64:64:64:32 -passes=hipspv-lower-to-hlsl-shape -disable-output %s 2>&1 | FileCheck %s --check-prefix=MIXED
; MIXED: in function generic {{.*}}: unsupported use of pointer argument 0

; CHECK: %[[#E:]] = ashr exact i32 %{{.*}}, 2
; CHECK: %[[#I:]] = sext i32 %[[#E]] to i64
; CHECK: %[[#X:]] = add i64 %{{.*}}, %[[#I]]
; CHECK: call ptr addrspace(11) @llvm.spv.resource.getpointer{{.*}}, i64 %[[#X]])

define spir_kernel void @k(ptr addrspace(1) %p, i32 %i) {
  %g = getelementptr inbounds float, ptr addrspace(1) %p, i32 %i
  store float 0.0, ptr addrspace(1) %g, align 4
  ret void
}

define spir_kernel void @generic(ptr addrspace(1) %p, i32 %i) {
  %q = addrspacecast ptr addrspace(1) %p to ptr addrspace(4)
  %g = getelementptr inbounds float, ptr addrspace(4) %q, i32 %i
  store float 0.0, ptr addrspace(4) %g, align 4
  ret void
}
