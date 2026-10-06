; A long array in a by-value argument is copied to the kernel's local by a
; loop, not one load and store per element.
; RUN: opt -mtriple=spirv64-unknown-chipstar-shadervulkan -passes=hipspv-lower-to-hlsl-shape -S %s | FileCheck %s
; RUN: llc -O0 -mtriple=spirv64-unknown-chipstar-shadervulkan %s -o - | FileCheck %s --check-prefix=SPV
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-chipstar-shadervulkan %s -o - -filetype=obj | spirv-val --target-env vulkan1.3 %}
; Short arrays stay unrolled, which stays valid when optimized.
; RUN: %if spirv-tools %{ llc -O2 -mtriple=spirv64-unknown-chipstar-shadervulkan %S/Inputs/byval-short-array.ll -o - -filetype=obj | spirv-val --target-env vulkan1.3 %}

; CHECK-LABEL: define spir_func void @large()
; CHECK: [[IV:%.+]] = phi i32 [ 0, %{{.*}} ], [ [[NEXT:%.+]], %{{.*}} ]
; CHECK: [[S:%.+]] = getelementptr inbounds %struct.Large, ptr addrspace(11) %{{.*}}, i32 0, i32 0, i32 [[IV]]
; CHECK: [[V:%.+]] = load i8, ptr addrspace(11) [[S]]
; CHECK: [[D:%.+]] = getelementptr inbounds %struct.Large, ptr %{{.*}}, i32 0, i32 0, i32 [[IV]]
; CHECK: store i8 [[V]], ptr [[D]]
; CHECK: [[NEXT]] = add nuw nsw i32 [[IV]], 1
; CHECK: icmp eq i32 [[NEXT]], 4080
; CHECK-NOT: load i8, ptr addrspace(11)
; CHECK: ret void

; Nested arrays loop over every level holding more than 128 leaves.
; CHECK-LABEL: define spir_func void @cube()
; CHECK-COUNT-2: icmp eq i32 {{.*}}, 16
; CHECK-NOT: icmp eq
; CHECK: ret void

; SPV-COUNT-3: OpLoopMerge
; SPV-NOT: OpLoopMerge

%struct.Large = type { [4080 x i8] }
define spir_kernel void @large(ptr byval(%struct.Large) align 1 %s, ptr addrspace(1) %out, i32 %i, i32 %j) {
  %p = getelementptr inbounds %struct.Large, ptr %s, i32 0, i32 0, i32 %i
  %v = load i8, ptr %p, align 1
  %w = add i8 %v, 1
  store i8 %w, ptr %p, align 1
  %q = getelementptr inbounds %struct.Large, ptr %s, i32 0, i32 0, i32 %j
  %x = load i8, ptr %q, align 1
  store i8 %x, ptr addrspace(1) %out, align 1
  ret void
}

%struct.Cube = type { [16 x [16 x [16 x i8]]] }
define spir_kernel void @cube(ptr byval(%struct.Cube) align 1 %s, ptr addrspace(1) %out, i32 %i) {
  %p = getelementptr inbounds %struct.Cube, ptr %s, i32 0, i32 0, i32 %i, i32 %i, i32 %i
  %v = load i8, ptr %p, align 1
  store i8 %v, ptr addrspace(1) %out, align 1
  ret void
}
