; RUN: opt -mtriple=spirv64-unknown-chipstar-shadervulkan -passes=hipspv-lower-to-hlsl-shape -S %s | FileCheck %s
; RUN: opt -mtriple=spirv64-unknown-chipstar-shadervulkan -passes=hipspv-lower-to-hlsl-shape %s -o %t.bc
; RUN: llc -O0 -mtriple=spirv64-unknown-chipstar-shadervulkan %t.bc -o - | FileCheck %s --check-prefix=SPV
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-chipstar-shadervulkan %t.bc -o - -filetype=obj | spirv-val --target-env vulkan1.3 %}

; Pointer arguments become storage buffers; every argument is a field of a
; push constant block, a pointer its byte offset into its buffer.

; CHECK: %vadd.pc = type { i64, i64, i64, float }
; CHECK: @vadd.pc = external hidden addrspace(13) externally_initialized global %vadd.pc

; CHECK-LABEL: define spir_func void @vadd()
; CHECK-SAME: #[[#ATTR:]] {
; CHECK-DAG: %[[#A:]] = call target("spirv.VulkanBuffer", [0 x float], 12, 1) @llvm.spv.resource.handlefrombinding{{.*}}(i32 0, i32 0, i32 1, i32 0, ptr @.str.vadd.0.in.coerce)
; CHECK-DAG: %[[#B:]] = call target("spirv.VulkanBuffer", [0 x float], 12, 1) @llvm.spv.resource.handlefrombinding{{.*}}(i32 0, i32 1, i32 1, i32 0, ptr @.str.vadd.2.out.coerce)
; CHECK-DAG: %[[#AO:]] = load i64, ptr addrspace(13) @vadd.pc
; CHECK-DAG: %[[#AE:]] = lshr i64 %[[#AO]], 2
; CHECK-DAG: %[[#N:]] = load i64, ptr addrspace(13) getelementptr inbounds nuw (i8, ptr addrspace(13) @vadd.pc, i64 8)
; CHECK-DAG: %[[#S:]] = load float, ptr addrspace(13) getelementptr inbounds nuw (i8, ptr addrspace(13) @vadd.pc, i64 24)
; CHECK: call i32 @llvm.spv.thread.id.in.group.i32(i32 0)
; CHECK-NOT: ashr
; CHECK: %[[#AI:]] = add i64 %[[#AE]], %{{.*}}
; CHECK: %[[#PA:]] = call ptr addrspace(11) @llvm.spv.resource.getpointer.p11.{{.*}}(target("spirv.VulkanBuffer", [0 x float], 12, 1) %[[#A]], i64 %[[#AI]])
; CHECK: load float, ptr addrspace(11) %[[#PA]]
; CHECK: %[[#PB:]] = call ptr addrspace(11) @llvm.spv.resource.getpointer.p11.{{.*}}(target("spirv.VulkanBuffer", [0 x float], 12, 1) %[[#B]], i64 %{{.*}})
; CHECK: store float %{{.*}}, ptr addrspace(11) %[[#PB]]

; CHECK-LABEL: define spir_func void @idx()
; CHECK-SAME: #[[#ATTR]] {
; CHECK: %[[#PO:]] = load i64, ptr addrspace(13) @idx.pc
; CHECK: %[[#PE:]] = lshr i64 %[[#PO]], 2
; CHECK: %[[#I:]] = load i32, ptr addrspace(13) getelementptr inbounds nuw (i8, ptr addrspace(13) @idx.pc, i64 8)
; CHECK: %[[#J:]] = load i64, ptr addrspace(13) getelementptr inbounds nuw (i8, ptr addrspace(13) @idx.pc, i64 16)
; CHECK: %[[#B8:]] = load i8, ptr addrspace(13) getelementptr inbounds nuw (i8, ptr addrspace(13) @idx.pc, i64 24)
; CHECK: %[[#BOOL:]] = trunc i8 %[[#B8]] to i1
; CHECK: %[[#IX:]] = sext i32 %[[#I]] to i64
; CHECK: %[[#IB:]] = mul i64 %[[#IX]], 4
; CHECK: %[[#IM3:]] = add i64 %[[#IB]], -12
; CHECK: %bytes = shl i64 %[[#J]], 2
; CHECK: %[[#B:]] = add i64 %[[#IM3]], %bytes
; CHECK: %f = uitofp i1 %[[#BOOL]] to float
; CHECK: %[[#E:]] = ashr exact i64 %[[#B]], 2
; CHECK: %[[#X:]] = add i64 %[[#PE]], %[[#E]]
; CHECK: %[[#P:]] = call ptr addrspace(11) @llvm.spv.resource.getpointer{{.*}}, i64 %[[#X]])
; CHECK: store float %f, ptr addrspace(11) %[[#P]]

; CHECK-LABEL: define spir_func void @wg()
; CHECK-SAME: #[[#WGATTR:]] !reqd_work_group_size ![[#]] {
; CHECK: %t = add i64 1, 64
; CHECK: store i64 %t, ptr addrspace(11)

; CHECK-LABEL: define spir_func void @gid()
; CHECK: %[[#G:]] = call i32 @llvm.spv.group.id.i32(i32 0)
; CHECK: %[[#G64:]] = zext i32 %[[#G]] to i64
; CHECK: %[[#S:]] = call i32 @_Z20__spirv_SpecConstantii(i32 0, i32 1)
; CHECK: %[[#S64:]] = zext i32 %[[#S]] to i64
; CHECK: %[[#L:]] = call i32 @llvm.spv.thread.id.in.group.i32(i32 0)
; CHECK: %[[#L64:]] = zext i32 %[[#L]] to i64
; CHECK: %[[#M:]] = mul i64 %[[#G64]], %[[#S64]]
; CHECK: add i64 %[[#M]], %[[#L64]]

; CHECK-LABEL: define spir_func void @lsize()
; CHECK: %[[#LS:]] = call i32 @_Z20__spirv_SpecConstantii(i32 1, i32 1)
; CHECK: %[[#LS64:]] = zext i32 %[[#LS]] to i64
; CHECK: store i64 %[[#LS64]], ptr addrspace(11)

; CHECK-LABEL: define spir_func void @wrapping()
; CHECK: %[[#W:]] = add i64 %{{.*}}, -4
; CHECK: %[[#WE:]] = ashr exact i64 %[[#W]], 2
; CHECK: %[[#WI:]] = add i64 %[[#WB:]], %[[#WE]]
; CHECK: call ptr addrspace(11) @llvm.spv.resource.getpointer{{.*}}, i64 %[[#WI]])
; CHECK: call ptr addrspace(11) @llvm.spv.resource.getpointer{{.*}}, i64 %[[#WB]])

; CHECK-LABEL: define spir_func void @byval()
; CHECK: %[[#COPY:]] = alloca %struct.S
; CHECK: %[[#F0:]] = load i32, ptr addrspace(13) @byval.pc
; CHECK: store i32 %[[#F0]], ptr %{{.*}}
; CHECK: load i8, ptr addrspace(13) getelementptr inbounds (i8, ptr addrspace(13) @byval.pc, i64 5)
; CHECK: getelementptr inbounds %struct.S, ptr %[[#COPY]], i32 0, i32 1, i32 1

; CHECK-LABEL: define spir_func void @atomics()
; CHECK: %[[#AP:]] = call ptr addrspace(11) @llvm.spv.resource.getpointer
; CHECK: atomicrmw add ptr addrspace(11) %[[#AP]], i32 1 syncscope("device") monotonic
; CHECK: %[[#CP:]] = call ptr addrspace(11) @llvm.spv.resource.getpointer
; CHECK: cmpxchg ptr addrspace(11) %[[#CP]], i32 0, i32 2 syncscope("device") monotonic monotonic

; CHECK-LABEL: define spir_func void @defined()
; CHECK: call spir_func i64 @_Z12get_group_idj(i32 0)

; A pointer phi or select becomes one of byte offsets.
; CHECK-LABEL: define spir_func void @loop_phi()
; CHECK: %[[#O:]] = phi i64 [ 0, %entry ], [ %[[#N:]], %loop ]
; CHECK: ashr exact i64 %[[#O]], 2
; CHECK: %[[#N]] = add i64 %[[#O]], 4

; Pointers compare by offset; a null argument has offset 1 << 63.
; CHECK-LABEL: define spir_func void @compare()
; CHECK: %[[#S:]] = select i1 %{{.*}}, i64 4, i64 0
; CHECK: icmp eq i64 %[[#S]], 4
; CHECK: icmp ne i64 %{{.*}}, -9223372036854775808

; Each access type gets its own view of the buffer.
; CHECK-LABEL: define spir_func void @mixed()
; CHECK-DAG: call target("spirv.VulkanBuffer", [0 x float], 12, 1) @llvm.spv.resource.handlefrombinding{{.*}}(i32 0, i32 0, i32 1, i32 0, ptr @.str.mixed.0.p)
; CHECK-DAG: call target("spirv.VulkanBuffer", [0 x i32], 12, 1) @llvm.spv.resource.handlefrombinding{{.*}}(i32 0, i32 0, i32 1, i32 0, ptr @".str.mixed.0~1.p")

; An offset not provably a whole number of elements is one if the access is
; aligned: the index is then that of the address in the buffer.
; Over 128 bytes of arguments go in the read-only storage buffer
; "<kernel>.args", bound after the pointer arguments.
; CHECK-LABEL: define spir_func void @big()
; CHECK: %[[#H:]] = call target("spirv.VulkanBuffer", [0 x %big.pc], 12, 0) @llvm.spv.resource.handlefrombinding{{.*}}(i32 0, i32 1, i32 1, i32 0, ptr @.str.big.args)
; CHECK: %[[#B:]] = call ptr addrspace(11) @llvm.spv.resource.getpointer{{.*}}(target("spirv.VulkanBuffer", [0 x %big.pc], 12, 0) %[[#H]], i64 0)
; CHECK: getelementptr inbounds nuw %big.pc, ptr addrspace(11) %[[#B]], i32 0, i32 1

; CHECK-LABEL: define spir_func void @aligned()
; CHECK: %[[#B:]] = load i64, ptr addrspace(13) @aligned.pc
; CHECK: %[[#A:]] = add i64 %[[#B]], %{{.*}}
; CHECK: lshr i64 %[[#A]], 2

; CHECK: attributes #[[#ATTR]] = { "hlsl.shader"="compute" "spirv.workgroup-size-spec-ids"="0,1,2" }
; CHECK: attributes #[[#WGATTR]] = { "hlsl.shader"="compute" }

; SPV-DAG: OpEntryPoint GLCompute %[[#]] "vadd"
; SPV-DAG: OpExecutionModeId %[[#]] LocalSizeId %[[#]] %[[#]] %[[#]]
; SPV-DAG: OpExecutionMode %[[#]] LocalSize 64 1 1
; SPV-DAG: OpExtension "SPV_KHR_float_controls"
; SPV-DAG: OpExecutionMode %[[#]] SignedZeroInfNanPreserve 32
; SPV-DAG: OpExecutionMode %[[#VADD:]] DenormPreserve 32
; SPV-DAG: OpExecutionMode %[[#]] DenormFlushToZero 32
; SPV-NOT: LocalSizeHint
; SPV-DAG: OpDecorate %[[#]] DescriptorSet 0
; SPV-DAG: OpDecorate %[[#]] Binding 1
; SPV: OpQuantizeToF16

; vadd(const float *in, long n, float *out, float s): out[i] = in[i] * s
define spir_kernel void @vadd(ptr addrspace(1) %in.coerce, i64 %n, ptr addrspace(1) %out.coerce, float %s) {
entry:
  %0 = ptrtoint ptr addrspace(1) %in.coerce to i64
  %in = inttoptr i64 %0 to ptr addrspace(4)
  %1 = ptrtoint ptr addrspace(1) %out.coerce to i64
  %out = inttoptr i64 %1 to ptr addrspace(4)
  %lid = call spir_func i64 @_Z12get_local_idj(i32 0)
  %cmp = icmp slt i64 %lid, %n
  br i1 %cmp, label %body, label %exit

body:
  %pin = getelementptr inbounds [4 x i8], ptr addrspace(4) %in, i64 %lid
  %v = load float, ptr addrspace(4) %pin, align 4
  %m = fmul float %v, %s
  %pout = getelementptr inbounds [4 x i8], ptr addrspace(4) %out, i64 %lid
  store float %m, ptr addrspace(4) %pout, align 4
  br label %exit

exit:
  ret void
}

declare spir_func i64 @_Z12get_local_idj(i32)

; idx(float *p, int i, long j, bool b): p[i - 3 + j] = b, the last offset in bytes.
define spir_kernel void @idx(ptr addrspace(1) %p, i32 %i, i64 %j, i1 %b) memory(argmem: write) {
entry:
  %g1 = getelementptr inbounds float, ptr addrspace(1) %p, i32 %i
  %g2 = getelementptr inbounds float, ptr addrspace(1) %g1, i64 -3
  %bytes = shl i64 %j, 2
  %g3 = getelementptr inbounds i8, ptr addrspace(1) %g2, i64 %bytes
  %f = uitofp i1 %b to float
  store float %f, ptr addrspace(1) %g3, align 4
  ret void
}

; A required work-group size replaces the spec constants and is the value of
; get_local_size, other kernel metadata is dropped, and dimensions other than
; 0, 1 and 2 fold to the OpenCL default.
define spir_kernel void @wg(ptr addrspace(1) %p) !reqd_work_group_size !0 !work_group_size_hint !0 {
entry:
  %n = call spir_func i64 @_Z14get_num_groupsj(i32 3)
  %s = call spir_func i64 @_Z14get_local_sizej(i32 0)
  %t = add i64 %n, %s
  store i64 %t, ptr addrspace(1) %p, align 8
  ret void
}

; get_global_id is computed in 64 bits: GlobalInvocationId is 32-bit.
define spir_kernel void @gid(ptr addrspace(1) %p) {
entry:
  %g = call spir_func i64 @_Z13get_global_idj(i32 0)
  store i64 %g, ptr addrspace(1) %p, align 8
  ret void
}

declare spir_func i64 @_Z13get_global_idj(i32)

; Otherwise get_local_size reads the spec constant of its dimension.
define spir_kernel void @lsize(ptr addrspace(1) %p) {
entry:
  %s = call spir_func i64 @_Z14get_local_sizej(i32 1)
  store i64 %s, ptr addrspace(1) %p, align 8
  ret void
}

declare spir_func i64 @_Z14get_num_groupsj(i32)
declare spir_func i64 @_Z14get_local_sizej(i32)

!0 = !{i32 64, i32 1, i32 1}

; Builtins the backend lowers to instructions are not undefined functions.
define spir_kernel void @builtin(ptr addrspace(1) %p, float %x) {
entry:
  %q = call spir_func float @__spirv_QuantizeToF16(float %x)
  store float %q, ptr addrspace(1) %p, align 4
  ret void
}

declare spir_func float @__spirv_QuantizeToF16(float)

; Offsets add up in wrapping arithmetic: clang -O2 emits p[i - 1] without
; inbounds, and two 2^63-byte steps return to element 0.
define spir_kernel void @wrapping(ptr addrspace(1) %p, i64 %i) {
entry:
  %a = getelementptr [4 x i8], ptr addrspace(1) %p, i64 %i
  %b = getelementptr i8, ptr addrspace(1) %a, i64 -4
  store float 0.0, ptr addrspace(1) %b, align 4
  %c = getelementptr float, ptr addrspace(1) %p, i64 2305843009213693952
  %d = getelementptr float, ptr addrspace(1) %c, i64 2305843009213693952
  store float 1.0, ptr addrspace(1) %d, align 4
  ret void
}

; A struct passed by value is a push constant field, copied to a local the
; kernel may write.
%struct.S = type { i32, [2 x i8] }
define spir_kernel void @byval(ptr byval(%struct.S) %s, ptr addrspace(1) %p) {
entry:
  %f = getelementptr inbounds %struct.S, ptr %s, i32 0, i32 1, i32 1
  store i8 7, ptr %f, align 1
  %v = load i32, ptr %s, align 4
  store i32 %v, ptr addrspace(1) %p, align 4
  ret void
}

; allocsize names a parameter the entry point no longer has.
define spir_kernel void @alloc_size(ptr addrspace(1) %p, i32 %n) allocsize(1) {
entry:
  store i32 %n, ptr addrspace(1) %p, align 4
  ret void
}

; Atomic accesses go through the buffer like loads and stores.
define spir_kernel void @atomics(ptr addrspace(1) %p) {
entry:
  %old = atomicrmw add ptr addrspace(1) %p, i32 1 syncscope("device") monotonic, align 4
  %q = getelementptr inbounds i32, ptr addrspace(1) %p, i64 1
  %x = cmpxchg ptr addrspace(1) %q, i32 0, i32 2 syncscope("device") monotonic monotonic, align 4
  ret void
}

; A definition with a builtin's name is not the builtin.
define spir_kernel void @defined(ptr addrspace(1) %p) {
entry:
  %g = call spir_func i64 @_Z12get_group_idj(i32 0)
  store i64 %g, ptr addrspace(1) %p, align 8
  ret void
}

define internal spir_func i64 @_Z12get_group_idj(i32 %d) {
  ret i64 42
}

define spir_kernel void @loop_phi(ptr addrspace(1) %p, i64 %n) {
entry:
  br label %loop

loop:
  %q = phi ptr addrspace(1) [ %p, %entry ], [ %next, %loop ]
  %i = phi i64 [ 0, %entry ], [ %i.next, %loop ]
  store i32 0, ptr addrspace(1) %q, align 4
  %next = getelementptr inbounds i32, ptr addrspace(1) %q, i64 1
  %i.next = add i64 %i, 1
  %done = icmp eq i64 %i.next, %n
  br i1 %done, label %exit, label %loop

exit:
  ret void
}

define spir_kernel void @compare(ptr addrspace(1) %p, ptr addrspace(1) %out, i1 %c) {
  %a = getelementptr inbounds i32, ptr addrspace(1) %p, i64 1
  %b = select i1 %c, ptr addrspace(1) %a, ptr addrspace(1) %p
  %same = icmp eq ptr addrspace(1) %b, %a
  %nonnull = icmp ne ptr addrspace(1) %p, null
  %both = and i1 %same, %nonnull
  %v = zext i1 %both to i32
  store i32 %v, ptr addrspace(1) %out, align 4
  ret void
}

define spir_kernel void @mixed(ptr addrspace(1) %p) {
  store i32 1, ptr addrspace(1) %p, align 4
  %f = getelementptr inbounds i8, ptr addrspace(1) %p, i64 4
  store float 2.0, ptr addrspace(1) %f, align 4
  ret void
}

%struct.Big = type { [40 x i32] }
define spir_kernel void @big(ptr addrspace(1) %out, ptr byval(%struct.Big) align 4 %s) {
  %g = getelementptr inbounds i8, ptr %s, i64 156
  %v = load i32, ptr %g, align 4
  store i32 %v, ptr addrspace(1) %out, align 4
  ret void
}

define spir_kernel void @aligned(ptr addrspace(1) %p, i64 %j) {
  %g = getelementptr inbounds i8, ptr addrspace(1) %p, i64 %j
  store float 0.0, ptr addrspace(1) %g, align 4
  ret void
}

; A kernel built to flush float denormals keeps flushing them.
define spir_kernel void @ftz(ptr addrspace(1) %p, float %x) denormal_fpenv(float: preservesign) {
entry:
  %y = fmul float %x, %x
  store float %y, ptr addrspace(1) %p, align 4
  ret void
}
