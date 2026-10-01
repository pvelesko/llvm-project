; Kernels the pass cannot express are diagnosed; the module stays valid.
; RUN: not opt -mtriple=spirv64-unknown-chipstar-shadervulkan -passes=hipspv-lower-to-hlsl-shape,verify -S %s 2>&1 | FileCheck %s

; CHECK-DAG: error: {{.*}} in function self_store {{.*}}: unsupported use of pointer argument 0
; CHECK-DAG: error: {{.*}} in function load_ptr {{.*}}: unsupported use of pointer argument 0
; CHECK-DAG: error: {{.*}} in function local_arg {{.*}}: unsupported argument 0
; CHECK-DAG: error: {{.*}} in function bool_vec {{.*}}: unsupported argument 0
; CHECK-DAG: error: {{.*}} in function bool_buf {{.*}}: unsupported use of pointer argument 0
; CHECK-DAG: error: {{.*}} in function scalable {{.*}}: unsupported use of pointer argument 0
; CHECK-DAG: error: {{.*}} in function i24_buf {{.*}}: unsupported use of pointer argument 0
; CHECK-DAG: error: {{.*}} in function global_use {{.*}}: unsupported global or generic pointer in @g
; CHECK-DAG: error: {{.*}} in function private_global {{.*}}: unsupported global or generic pointer
; CHECK-DAG: error: {{.*}} in function dynamic_shared {{.*}}: unsupported global or generic pointer
; CHECK-DAG: error: {{.*}} in function exec_mode {{.*}}: unsupported spirv.ExecutionMode
; CHECK-DAG: error: {{.*}} in function recursive {{.*}}: recursive call to fact
; CHECK-DAG: error: {{.*}} in function indirect {{.*}}: indirect call
; CHECK-DAG: error: {{.*}} in function alias_call {{.*}}: unsupported global or generic pointer
; CHECK-DAG: error: {{.*}} in function aggregate {{.*}}: unsupported global or generic pointer
; CHECK-DAG: error: {{.*}} in function ext_builtin {{.*}}: call to undefined function _Z3Sinf
; CHECK-DAG: error: {{.*}} in function inline_asm {{.*}}: inline assembly
; CHECK-DAG: error: {{.*}} in function shared_type {{.*}}: unsupported global or generic pointer
; CHECK-DAG: error: {{.*}} in function shared_alias {{.*}}: unsupported global or generic pointer
; CHECK-DAG: error: {{.*}} in function calls_kernel {{.*}}: call to kernel early_kernel
; CHECK-DAG: error: {{.*}} in function calls_late_kernel {{.*}}: call to kernel late_kernel
; CHECK-DAG: error: {{.*}} in function byval {{.*}}: unsupported argument 0
; CHECK-DAG: error: {{.*}} in function ptr_select {{.*}}: unsupported use of pointer argument 0
; CHECK-DAG: error: {{.*}} in function alloca_type {{.*}}: unsupported global or generic pointer
; CHECK-DAG: error: {{.*}} in function aggregate_value {{.*}}: unsupported global or generic pointer
; CHECK-DAG: error: {{.*}} in function no_dim {{.*}}: call to undefined function _Z12get_group_idj
; CHECK-DAG: error: {{.*}} in function var_dim {{.*}}: call to undefined function _Z13get_global_idj
; CHECK-DAG: error: {{.*}} in function callee_global {{.*}}: unsupported global or generic pointer
; CHECK-DAG: error: {{.*}} in function cexpr_global {{.*}}: unsupported global or generic pointer
; CHECK-DAG: error: {{.*}} in function wide_vec {{.*}}: unsupported use of pointer argument 0
; CHECK-DAG: error: {{.*}} in function wide_dim {{.*}}: call to undefined function _Z14get_num_groupsj
; CHECK-DAG: error: {{.*}} in function misaligned {{.*}}: unsupported use of pointer argument 0 by store
; CHECK-DAG: error: {{.*}} in function unknown_align {{.*}}: unsupported use of pointer argument 0 by store

; CHECK: @llvm.used = appending global [1 x ptr] [ptr @self_store]
; CHECK: @alias = available_externally alias void (ptr addrspace(3)), ptr @local_arg
; CHECK: define spir_func void @self_store()
; CHECK: define spir_func void @load_ptr()
; CHECK: define available_externally spir_func void @local_arg()

@llvm.used = appending global [1 x ptr] [ptr @self_store], section "llvm.metadata"
@alias = available_externally alias void (ptr addrspace(3)), ptr @local_arg
@g = addrspace(1) global i32 0
@u = internal global i32 0
@dyn = external addrspace(3) global [0 x float]
@shared = internal addrspace(3) global { i32, ptr addrspace(4) } poison
@shared_a = internal alias { i32, ptr addrspace(4) }, ptr addrspace(3) @shared
@read_g_alias = internal alias i32 (), ptr @read_g

define spir_kernel void @early_kernel(i32 %x) {
  ret void
}

define spir_kernel void @self_store(ptr addrspace(1) %p) {
  store ptr addrspace(1) %p, ptr addrspace(1) %p, align 8
  ret void
}

define spir_kernel void @load_ptr(ptr addrspace(1) %pp) {
  %p = load ptr addrspace(1), ptr addrspace(1) %pp, align 8
  store i32 1, ptr addrspace(1) %p, align 4
  ret void
}

define available_externally spir_kernel void @local_arg(ptr addrspace(3) %p) {
  store i32 1, ptr addrspace(3) %p, align 4
  ret void
}

define spir_kernel void @bool_vec(<2 x i1> %b) {
  ret void
}

define spir_kernel void @bool_buf(ptr addrspace(1) %p) {
  store i1 true, ptr addrspace(1) %p, align 1
  ret void
}

define spir_kernel void @scalable(ptr addrspace(1) %p) {
  store <vscale x 4 x i32> zeroinitializer, ptr addrspace(1) %p, align 16
  ret void
}

; A shader would store i24 as a wider integer.
define spir_kernel void @i24_buf(ptr addrspace(1) %p) {
  store i24 0, ptr addrspace(1) %p, align 4
  ret void
}

; Dynamic shared memory (extern __shared__ float dyn[]).
define spir_kernel void @dynamic_shared(float %x) {
  store float %x, ptr addrspace(3) @dyn, align 4
  ret void
}

; A mutable addrspace(0) global is not shared between functions.
define spir_kernel void @private_global(i32 %i) {
  store i32 %i, ptr @u, align 4
  ret void
}

define spir_kernel void @global_use(ptr addrspace(1) %p) {
  %v = load i32, ptr addrspace(1) @g, align 4
  store i32 %v, ptr addrspace(1) %p, align 4
  ret void
}

define spir_kernel void @callee_global(ptr addrspace(1) %p) {
  %v = call spir_func i32 @read_g()
  store i32 %v, ptr addrspace(1) %p, align 4
  ret void
}

define internal spir_func i32 @read_g() noinline {
  %v = load i32, ptr addrspace(1) @g, align 4
  ret i32 %v
}

define spir_kernel void @cexpr_global(ptr addrspace(1) %p) {
  store i64 ptrtoint (ptr addrspace(1) @g to i64), ptr addrspace(1) %p, align 8
  ret void
}

; Vulkan vectors have at most four components.
define spir_kernel void @wide_vec(ptr addrspace(1) %p) {
  store <8 x float> zeroinitializer, ptr addrspace(1) %p, align 32
  ret void
}

; Vulkan forbids recursion.
define spir_kernel void @recursive(ptr addrspace(1) %p, i32 %n) {
  %v = call spir_func i32 @fact(i32 %n)
  store i32 %v, ptr addrspace(1) %p, align 4
  ret void
}

define internal spir_func i32 @fact(i32 %n) {
  %c = icmp eq i32 %n, 0
  br i1 %c, label %done, label %rec
rec:
  %m = sub i32 %n, 1
  %f = call spir_func i32 @fact(i32 %m)
  %r = mul i32 %n, %f
  ret i32 %r
done:
  ret i32 1
}

define spir_kernel void @indirect(ptr addrspace(1) %p, i1 %c) {
  %fp = select i1 %c, ptr @fact, ptr @read_g
  %v = call spir_func i32 %fp(i32 1)
  store i32 %v, ptr addrspace(1) %p, align 4
  ret void
}

define spir_kernel void @alias_call(ptr addrspace(1) %p) {
  %v = call spir_func i32 @read_g_alias()
  store i32 %v, ptr addrspace(1) %p, align 4
  ret void
}

define spir_kernel void @aggregate(ptr addrspace(1) %p) {
  %q = alloca { ptr addrspace(1) }, align 8
  store { ptr addrspace(1) } { ptr addrspace(1) @g }, ptr %q, align 8
  store i32 0, ptr addrspace(1) %p, align 4
  ret void
}

define spir_kernel void @aggregate_value(ptr addrspace(1) %p, ptr addrspace(1) %q) {
  %a = alloca { ptr addrspace(1) }, align 8
  %v = load { ptr addrspace(1) }, ptr %a, align 8
  store i32 0, ptr addrspace(1) %p, align 4
  ret void
}

; The backend lowers extended (OpExtInst) builtins for OpenCL.std only.
define spir_kernel void @ext_builtin(ptr addrspace(1) %p, float %x) {
  %s = call spir_func float @_Z3Sinf(float %x)
  store float %s, ptr addrspace(1) %p, align 4
  ret void
}

declare spir_func float @_Z3Sinf(float)

define spir_kernel void @inline_asm(ptr addrspace(1) %p) {
  %v = call i32 asm "", "=r"()
  store i32 %v, ptr addrspace(1) %p, align 4
  ret void
}

define spir_kernel void @shared_type(ptr addrspace(1) %p) {
  store i32 1, ptr addrspace(3) @shared, align 4
  ret void
}

define spir_kernel void @shared_alias(ptr addrspace(1) %p) {
  store i32 1, ptr addrspace(3) @shared_a, align 4
  ret void
}

define spir_kernel void @alloca_type(ptr addrspace(1) %p) {
  %a = alloca { i32, ptr addrspace(4) }, align 8
  %f = getelementptr inbounds { i32, ptr addrspace(4) }, ptr %a, i32 0, i32 0
  store i32 1, ptr %f, align 4
  ret void
}

; A declaration without the dimension parameter is not the builtin.
define spir_kernel void @no_dim(ptr addrspace(1) %p) {
  %id = call spir_func i64 @_Z12get_group_idj()
  store i64 %id, ptr addrspace(1) %p, align 8
  ret void
}

declare spir_func i64 @_Z12get_group_idj()

define spir_kernel void @var_dim(ptr addrspace(1) %p, i32 %d) {
  %id = call spir_func i64 @_Z13get_global_idj(i32 %d)
  store i64 %id, ptr addrspace(1) %p, align 8
  ret void
}

declare spir_func i64 @_Z13get_global_idj(i32)

; The dimension parameter is an unsigned int.
define spir_kernel void @wide_dim(ptr addrspace(1) %p) {
  %n = call spir_func i64 @_Z14get_num_groupsj(i64 4294967296)
  store i64 %n, ptr addrspace(1) %p, align 8
  ret void
}

declare spir_func i64 @_Z14get_num_groupsj(i64)

; Byte offsets must be provably whole elements.
define spir_kernel void @misaligned(ptr addrspace(1) %p) {
  %g = getelementptr inbounds i8, ptr addrspace(1) %p, i64 2
  store float 0.0, ptr addrspace(1) %g, align 2
  ret void
}

define spir_kernel void @unknown_align(ptr addrspace(1) %p, i64 %j) {
  %g = getelementptr inbounds i8, ptr addrspace(1) %p, i64 %j
  store float 0.0, ptr addrspace(1) %g, align 1
  ret void
}

; Calls to kernels lowered before and after the caller.
define spir_kernel void @calls_kernel(i32 %x) {
  call spir_func void @early_kernel(i32 %x)
  ret void
}

define spir_kernel void @calls_late_kernel(i32 %x) {
  call spir_func void @late_kernel(i32 %x)
  ret void
}

define spir_kernel void @late_kernel(i32 %x) {
  ret void
}

; A struct holding a pointer has no place in a push constant block.
define spir_kernel void @byval(ptr byval({ ptr addrspace(1), i32 }) %s) {
  ret void
}

define spir_kernel void @ptr_select(ptr addrspace(1) %p, ptr addrspace(1) %q, i1 %c) {
  %r = select i1 %c, ptr addrspace(1) %p, ptr addrspace(1) %q
  store i32 0, ptr addrspace(1) %r, align 4
  ret void
}

define spir_kernel void @exec_mode(i32 %x) {
  ret void
}

; VecTypeHint, a Kernel execution mode.
!spirv.ExecutionMode = !{!0}
!0 = !{ptr @exec_mode, i32 30, i32 262149}
