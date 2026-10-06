%S = type { [2 x [9 x i8]] }
define spir_kernel void @short(ptr byval(%S) align 1 %s, ptr addrspace(1) %out, i32 %i) {
  %p = getelementptr inbounds %S, ptr %s, i32 0, i32 0, i32 %i, i32 0
  %v = load i8, ptr %p, align 1
  store i8 %v, ptr addrspace(1) %out, align 1
  ret void
}
