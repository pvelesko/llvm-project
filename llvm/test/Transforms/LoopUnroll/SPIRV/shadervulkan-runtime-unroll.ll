; RUN: opt -S -mtriple=spirv64-unknown-chipstar-shadervulkan -passes=loop-unroll %s | FileCheck %s --check-prefix=VK
; RUN: opt -S -mtriple=spirv64-unknown-vulkan1.3-compute -passes=loop-unroll %s | FileCheck %s --check-prefix=NOUNROLL

; A store loop of unknown trip count is runtime-unrolled by 8 for chipStar
; Vulkan only.

; VK-LABEL: @fill(
; VK:       {{^}}loop:
; VK-COUNT-8: store i32
; VK-NOT:   store i32
; VK:       {{^}}loop.epil:
; NOUNROLL-LABEL: @fill(
; NOUNROLL:     store i32
; NOUNROLL-NOT: store i32

define void @fill(ptr addrspace(1) %p, i64 %n) {
entry:
  %cmp = icmp sgt i64 %n, 0
  br i1 %cmp, label %loop, label %exit

loop:
  %i = phi i64 [ 0, %entry ], [ %i.next, %loop ]
  %gep = getelementptr inbounds i32, ptr addrspace(1) %p, i64 %i
  %v = trunc i64 %i to i32
  store i32 %v, ptr addrspace(1) %gep, align 4
  %i.next = add nuw nsw i64 %i, 1
  %done = icmp eq i64 %i.next, %n
  br i1 %done, label %exit, label %loop

exit:
  ret void
}
