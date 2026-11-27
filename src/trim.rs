//! 载入完了把分配器手里空着的页还给系统（EM-5）。
//!
//! 为什么：载入时 ONNX Runtime 先把整份模型读进内存解析（61 MB），再一块块造权重、做图优化，大大小小的分配交错着来。解析用的
//! 那些放掉以后，glibc 的堆中间留下一片片空着的页：它只会把堆顶上空着的还给系统，中间的一直算在这个进程头上，实测约 9 MB。
//! `malloc_trim(0)` 把各个分配区里整页空着的都还回去（`madvise`），只在载入完了做一次，几毫秒。
//!
//! 只有 glibc 有 `malloc_trim`；别的平台（musl、macOS、Windows）什么都不做，照原来。

// 这一个文件要 `unsafe`：调 C 库的函数没有安全的接口。别处照旧不许。
#![allow(unsafe_code)]

/// 把分配器手里整页空着的内存还给系统。
#[cfg(all(target_os = "linux", target_env = "gnu"))]
pub(crate) fn release_free_pages() {
    // 安全：`malloc_trim` 没有前提条件，任何时候都可以调；交回的只是有没有还掉，不影响什么。
    unsafe {
        libc::malloc_trim(0);
    }
}

/// 别的平台什么都不做。
#[cfg(not(all(target_os = "linux", target_env = "gnu")))]
pub(crate) fn release_free_pages() {}
