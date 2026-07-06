/* 提供 glibc>=2.32 的 __libc_single_threaded 符号(带 GLIBC_2.32 版本号),
 * 供针对新 glibc 编译、却运行在 glibc 2.28 上的 flash_attn_2_cuda.so 解析。
 * 值取 0 = "非单线程", 触发 libstdc++ 的多线程(原子)安全路径, 语义正确。
 * 用法: LD_PRELOAD=.../glibc_single_threaded_shim.so python ... */
char __libc_single_threaded = 0;
