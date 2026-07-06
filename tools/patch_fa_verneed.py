"""把 flash_attn_2_cuda.so 中 libc.so.6 的 GLIBC_2.32 版本需求改名为 GLIBC_2.14。
__libc_single_threaded 的版本已被 patchelf 清除并由 shim 提供, 该 verneed 条目无符号引用,
改名为系统 glibc 2.28 实际存在的版本即可通过加载器的版本存在性校验。"""
import sys
import lief

so = sys.argv[1]
b = lief.parse(so)
done = False
for req in b.symbols_version_requirement:
    if req.name != "libc.so.6":
        continue
    auxs = list(req.get_auxiliary_symbols())
    hash14 = next((a.hash for a in auxs if a.name == "GLIBC_2.14"), None)
    for a in auxs:
        if a.name == "GLIBC_2.32":
            a.name = "GLIBC_2.14"
            if hash14 is not None:
                a.hash = hash14
            done = True
            print(f"已改名 GLIBC_2.32 -> GLIBC_2.14 (hash={hash14})")
assert done, "未找到 GLIBC_2.32 verneed 条目"
b.write(so)
print("写回完成:", so)
