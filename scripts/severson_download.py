r"""
下载真实 Severson(MIT-Stanford)锂电池快充寿命数据集 (.mat) 到 data\

来源: data.matr.io  (Severson et al., Nature Energy 2019)
项目页(可手动下载更多批次): https://data.matr.io/1/projects/5c48dd2bc625d700019f3204

⚠️ 文件很大：batch1≈3.0GB, batch2≈2.0GB。请确保磁盘空间与网络。
   支持断点续传：中断后重跑本脚本会从断点继续。

batch3(2018-04-12)/ batch4(2019-01-24) 的稳定直链 matr.io 未公开，
若需要：去上面项目页手动下载这两个 .mat，放进 data\ 即可，
解析脚本 severson_parse.py 会自动识别 data\ 下所有 .mat。

运行: .venv\Scripts\python.exe severson_download.py
"""
from pathlib import Path
import requests

DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(exist_ok=True)

# 已验证可用的直链(2026-06 探测：HTTP 200 + 正确文件名/大小)
FILES = {
    "2017-05-12_batchdata_updated_struct_errorcorrect.mat":
        "https://data.matr.io/1/api/v1/file/5c86c0b5fa2ede00015ddf66/download",   # batch1 ~3.0GB
    "2017-06-30_batchdata_updated_struct_errorcorrect.mat":
        "https://data.matr.io/1/api/v1/file/5c86bf13fa2ede00015ddd82/download",   # batch2 ~2.0GB
    "2018-04-12_batchdata_updated_struct_errorcorrect.mat":
        "https://data.matr.io/1/api/v1/file/5c86bd64fa2ede00015ddbb2/download",   # batch3 ~3.2GB
    "2019-01-24_batchdata_updated_struct_errorcorrect.mat":
        "https://data.matr.io/1/api/v1/file/5dcef152110002c7215b2c90/download",   # batch4 ~2.6GB
}


def download(name, url):
    dst = DATA_DIR / name
    headers = {}
    pos = dst.stat().st_size if dst.exists() else 0
    # 先查总大小
    h = requests.head(url, allow_redirects=True, timeout=60)
    total = int(h.headers.get("Content-Length", 0))
    if pos and total and pos >= total:
        print(f"[跳过] {name} 已完整 ({total/1e9:.2f} GB)")
        return
    if pos:
        headers["Range"] = f"bytes={pos}-"
        print(f"[续传] {name} 从 {pos/1e9:.2f}/{total/1e9:.2f} GB 继续")
    else:
        print(f"[下载] {name} ({total/1e9:.2f} GB)")

    with requests.get(url, headers=headers, stream=True, timeout=120) as r:
        r.raise_for_status()
        mode = "ab" if pos else "wb"
        done = pos
        with open(dst, mode) as f:
            for chunk in r.iter_content(chunk_size=1 << 20):  # 1MB
                f.write(chunk)
                done += len(chunk)
                if total:
                    pct = done / total * 100
                    print(f"\r  {name[:24]}… {done/1e9:6.2f}/{total/1e9:.2f} GB ({pct:5.1f}%)",
                          end="", flush=True)
        print()
    print(f"[完成] {name}")


if __name__ == "__main__":
    print("=" * 60)
    print("Severson 数据集下载  (总计 ≈ 5 GB)")
    print("=" * 60)
    for name, url in FILES.items():
        try:
            download(name, url)
        except Exception as e:
            print(f"\n[失败] {name}: {e}\n  可手动下载放入 data\\ 后重试。")
    print("\n下一步: .venv\\Scripts\\python.exe severson_parse.py")
