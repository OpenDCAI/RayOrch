import os
import urllib.request
from typing import List

def download_sample_images(cache_dir: str = "./.cache") -> List[str]:
    """
    从中国境内相对可用的图床/直链下载 5 张经典样例图到 ./cache
    优先：Gitee OpenCV 镜像 raw
    兜底：GitHub raw（若可访问）
    返回：本地绝对路径列表
    """
    os.makedirs(cache_dir, exist_ok=True)

    # 5 张 OpenCV 生态里常用的经典测试图
    names = ["lena.jpg", "fruits.jpg", "butterfly.jpg", "leuvenB.jpg", "fruits.jpg"]

    # ✅ 优先使用 Gitee 镜像：mirrors_opencv/opencv
    # Gitee 原始文件直链通常是：.../raw/<branch_or_tag>/<path>
    # 这里用 tag 4.13.0 更稳定，其次用 4.x 分支兜底
    bases = [
        # "https://gitee.com/mirrors_opencv/opencv/raw/4.13.0/samples/data/",
        "https://gitee.com/opencv/opencv/raw/4.x/samples/data/",
        # 兜底：GitHub raw（如果你网络能访问）
        # "https://raw.githubusercontent.com/opencv/opencv/4.13.0/samples/data/",
        # "https://raw.githubusercontent.com/opencv/opencv/4.x/samples/data/",
    ]

    downloaded: List[str] = []

    for name in names:
        out_path = os.path.abspath(os.path.join(cache_dir, name))
        print(out_path)
        if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
            downloaded.append(out_path)
            continue

        last_err: Exception | None = None
        ok = False

        for base in bases:
            url = base + name
            print(url)
            try:
                req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
                with urllib.request.urlopen(req, timeout=30) as resp:
                    data = resp.read()
                if not data:
                    raise RuntimeError("empty response")

                with open(out_path, "wb") as f:
                    f.write(data)

                ok = True
                break
            except Exception as e:
                last_err = e

        if not ok:
            raise RuntimeError(f"Failed to download {name}. last_err={last_err}")

        downloaded.append(out_path)

    print("Downloaded 5 images into:", os.path.abspath(cache_dir))
    return downloaded
