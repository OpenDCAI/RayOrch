import os
import sys
import hashlib
import urllib.request
from typing import List, Optional


def sha256sum(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def download_file(urls: List[str], out_path: str, timeout: int = 60) -> str:
    """
    Try urls one by one, download to out_path (atomic rename).
    """
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    tmp_path = out_path + ".tmp"
    last_err: Optional[Exception] = None

    for url in urls:
        try:
            print(f"[download] try: {url}")
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                total = resp.headers.get("Content-Length")
                total = int(total) if total and total.isdigit() else None

                downloaded = 0
                with open(tmp_path, "wb") as f:
                    while True:
                        buf = resp.read(1024 * 1024)
                        if not buf:
                            break
                        f.write(buf)
                        downloaded += len(buf)
                        if total:
                            pct = downloaded * 100 // total
                            sys.stdout.write(f"\r[download] {downloaded/1024/1024:.1f}MB / {total/1024/1024:.1f}MB ({pct}%)")
                            sys.stdout.flush()

            if total:
                sys.stdout.write("\n")

            if os.path.exists(out_path):
                os.remove(out_path)
            os.replace(tmp_path, out_path)
            print(f"[download] saved: {out_path}")
            return out_path

        except Exception as e:
            last_err = e
            try:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
            except Exception:
                pass
            print(f"[download] failed: {e}")

    raise RuntimeError(f"All download urls failed for {out_path}. last_err={last_err}")


def ensure_ckpt(
    name: str,
    urls: List[str],
    out_path: str,
    expected_sha256: Optional[str] = None,
) -> str:
    if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
        print(f"[ok] exists: {out_path}")
    else:
        download_file(urls, out_path)

    if expected_sha256:
        got = sha256sum(out_path)
        if got.lower() != expected_sha256.lower():
            raise RuntimeError(f"[sha256 mismatch] {name}\n  expected: {expected_sha256}\n  got:      {got}")
        print(f"[ok] sha256: {got}")

    return out_path


def main():
    cache_dir = os.path.abspath("./cache/ckpt")
    os.makedirs(cache_dir, exist_ok=True)

    # --- YOLO26n weights (official GitHub release asset) ---
    # 官方地址：https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26n.pt
    # 国内可能需要代理/镜像，所以这里准备几个常见“GitHub 文件代理前缀”（可按你环境增减）
    yolo_url_main = "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26n.pt"
    yolo_urls = [
        yolo_url_main,
        "https://ghproxy.com/" + yolo_url_main,
        "https://mirror.ghproxy.com/" + yolo_url_main,
    ]
    yolo_out = os.path.join(cache_dir, "yolo26n.pt")

    # --- SAM ViT-B checkpoint (official dl.fbaipublicfiles.com) ---
    sam_url_main = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"
    sam_urls = [
        sam_url_main,
        # 这俩同样是可选代理前缀（如果你那边直连不稳）
        "https://ghproxy.com/" + sam_url_main,
        "https://mirror.ghproxy.com/" + sam_url_main,
    ]
    sam_out = os.path.join(cache_dir, "sam_vit_b_01ec64.pth")

    # 如果你想严格校验，可把 sha256 填上（SAM 的 sha256 常见公开，但这里不强行写死）
    ensure_ckpt("yolo26n", yolo_urls, yolo_out, expected_sha256=None)
    ensure_ckpt("sam_vit_b", sam_urls, sam_out, expected_sha256=None)

    print("\nAll checkpoints ready in:")
    print(cache_dir)
    print("\nPaths:")
    print("YOLO:", yolo_out)
    print("SAM :", sam_out)


if __name__ == "__main__":
    main()
