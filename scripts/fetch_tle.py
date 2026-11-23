"""
可选辅助脚本：从 CelesTrak 下载最新 Starlink 公开 TLE 星历
==========================================================
对应汇报第 13、18 页（[T1] CelesTrak GP / TLE 与 SGP4）。
运行方式：
    python scripts/fetch_tle.py
下载完成后，可在 config/scenarios.yaml 中将 constellation.mode 改为 "tle"（需 pip install sgp4）。
"""

from pathlib import Path
import urllib.request

CELESTRAK_URL = "https://celestrak.org/NORAD/elements/gp.php?GROUP=starlink&FORMAT=tle"
OUT_PATH = Path(__file__).resolve().parent.parent / "data" / "starlink.tle"


def main() -> None:
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    print(f"正在从 CelesTrak 下载 Starlink TLE 星历: {CELESTRAK_URL}")
    req = urllib.request.Request(CELESTRAK_URL, headers={"User-Agent": "PAPH-LEO-Sim/1.0"})
    with urllib.request.urlopen(req, timeout=20) as resp:
        text = resp.read().decode("utf-8")
    OUT_PATH.write_text(text, encoding="utf-8")
    n_sats = len([ln for ln in text.splitlines() if ln.startswith("1 ")])
    print(f"下载成功！共获取 {n_sats} 颗卫星的 TLE 数据，已保存至: {OUT_PATH}")


if __name__ == "__main__":
    main()
