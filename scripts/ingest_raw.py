#!/usr/bin/env python3
"""Audit the raw/ enrollment dump.

Phase 1 (this script): structure + image-quality audit, no ML.
  - folder = idpersonal, files = <idpersonal>_NNNN.jpg
  - detect exact-duplicate files (md5)
  - per image: dimensions, mean luma, blur (variance of Laplacian), EXIF orientation
  - gate each image (dark / bright / blurry / tiny / corrupt)
  - roll up per idpersonal: usable image count -> verdict OK / WEAK / UNUSABLE

Writes a JSON report and prints a summary. Nothing here mutates raw/.

Phase 2 (separate, later): face detect + align + embedding + identity clustering.

Usage:
  python scripts/ingest_raw.py --raw /path/to/raw --out report.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from collections import Counter
from dataclasses import dataclass, asdict, field
from pathlib import Path

import numpy as np

try:
    from PIL import Image, ImageOps
except ImportError:  # pragma: no cover
    sys.exit("Pillow required: pip install pillow")

log = logging.getLogger("ingest_raw")

# --- quality gates (heuristic; override via CLI, distributions are reported so you can retune) ---
DEFAULTS = dict(
    dark_luma=50.0,      # mean grayscale < this  -> underexposed
    bright_luma=205.0,   # mean grayscale > this  -> overexposed / blown
    blur_var=100.0,      # variance of Laplacian < this -> soft / motion blur
    min_side=112,        # min(width, height) < this -> too small for ArcFace align
    weak_max=2,          # <= this many usable images -> WEAK identity
)


@dataclass
class ImgStat:
    path: str
    w: int = 0
    h: int = 0
    bytes: int = 0
    luma: float = 0.0
    blur: float = 0.0
    oriented: bool = False        # EXIF orientation tag present (rotation baked at read)
    md5: str = ""
    flags: list[str] = field(default_factory=list)

    @property
    def usable(self) -> bool:
        return not self.flags


def lap_var(gray: np.ndarray) -> float:
    """Variance of the 3x3 Laplacian response — classic sharpness proxy."""
    k = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float32)
    a = gray.astype(np.float32)
    # valid convolution without scipy
    r = (
        k[0, 1] * a[:-2, 1:-1] + k[1, 0] * a[1:-1, :-2] + k[1, 1] * a[1:-1, 1:-1]
        + k[1, 2] * a[1:-1, 2:] + k[2, 1] * a[2:, 1:-1]
    )
    return float(r.var())


def analyze_image(p: Path, cfg: dict) -> ImgStat:
    st = ImgStat(path=str(p))
    try:
        st.bytes = p.stat().st_size
        st.md5 = hashlib.md5(p.read_bytes()).hexdigest()
        with Image.open(p) as im:
            st.oriented = bool(im.getexif().get(0x0112, 0) not in (0, 1))
            im = ImageOps.exif_transpose(im)          # bake rotation
            st.w, st.h = im.size
            g = np.asarray(im.convert("L"))
        st.luma = float(g.mean())
        st.blur = lap_var(g) if g.ndim == 2 and min(g.shape) > 2 else 0.0
    except Exception as e:  # corrupt / unreadable
        st.flags.append("corrupt")
        log.debug("corrupt %s: %s", p, e)
        return st

    if min(st.w, st.h) < cfg["min_side"]:
        st.flags.append("tiny")
    if st.luma < cfg["dark_luma"]:
        st.flags.append("dark")
    elif st.luma > cfg["bright_luma"]:
        st.flags.append("bright")
    if st.blur < cfg["blur_var"]:
        st.flags.append("blurry")
    return st


def verdict(n_usable: int, weak_max: int) -> str:
    if n_usable == 0:
        return "UNUSABLE"
    if n_usable <= weak_max:
        return "WEAK"
    return "OK"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", type=Path, required=True, help="raw/ dir (subdir per idpersonal)")
    ap.add_argument("--out", type=Path, default=Path("raw_audit.json"))
    ap.add_argument("--dark-luma", type=float, default=DEFAULTS["dark_luma"])
    ap.add_argument("--bright-luma", type=float, default=DEFAULTS["bright_luma"])
    ap.add_argument("--blur-var", type=float, default=DEFAULTS["blur_var"])
    ap.add_argument("--min-side", type=int, default=DEFAULTS["min_side"])
    ap.add_argument("--weak-max", type=int, default=DEFAULTS["weak_max"])
    ap.add_argument("-v", "--verbose", action="store_true")
    a = ap.parse_args()
    logging.basicConfig(level=logging.DEBUG if a.verbose else logging.INFO, format="%(levelname)s %(message)s")

    cfg = dict(dark_luma=a.dark_luma, bright_luma=a.bright_luma, blur_var=a.blur_var,
               min_side=a.min_side, weak_max=a.weak_max)

    if not a.raw.is_dir():
        log.error("not a dir: %s", a.raw)
        return 2

    folders = sorted([d for d in a.raw.iterdir() if d.is_dir()])
    log.info("scanning %d folders under %s", len(folders), a.raw)

    per_person: dict[str, dict] = {}
    all_stats: list[ImgStat] = []
    md5_global: Counter[str] = Counter()

    for i, d in enumerate(folders, 1):
        imgs = sorted(d.glob("*.jpg")) + sorted(d.glob("*.jpeg")) + sorted(d.glob("*.png"))
        stats = [analyze_image(p, cfg) for p in imgs]
        all_stats.extend(stats)
        for s in stats:
            md5_global[s.md5] += 1
        seen: set[str] = set()
        uniq = [s for s in stats if not (s.md5 in seen or seen.add(s.md5))]
        n_usable = sum(1 for s in uniq if s.usable)
        per_person[d.name] = dict(
            n_files=len(stats),
            n_unique=len(uniq),
            n_dupes=len(stats) - len(uniq),
            n_usable=n_usable,
            verdict=verdict(n_usable, cfg["weak_max"]),
            flag_counts=dict(Counter(f for s in stats for f in s.flags)),
        )
        if i % 200 == 0:
            log.info("  %d/%d", i, len(folders))

    # global rollups
    dupe_files = sum(c - 1 for c in md5_global.values() if c > 1)
    v_counts = Counter(p["verdict"] for p in per_person.values())
    usable_hist = Counter()
    for p in per_person.values():
        b = p["n_usable"]
        usable_hist[str(b) if b <= 5 else "6-10" if b <= 10 else "11-20" if b <= 20 else "21+"] += 1
    flag_totals = Counter()
    for s in all_stats:
        for f in s.flags:
            flag_totals[f] += 1

    luma = np.array([s.luma for s in all_stats if not ("corrupt" in s.flags)])
    blur = np.array([s.blur for s in all_stats if not ("corrupt" in s.flags)])

    def pct(x, q):
        return round(float(np.percentile(x, q)), 1) if len(x) else None

    report = dict(
        raw=str(a.raw),
        config=cfg,
        totals=dict(
            idpersonal=len(folders),
            files=len(all_stats),
            unique_files=len(all_stats) - dupe_files,
            dupe_files=dupe_files,
            corrupt=flag_totals.get("corrupt", 0),
        ),
        identity_verdict=dict(v_counts),
        usable_images_per_identity=dict(sorted(usable_hist.items(), key=lambda kv: (len(kv[0]), kv[0]))),
        image_flag_totals=dict(flag_totals),
        luma_percentiles={q: pct(luma, q) for q in (5, 25, 50, 75, 95)},
        blur_percentiles={q: pct(blur, q) for q in (5, 25, 50, 75, 95)},
        oriented_images=sum(1 for s in all_stats if s.oriented),
        per_person=per_person,
    )
    a.out.write_text(json.dumps(report, indent=2))

    t = report["totals"]
    print("\n================ RAW AUDIT ================")
    print(f" idpersonal (folders) : {t['idpersonal']}")
    print(f" image files          : {t['files']}  (unique {t['unique_files']}, exact-dupe {t['dupe_files']}, corrupt {t['corrupt']})")
    print(f" images w/ EXIF rotation: {report['oriented_images']}")
    print("\n -- identity verdict (usable images after dedup + quality gate) --")
    for k in ("OK", "WEAK", "UNUSABLE"):
        print(f"   {k:9}: {v_counts.get(k, 0)}")
    print("\n -- usable images per identity --")
    for k, n in report["usable_images_per_identity"].items():
        print(f"   {k:>5} img : {n}")
    print("\n -- image quality flags (per image, images can have >1) --")
    for k, n in flag_totals.most_common():
        print(f"   {k:9}: {n}")
    print("\n -- mean luma percentiles (0-255) --")
    print("  ", report["luma_percentiles"])
    print(" -- blur (var-of-Laplacian) percentiles; higher = sharper --")
    print("  ", report["blur_percentiles"])
    print(f"\n report -> {a.out}")
    print("==========================================\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
