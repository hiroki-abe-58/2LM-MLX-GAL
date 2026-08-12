"""ターミナルの出力（ANSI付き）を画像にする (記事用).

実際に走らせた CLI の出力をそのまま流し込んで、記事に貼れる画像にする。
日本語は全角として2セル分進めることで、等幅レイアウトを崩さない。

    python src/chat_cli.py < demo.txt > /tmp/cli.txt
    python tools/render_terminal.py /tmp/cli.txt --out docs/images/cli-chat.png
"""

from __future__ import annotations

import argparse
import re
import unicodedata
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

MONO_FONT = "/System/Library/Fonts/Menlo.ttc"
JP_FONT_CANDIDATES = (
    Path.home() / "Library/Fonts/NotoSansJP-Regular.otf",
    Path("/System/Library/Fonts/Hiragino Sans GB.ttc"),
    Path("/System/Library/Fonts/ヒラギノ角ゴシック W3.ttc"),
)

ANSI_RE = re.compile(r"\033\[([0-9;]*)m")
PALETTE = {
    0: (226, 232, 240),
    2: (128, 138, 158),   # dim
    31: (255, 122, 138),
    32: (126, 231, 160),
    33: (255, 214, 122),
    36: (125, 211, 252),
}


def jp_font_path() -> str:
    for path in JP_FONT_CANDIDATES:
        if path.exists():
            return str(path)
    return MONO_FONT


def parse_ansi(text: str) -> list[list[tuple[str, tuple[int, int, int]]]]:
    """ANSIを解釈して、行ごとの (文字, 色) 列にする."""
    lines: list[list[tuple[str, tuple[int, int, int]]]] = [[]]
    color = PALETTE[0]
    pos = 0
    for match in ANSI_RE.finditer(text):
        for ch in text[pos : match.start()]:
            if ch == "\n":
                lines.append([])
            elif ch != "\r":
                lines[-1].append((ch, color))
        codes = [int(c) for c in match.group(1).split(";") if c != ""] or [0]
        for code in codes:
            color = PALETTE.get(code, color if code not in (0,) else PALETTE[0])
        pos = match.end()
    for ch in text[pos:]:
        if ch == "\n":
            lines.append([])
        elif ch != "\r":
            lines[-1].append((ch, color))
    return lines


Cell = tuple[str, tuple[int, int, int]]


def is_wide(ch: str) -> bool:
    return unicodedata.east_asian_width(ch) in ("W", "F", "A") and ord(ch) > 0x2000


def wrap(lines: list[list[Cell]], cols: int) -> list[list[Cell]]:
    """ターミナルと同じように、桁数を超えた分を次の行へ折り返す."""
    wrapped: list[list[Cell]] = []
    for line in lines:
        current: list[Cell] = []
        width = 0
        for cell in line:
            w = 2 if is_wide(cell[0]) else 1
            if width + w > cols:
                wrapped.append(current)
                current, width = [], 0
            current.append(cell)
            width += w
        wrapped.append(current)
    return wrapped


def gradient_background(width: int, height: int) -> Image.Image:
    top = np.array([16, 23, 54], dtype=np.float32)
    bottom = np.array([21, 15, 44], dtype=np.float32)
    ramp = np.linspace(0, 1, height, dtype=np.float32)[:, None, None]
    data = top[None, None, :] * (1 - ramp) + bottom[None, None, :] * ramp
    return Image.fromarray(np.repeat(data.astype(np.uint8), width, axis=1))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("input")
    ap.add_argument("--out", default="docs/images/cli-chat.png")
    ap.add_argument("--title", default="2lm — python src/chat_cli.py")
    ap.add_argument("--font-size", type=int, default=15)
    ap.add_argument("--scale", type=int, default=2)
    ap.add_argument("--cols", type=int, default=92)
    args = ap.parse_args()

    s = args.scale
    size = args.font_size * s
    mono = ImageFont.truetype(MONO_FONT, size)
    cell_w = mono.getlength("M")
    # 全角は2セル幅。フォントのemを2セルに合わせておかないと字間が空いて見える。
    jp = ImageFont.truetype(jp_font_path(), int(cell_w * 2 * 0.95))
    baseline_offset = mono.getmetrics()[0]

    line_h = size * 1.62
    pad = 22 * s
    bar_h = 34 * s
    margin = 26 * s

    lines = parse_ansi(Path(args.input).read_text(encoding="utf-8"))
    while lines and not lines[-1]:
        lines.pop()
    lines = wrap(lines, args.cols)

    win_w = int(cell_w * args.cols + pad * 2)
    win_h = int(bar_h + pad * 2 + line_h * len(lines))
    img = gradient_background(win_w + margin * 2, win_h + margin * 2)
    draw = ImageDraw.Draw(img, "RGBA")

    radius = 14 * s
    draw.rounded_rectangle(
        [margin, margin, margin + win_w, margin + win_h],
        radius=radius,
        fill=(10, 14, 30, 235),
        outline=(255, 255, 255, 46),
        width=s,
    )
    draw.rounded_rectangle(
        [margin, margin, margin + win_w, margin + bar_h + radius],
        radius=radius,
        fill=(255, 255, 255, 14),
    )
    for i, color in enumerate([(255, 95, 87), (255, 189, 46), (39, 201, 63)]):
        cx = margin + pad + i * 15 * s
        cy = margin + bar_h / 2
        r = 5.5 * s
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=color)
    title_font = ImageFont.truetype(jp_font_path(), int(size * 0.82))
    draw.text(
        (margin + win_w / 2, margin + bar_h / 2),
        args.title,
        font=title_font,
        fill=(168, 178, 198),
        anchor="mm",
    )

    y = margin + bar_h + pad
    for line in lines:
        x = margin + pad
        for ch, color in line:
            if ch == " ":
                x += cell_w
                continue
            wide = is_wide(ch)
            font = jp if ord(ch) > 0x2000 else mono
            # ベースラインを揃えないと、和欧混在の行で文字が上下にずれる
            draw.text((x, y + baseline_offset), ch, font=font, fill=color, anchor="ls")
            x += cell_w * (2 if wide else 1)
        y += line_h

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    img.save(out)
    print(f"保存: {out} ({img.width}x{img.height}, {len(lines)}行)")


if __name__ == "__main__":
    main()
