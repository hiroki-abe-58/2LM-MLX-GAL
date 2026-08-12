"""複数のコーパスを比率を決めて混ぜる.

追加学習でキャラクターを付けるとき、新しいデータだけで回すとモデルは
元の知識を忘れる。ギャル語は覚えるが、質問に答えられなくなる。
防ぐには、元のデータを一定量混ぜたまま学習する。

コーパスは「1行1会話」なので、混ぜるのは行を混ぜるだけで済む。

引数は path:倍率 の形で並べる。倍率はそのファイル自身に対する掛け算で、
1より大きければ繰り返し、小さければランダムに間引く。

    # ギャルを3周ぶん、公開データを2%だけ混ぜる
    python tools/mix_corpus.py --out data/corpus_ft.txt \\
        data/corpus_gal.txt:3 data/corpus.txt:0.02
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path


def load_part(spec: str, rng: random.Random) -> tuple[str, list[str]]:
    path_str, _, ratio_str = spec.rpartition(":")
    if not path_str:
        raise SystemExit(f"倍率がありません: {spec} (例: data/corpus.txt:0.5)")
    try:
        ratio = float(ratio_str)
    except ValueError:
        raise SystemExit(f"倍率が数値ではありません: {spec}") from None
    if ratio <= 0:
        raise SystemExit(f"倍率は正の数にしてください: {spec}")

    path = Path(path_str)
    if not path.exists():
        raise SystemExit(f"見つかりません: {path}")
    lines = [ln for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]

    whole, fraction = divmod(ratio, 1.0)
    out = lines * int(whole)
    if fraction:
        out += rng.sample(lines, round(len(lines) * fraction))
    return path_str, out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("parts", nargs="+", help="path:倍率 を並べる")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    merged: list[str] = []
    print(f"{'元データ':<34} {'行数':>9} {'文字数':>12}")
    for spec in args.parts:
        name, lines = load_part(spec, rng)
        chars = sum(len(ln) for ln in lines)
        print(f"{spec:<34} {len(lines):>9,} {chars:>12,}")
        merged += lines

    rng.shuffle(merged)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(merged) + "\n", encoding="utf-8")

    total = sum(len(ln) for ln in merged)
    print(f"\n書き出し: {out}")
    print(f"  会話数   : {len(merged):,}")
    print(f"  総文字数 : {total:,}")


if __name__ == "__main__":
    main()
