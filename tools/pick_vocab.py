"""語彙サイズを決めるための実測ツール.

サブワード化の効果は「1トークンあたり何文字を運べるか」で決まるが、
語彙を大きくするほど埋め込みのパラメータが増え、1トークンあたりの
出現回数が減る。手元のコーパス量に対して割に合う点を、実測して選ぶ。

    python tools/pick_vocab.py --corpus data/exp/corpus_b.txt
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.tokenizer import SubwordTokenizer  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="data/exp/corpus_b.txt")
    ap.add_argument("--sizes", default="4000,8000,12000,16000,24000")
    ap.add_argument("--n-embd", type=int, default=384)
    ap.add_argument("--sample-lines", type=int, default=4000, help="圧縮率の測定に使う行数")
    args = ap.parse_args()

    text = Path(args.corpus).read_text(encoding="utf-8")
    lines = text.splitlines()
    sample = "\n".join(lines[: args.sample_lines])
    n_chars_sample = len(sample)
    n_chars_total = len(text)

    print(f"コーパス: {args.corpus}  ({n_chars_total:,} 文字 / {len(lines):,} 会話)")
    print(f"測定標本: 先頭 {args.sample_lines:,} 会話 ({n_chars_sample:,} 文字)\n")

    header = (
        f"{'語彙':>7}  {'文字/トークン':>12}  {'総トークン数':>13}"
        f"  {'埋め込み':>9}  {'トークン/語彙':>12}"
    )
    print(header)
    print("-" * len(header))

    for size in [int(s) for s in args.sizes.split(",")]:
        with tempfile.TemporaryDirectory() as tmp:
            tokenizer = SubwordTokenizer.train(text, vocab_size=size, model_dir=Path(tmp))
            ids = tokenizer.encode(sample)
        chars_per_token = n_chars_sample / len(ids)
        total_tokens = n_chars_total / chars_per_token
        embedding_m = size * args.n_embd / 1e6
        print(
            f"{size:>7,}  {chars_per_token:>12.3f}  {total_tokens:>13,.0f}"
            f"  {embedding_m:>8.2f}M  {total_tokens / size:>12,.0f}"
        )

    print(
        "\n文字/トークン が大きいほど文脈に多くの文章が入る。"
        "\nトークン/語彙 は1つの語彙あたりの平均学習回数。小さすぎると覚えきれない。"
    )


if __name__ == "__main__":
    main()
