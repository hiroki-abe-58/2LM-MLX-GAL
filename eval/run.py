"""固定評価セットでモデルを採点する.

「賢くなった気がする」を潰すための道具。改善のたびにこれを回して、
同じ数字で前のモデルと比べる。

指標は4つ。

    bits/char   : 検証セットの損失を「1文字あたりのビット数」に正規化したもの。
                  トークナイザを文字レベルからサブワードに変えても比較できる。
    反復率      : 同じ3文字並びが3回以上出てくる返答の割合。
    主題保持率  : 質問の主題語が返答に現れた割合 (keywords を持つ設問のみ)。
    破綻率      : 空・極端に短い・打ち切りで終わった返答の割合。

使い方:
    python eval/run.py --make-holdout        # 固定検証セットを切り出す (最初に1回)
    python eval/run.py --tag v1              # 採点して runs/eval_v1.json に保存
    python eval/run.py --tag v2 --compare v1 # 前回と並べて表示
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import time
from collections import Counter
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.generate import build_chat_prompt, generate_stream, load_bundle  # noqa: E402
from src.tokenizer import Tokenizer  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
QUESTIONS = HERE / "questions.jsonl"
HOLDOUT = HERE / "holdout.txt"
RUNS = ROOT / "runs"

# 学習時の文脈長が変わっても bits/char を比べられるよう、評価の文脈長は固定する。
EVAL_BLOCK = 256
_MARKER_RE = re.compile(r"<\|(?:user|assistant|end|unk)\|>")


# --- 固定検証セット ---------------------------------------------------------


def make_holdout(corpus: Path, min_chars: int) -> None:
    """コーパスの末尾から固定検証セットを切り出す.

    data/prepare.py は書き出す前に会話をシャッフルするので、末尾を取ると
    使ったデータセット全体からの無作為標本になる。ここを外すと物差しが歪む。
    最初は oasst1 だけのコーパスから切っていたが、それだと oasst1 だけで
    学習したモデルにとってだけ「見慣れた文体」になり、データを増やした
    モデルが不当に低く出た。**検証セットは必ず全ソースから切ること。**

    以降のバージョンは data/prepare.py --exclude でこの会話を除いて学習する。
    """
    lines = corpus.read_text(encoding="utf-8").splitlines()
    picked: list[str] = []
    total = 0
    for line in reversed(lines):
        picked.append(line)
        total += len(_MARKER_RE.sub("", line))
        if total >= min_chars:
            break
    picked.reverse()
    HOLDOUT.write_text("\n".join(picked) + "\n", encoding="utf-8")
    print(f"固定検証セット: {HOLDOUT}")
    print(f"  会話数: {len(picked)}")
    print(f"  文字数: {total}")
    print("\n以降の学習では次を付けて、この会話を訓練データから除いてください。")
    print(f"  python data/prepare.py --exclude {HOLDOUT.relative_to(ROOT)}")


# --- 指標 -------------------------------------------------------------------


def bits_per_char(model, tokenizer: Tokenizer, text: str, batch_size: int = 16) -> float:
    """検証テキストの交差エントロピーを1文字あたりのビット数に直す.

    分母をトークン数ではなく **文字数** にするのがポイント。
    サブワード化するとトークン数が減るので、nats/token のままでは
    「1トークンの予測が難しくなっただけ」なのに悪化して見える。
    """
    ids = tokenizer.encode(text)
    n_chars = len(_MARKER_RE.sub("", text))
    stride = EVAL_BLOCK
    windows = [ids[i : i + stride + 1] for i in range(0, len(ids) - 1, stride)]
    windows = [w for w in windows if len(w) == stride + 1]

    total_nats = 0.0
    for start in range(0, len(windows), batch_size):
        chunk = windows[start : start + batch_size]
        arr = mx.array(np.array(chunk, dtype=np.int32))
        logits = model(arr[:, :-1])
        loss = nn.losses.cross_entropy(
            logits.reshape(-1, logits.shape[-1]), arr[:, 1:].reshape(-1), reduction="sum"
        )
        total_nats += float(loss.item())
        mx.eval(loss)
    # 端数の窓を捨てているので、実際に採点した文字数で割り直す。
    scored_tokens = len(windows) * stride
    scored_ratio = scored_tokens / max(len(ids) - 1, 1)
    return total_nats / math.log(2) / (n_chars * scored_ratio)


def has_repetition(text: str, n: int = 3, threshold: int = 3) -> bool:
    if len(text) < n * threshold:
        return False
    grams = Counter(text[i : i + n] for i in range(len(text) - n + 1))
    return max(grams.values()) >= threshold


def is_broken(text: str, truncated: bool) -> bool:
    return len(text.strip()) < 4 or truncated or "<|unk|>" in text


# --- 実行 -------------------------------------------------------------------


def answer(model, tokenizer, history, user_text, max_new_tokens, **kwargs) -> tuple[str, bool]:
    """1問に答える. 返り値は (返答, 打ち切られたか)."""
    prompt = build_chat_prompt(tokenizer, history, user_text, model.cfg.block_size)
    stop_ids = (tokenizer.end_id, tokenizer.user_id)
    out: list[int] = []
    for token_id in generate_stream(
        model, prompt, stop_ids=stop_ids, max_new_tokens=max_new_tokens, **kwargs
    ):
        out.append(token_id)
    return tokenizer.decode(out), len(out) >= max_new_tokens


def evaluate(args) -> dict:
    model, tokenizer = load_bundle(args.ckpt)

    print("検証セットの損失を計算中...")
    text = HOLDOUT.read_text(encoding="utf-8")
    started = time.time()
    bpc = bits_per_char(model, tokenizer, text)
    print(f"  bits/char = {bpc:.3f}  ({time.time() - started:.1f}秒)\n")

    questions = [
        json.loads(line)
        for line in QUESTIONS.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    results = []
    for index, q in enumerate(questions):
        # 設問ごとに種を固定する。順番を変えても同じ結果になるようにするため。
        mx.random.seed(args.seed + index)
        history: list[tuple[str, str]] = []
        reply, truncated = "", False
        for turn in q["turns"]:
            reply, truncated = answer(
                model,
                tokenizer,
                history,
                turn,
                args.max_new_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
                repetition_penalty=args.repetition_penalty,
            )
            history.append((turn, reply))
        keywords = q.get("keywords", [])
        results.append(
            {
                "id": q["id"],
                "category": q["category"],
                "turns": q["turns"],
                "reply": reply,
                "repetition": has_repetition(reply),
                "broken": is_broken(reply, truncated),
                "keyword_hit": (any(k in reply for k in keywords) if keywords else None),
            }
        )

    scored = [r for r in results if r["keyword_hit"] is not None]
    summary = {
        "tag": args.tag,
        "ckpt": args.ckpt,
        "bits_per_char": round(bpc, 3),
        "repetition_rate": round(sum(r["repetition"] for r in results) / len(results), 3),
        "topic_rate": (
            round(sum(r["keyword_hit"] for r in scored) / len(scored), 3) if scored else None
        ),
        "broken_rate": round(sum(r["broken"] for r in results) / len(results), 3),
        "avg_reply_len": round(sum(len(r["reply"]) for r in results) / len(results), 1),
        "sampling": {
            "temperature": args.temperature,
            "top_k": args.top_k,
            "repetition_penalty": args.repetition_penalty,
            "seed": args.seed,
        },
        "results": results,
    }
    return summary


def show(summary: dict, baseline: dict | None) -> None:
    def line(label: str, key: str, fmt: str = "{:.3f}", better: str = "low") -> str:
        now = summary[key]
        if now is None:
            return f"  {label:<14}: 未測定"
        text = f"  {label:<14}: {fmt.format(now)}"
        if baseline and baseline.get(key) is not None:
            before = baseline[key]
            delta = now - before
            arrow = "改善" if (delta < 0) == (better == "low") else "悪化"
            if abs(delta) < 1e-9:
                arrow = "変化なし"
            text += f"   (前回 {fmt.format(before)} / {delta:+.3f} {arrow})"
        return text

    print("=" * 70)
    print(f"  評価: {summary['tag']}   ({summary['ckpt']})")
    print("=" * 70)
    print(line("bits/char", "bits_per_char"))
    print(line("反復率", "repetition_rate"))
    print(line("主題保持率", "topic_rate", better="high"))
    print(line("破綻率", "broken_rate"))
    print(line("平均返答長", "avg_reply_len", fmt="{:.1f}", better="high"))
    print()
    for r in summary["results"]:
        flags = "".join(
            [
                "反" if r["repetition"] else "",
                "破" if r["broken"] else "",
                "主" if r["keyword_hit"] else "",
            ]
        )
        print(f"  [{r['category']}] {' / '.join(r['turns'])}")
        print(f"    -> {r['reply']}  {flags}")
    print()
    print("  凡例: 反=反復あり 破=破綻 主=主題語を保持")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/final")
    ap.add_argument("--tag", default="v1")
    ap.add_argument("--compare", default=None, help="比較する過去のタグ")
    ap.add_argument("--make-holdout", action="store_true")
    ap.add_argument("--holdout-chars", type=int, default=40000)
    ap.add_argument("--corpus", default=str(ROOT / "data" / "corpus.txt"))
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-k", type=int, default=40)
    ap.add_argument("--repetition-penalty", type=float, default=1.15)
    ap.add_argument("--max-new-tokens", type=int, default=200)
    ap.add_argument("--seed", type=int, default=777)
    args = ap.parse_args()

    if args.make_holdout:
        make_holdout(Path(args.corpus), args.holdout_chars)
        return

    if not HOLDOUT.exists():
        raise SystemExit("固定検証セットがありません。先に --make-holdout を実行してください。")

    summary = evaluate(args)
    baseline = None
    if args.compare:
        path = RUNS / f"eval_{args.compare}.json"
        if path.exists():
            baseline = json.loads(path.read_text(encoding="utf-8"))
        else:
            print(f"比較対象が見つかりません: {path}\n")
    show(summary, baseline)

    RUNS.mkdir(exist_ok=True)
    out = RUNS / f"eval_{args.tag}.json"
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n保存: {out}")


if __name__ == "__main__":
    main()
