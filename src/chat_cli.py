"""CLIチャット.

    python src/chat_cli.py
    python src/chat_cli.py --ckpt checkpoints/final --temperature 0.9

チャット中に使えるコマンド:
    /reset          会話履歴を消す
    /temp 0.9       ランダムさを変える (0に近いほど堅い)
    /topk 40        候補を上位k個に絞る (0で無効)
    /penalty 1.2    繰り返しへのペナルティ
    /tokens 200     1回に生成する最大文字数
    /config         現在の設定を表示
    /exit           終了
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import mlx.core as mx

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.generate import DEFAULTS, chat_stream, load_bundle  # noqa: E402

RESET = "\033[0m"
DIM = "\033[2m"
CYAN = "\033[36m"
GREEN = "\033[32m"
YELLOW = "\033[33m"


def parse_command(line: str, params: dict) -> bool | None:
    """コマンドなら処理して True/False を返す. 通常の発言なら None."""
    if not line.startswith("/"):
        return None
    parts = line.split()
    name = parts[0][1:]
    arg = parts[1] if len(parts) > 1 else None
    keys = {"temp": "temperature", "topk": "top_k", "penalty": "repetition_penalty",
            "tokens": "max_new_tokens"}
    if name in ("exit", "quit", "q"):
        return False
    if name == "reset":
        print(f"{DIM}会話履歴を消しました{RESET}")
        return True
    if name == "config":
        print(f"{DIM}" + "  ".join(f"{k}={v}" for k, v in params.items()) + RESET)
        return True
    if name == "help":
        print(__doc__)
        return True
    if name in keys and arg is not None:
        key = keys[name]
        params[key] = int(arg) if isinstance(DEFAULTS[key], int) else float(arg)
        print(f"{DIM}{key} = {params[key]}{RESET}")
        return True
    print(f"{YELLOW}不明なコマンドです。/help を見てください。{RESET}")
    return True


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/final")
    ap.add_argument("--temperature", type=float, default=DEFAULTS["temperature"])
    ap.add_argument("--top-k", type=int, default=DEFAULTS["top_k"])
    ap.add_argument("--repetition-penalty", type=float, default=DEFAULTS["repetition_penalty"])
    ap.add_argument("--max-new-tokens", type=int, default=DEFAULTS["max_new_tokens"])
    ap.add_argument("--history", type=int, default=2, help="何ターン前まで文脈に含めるか")
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()

    if args.seed is not None:
        mx.random.seed(args.seed)

    model, tokenizer = load_bundle(args.ckpt)
    params = {
        "temperature": args.temperature,
        "top_k": args.top_k,
        "repetition_penalty": args.repetition_penalty,
        "max_new_tokens": args.max_new_tokens,
    }

    print("=" * 60)
    print(f"  2LM chat  {model.n_params/1e6:.2f}M params / "
          f"vocab {tokenizer.vocab_size} / context {model.cfg.block_size}")
    print(f"{DIM}  /help でコマンド一覧, /exit で終了{RESET}")
    print("=" * 60)

    history: list[tuple[str, str]] = []
    while True:
        try:
            line = input(f"\n{CYAN}あなた>{RESET} ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not line:
            continue
        if not sys.stdin.isatty():
            # パイプで流し込んだときは入力が画面に出ないため、自分で echo する
            print(line)
        result = parse_command(line, params)
        if result is False:
            break
        if result is True:
            if line.startswith("/reset"):
                history.clear()
            continue

        print(f"{GREEN}2LM  >{RESET} ", end="", flush=True)
        start = time.time()
        pieces = []
        for piece in chat_stream(model, tokenizer, history[-args.history:], line, **params):
            print(piece, end="", flush=True)
            pieces.append(piece)
        reply = "".join(pieces)
        took = time.time() - start
        speed = len(reply) / took if took > 0 else 0
        print(f"\n{DIM}({len(reply)} 文字 / {took:.1f}秒 / {speed:.0f} 文字毎秒){RESET}")
        history.append((line, reply))

    print("またね。")


if __name__ == "__main__":
    main()
