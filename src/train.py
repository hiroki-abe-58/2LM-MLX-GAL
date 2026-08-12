"""学習ループ.

やることは3行で書ける:
  1. コーパスから連続した文字列をランダムに切り出す
  2. 「1文字ずらした列」を正解として次文字予測の誤差を計算する
  3. 誤差が小さくなる方向にパラメータを動かす

これを数千回繰り返すだけで、モデルは日本語の並び方と会話の書式を覚える。

使い方:
    python src/train.py                       # 既定 (約30分の予算で学習)
    python src/train.py --minutes 5           # まず5分だけ試す
    python src/train.py --steps 8000 --minutes 60
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
import time
from functools import partial
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.generate import chat_stream  # noqa: E402
from src.model import GPTConfig, MiniGPT  # noqa: E402
from src.tokenizer import CharTokenizer, SubwordTokenizer, Tokenizer, load_tokenizer  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SAMPLE_PROMPTS = ("こんにちは", "おすすめの本を教えてください")


def build_dataset(
    corpus: Path, cache_dir: Path, min_char_freq: int, vocab_size: int
) -> tuple[np.ndarray, Tokenizer]:
    """コーパスをトークンID列 (uint16) に変換してキャッシュする.

    語彙が65536未満なら uint16 で足りる。1,100万文字なら約22MBで、
    毎回エンコードし直すより速いしメモリにも丸ごと載る。

    vocab_size が 0 なら文字レベル、正の値なら SentencePiece を学習する。
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    tokens_path = cache_dir / "tokens.npy"
    stamp_path = cache_dir / "stamp.json"
    stamp = {
        "corpus": str(corpus),
        "mtime": corpus.stat().st_mtime,
        "min_char_freq": min_char_freq,
        "vocab_size": vocab_size,
    }

    cached = tokens_path.exists() and stamp_path.exists()
    if cached and json.loads(stamp_path.read_text()) == stamp:
        return np.load(tokens_path), load_tokenizer(cache_dir)

    text = corpus.read_text(encoding="utf-8")
    if vocab_size:
        tokenizer = SubwordTokenizer.train(text, vocab_size=vocab_size, model_dir=cache_dir)
    else:
        tokenizer = CharTokenizer.train(text, min_freq=min_char_freq)
    if tokenizer.vocab_size >= 2**16:
        raise SystemExit("語彙が65536を超えました。--vocab-size を下げてください。")
    tokens = np.array(tokenizer.encode(text), dtype=np.uint16)
    np.save(tokens_path, tokens)
    tokenizer.save(cache_dir)
    stamp_path.write_text(json.dumps(stamp))
    return tokens, tokenizer


def save_checkpoint(path: Path, model: MiniGPT, tokenizer: Tokenizer) -> None:
    tmp = path.with_name(path.name + ".tmp")
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    model.save_weights(str(tmp / "model.safetensors"))
    model.cfg.save(tmp / "config.json")
    tokenizer.save(tmp)
    if path.exists():
        shutil.rmtree(path)
    tmp.rename(path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default=str(ROOT / "data" / "corpus.txt"))
    ap.add_argument("--out", default=str(ROOT / "checkpoints" / "final"))
    ap.add_argument("--log", default="", help="損失ログの出力先 (既定: runs/loss.csv)")
    # 既定値は M1 Max (32コアGPU) の実測 47k tok/s から逆算したもの。
    # 自分のマシンでは --minutes 2 くらいで tok/s を測ってから決め直すとよい。
    ap.add_argument("--steps", type=int, default=4300)
    ap.add_argument("--minutes", type=float, default=35.0, help="この時間を超えたら打ち切る")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--block-size", type=int, default=256)
    ap.add_argument("--n-layer", type=int, default=6)
    ap.add_argument("--n-head", type=int, default=6)
    ap.add_argument("--n-embd", type=int, default=384)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--min-lr-ratio", type=float, default=0.1)
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--weight-decay", type=float, default=0.1)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--eval-interval", type=int, default=250)
    ap.add_argument("--eval-batches", type=int, default=20)
    ap.add_argument("--log-interval", type=int, default=50)
    ap.add_argument("--min-char-freq", type=int, default=1)
    ap.add_argument(
        "--vocab-size",
        type=int,
        default=0,
        help="SentencePiece の語彙数。0 なら文字レベル (既定)",
    )
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--no-compile", action="store_true")
    args = ap.parse_args()

    mx.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)

    tokens, tokenizer = build_dataset(
        Path(args.corpus), ROOT / "data" / "cache", args.min_char_freq, args.vocab_size
    )
    n_val = min(len(tokens) // 100, 200_000)
    train_data, val_data = tokens[:-n_val], tokens[-n_val:]

    cfg = GPTConfig(
        vocab_size=tokenizer.vocab_size,
        block_size=args.block_size,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
        dropout=args.dropout,
    )
    model = MiniGPT(cfg)
    mx.eval(model.parameters())

    print("=" * 62)
    print(f"  語彙数        : {cfg.vocab_size}")
    print(f"  学習トークン  : {len(train_data):,} (検証 {len(val_data):,})")
    print(f"  パラメータ数  : {model.n_params/1e6:.2f} M")
    print(f"  1ステップ     : {args.batch_size} x {args.block_size} = "
          f"{args.batch_size*args.block_size:,} トークン")
    print(f"  予算          : {args.steps} ステップ / {args.minutes} 分")
    print("=" * 62)

    schedule = optim.join_schedules(
        [
            optim.linear_schedule(args.lr * 0.02, args.lr, args.warmup),
            optim.cosine_decay(args.lr, max(args.steps - args.warmup, 1), args.lr * args.min_lr_ratio),
        ],
        [args.warmup],
    )
    optimizer = optim.AdamW(learning_rate=schedule, weight_decay=args.weight_decay)

    def loss_fn(m: MiniGPT, x: mx.array, y: mx.array) -> mx.array:
        return m.loss(x, y)

    loss_and_grad = nn.value_and_grad(model, loss_fn)
    # mx.random.state を inputs/outputs に含めないと Dropout の乱数が固定され、
    # コンパイル済み関数が毎回同じマスクを使ってしまう。
    state = [model.state, optimizer.state, mx.random.state]

    def _step(x: mx.array, y: mx.array) -> mx.array:
        loss, grads = loss_and_grad(model, x, y)
        if args.grad_clip > 0:
            grads, _ = optim.clip_grad_norm(grads, args.grad_clip)
        optimizer.update(model, grads)
        return loss

    step_fn = _step if args.no_compile else partial(mx.compile, inputs=state, outputs=state)(_step)

    def get_batch(data: np.ndarray, generator: np.random.Generator) -> tuple[mx.array, mx.array]:
        ix = generator.integers(0, len(data) - args.block_size - 1, size=args.batch_size)
        x = np.stack([data[i : i + args.block_size] for i in ix]).astype(np.int32)
        y = np.stack([data[i + 1 : i + 1 + args.block_size] for i in ix]).astype(np.int32)
        return mx.array(x), mx.array(y)

    def evaluate() -> float:
        model.eval()
        eval_rng = np.random.default_rng(0)  # 毎回同じ検証バッチで比較できるようにする
        total = 0.0
        for _ in range(args.eval_batches):
            x, y = get_batch(val_data, eval_rng)
            total += float(model.loss(x, y).item())
        model.train()
        return total / args.eval_batches

    runs = ROOT / "runs"
    runs.mkdir(exist_ok=True)
    log_path = Path(args.log) if args.log else runs / "loss.csv"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text("step,elapsed_sec,lr,train_loss,val_loss\n", encoding="utf-8")

    model.train()
    best_val = math.inf
    start = time.time()
    window: list[float] = []
    stop_reason = "ステップ数に到達"

    for step in range(1, args.steps + 1):
        x, y = get_batch(train_data, rng)
        loss = step_fn(x, y)
        mx.eval(state)  # ここで初めて実際に計算される (MLXは遅延評価)
        window.append(float(loss.item()))

        elapsed = time.time() - start
        if step % args.log_interval == 0:
            train_loss = sum(window) / len(window)
            window.clear()
            tps = step * args.batch_size * args.block_size / elapsed
            print(
                f"step {step:5d}/{args.steps} | loss {train_loss:.4f} | "
                f"lr {float(schedule(mx.array(step)).item()):.2e} | "
                f"{tps/1e3:.0f}k tok/s | {elapsed/60:.1f}分",
                flush=True,
            )
            with log_path.open("a", encoding="utf-8") as f:
                f.write(f"{step},{elapsed:.1f},{float(schedule(mx.array(step)).item()):.6f},{train_loss:.4f},\n")

        if step % args.eval_interval == 0 or step == args.steps:
            val_loss = evaluate()
            marker = ""
            if val_loss < best_val:
                best_val = val_loss
                save_checkpoint(Path(args.out), model, tokenizer)
                marker = "  <- 保存"
            print(f"  [検証] step {step} val_loss {val_loss:.4f} (最良 {best_val:.4f}){marker}", flush=True)
            with log_path.open("a", encoding="utf-8") as f:
                f.write(f"{step},{elapsed:.1f},,,{val_loss:.4f}\n")

            model.eval()
            for prompt in SAMPLE_PROMPTS:
                reply = "".join(
                    chat_stream(model, tokenizer, [], prompt, max_new_tokens=60, temperature=0.8)
                )
                print(f"  [試し] {prompt} -> {reply}", flush=True)
            model.train()

        if elapsed > args.minutes * 60:
            stop_reason = f"時間予算 {args.minutes} 分に到達"
            break

    print("=" * 62)
    print(f"終了: {stop_reason} / 経過 {(time.time()-start)/60:.1f} 分 / 最良 val_loss {best_val:.4f}")
    print(f"チェックポイント: {args.out}")
    print("=" * 62)


if __name__ == "__main__":
    main()
