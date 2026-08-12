"""生成 (サンプリング) 処理.

学習済みモデルは「次の1文字の確率分布」しか出さない。
そこからどう1文字を選ぶかがサンプリングで、ここの設定だけで
同じモデルが「壊れた繰り返し」と「それっぽい返答」の間を行き来する。

CLI (chat_cli.py) と Web API (server.py) はどちらもこのモジュールを呼ぶ。
"""

from __future__ import annotations

import sys
from collections.abc import Iterator
from pathlib import Path

import mlx.core as mx

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.model import GPTConfig, MiniGPT  # noqa: E402
from src.tokenizer import ASSISTANT, END, USER, Tokenizer, load_tokenizer  # noqa: E402

DEFAULTS = {
    "temperature": 0.8,
    "top_k": 40,
    "repetition_penalty": 1.15,
    "max_new_tokens": 200,
}


def load_bundle(ckpt_dir: str | Path) -> tuple[MiniGPT, Tokenizer]:
    ckpt = Path(ckpt_dir)
    tokenizer = load_tokenizer(ckpt)
    cfg = GPTConfig.load(ckpt / "config.json")
    model = MiniGPT(cfg)
    model.load_weights(str(ckpt / "model.safetensors"))
    model.eval()  # Dropout を切る。忘れると返答が毎回ぶれる。
    mx.eval(model.parameters())
    return model, tokenizer


def _apply_repetition_penalty(logits: mx.array, recent: list[int], penalty: float) -> mx.array:
    if penalty == 1.0 or not recent:
        return logits
    idx = mx.array(list(set(recent)))
    vals = logits[idx]
    logits[idx] = mx.where(vals > 0, vals / penalty, vals * penalty)
    return logits


def _sample(logits: mx.array, temperature: float, top_k: int) -> mx.array:
    if temperature <= 0:
        return mx.argmax(logits)
    logits = logits * (1.0 / temperature)
    if top_k and 0 < top_k < logits.size:
        # 上位k個より小さいロジットを -inf にして選択肢から外す。
        threshold = mx.min(mx.topk(logits, top_k))
        logits = mx.where(logits < threshold, -float("inf"), logits)
    return mx.random.categorical(logits)


def generate_stream(
    model: MiniGPT,
    prompt_ids: list[int],
    max_new_tokens: int = DEFAULTS["max_new_tokens"],
    temperature: float = DEFAULTS["temperature"],
    top_k: int = DEFAULTS["top_k"],
    repetition_penalty: float = DEFAULTS["repetition_penalty"],
    stop_ids: tuple[int, ...] = (),
    penalty_window: int = 64,
) -> Iterator[int]:
    ids = list(prompt_ids)
    block = model.cfg.block_size
    generated: list[int] = []
    for _ in range(max_new_tokens):
        context = mx.array(ids[-block:])[None]
        logits = model(context)[0, -1]
        logits = _apply_repetition_penalty(
            logits, generated[-penalty_window:], repetition_penalty
        )
        next_id = int(_sample(logits, temperature, top_k).item())
        if next_id in stop_ids:
            return
        ids.append(next_id)
        generated.append(next_id)
        yield next_id


def build_chat_prompt(
    tokenizer: Tokenizer,
    history: list[tuple[str, str]],
    user_text: str,
    block_size: int,
) -> list[int]:
    """会話履歴を1本のトークン列にする.

    学習コーパスと完全に同じ並び (<|user|>...<|assistant|>...<|end|>) にすることが重要。
    ここが1文字でも違うとモデルは「知らない書式」として扱い、返答が崩れる。
    """
    ids: list[int] = []
    for past_user, past_bot in history:
        ids += tokenizer.encode(f"{USER}{past_user}{ASSISTANT}{past_bot}{END}")
    ids += tokenizer.encode(f"{USER}{user_text}{ASSISTANT}")
    # 文脈長を超えたら古い方から切る。block_size が短いので会話はすぐ忘れる。
    return ids[-block_size:]


def decode_incrementally(tokenizer: Tokenizer, token_ids: Iterator[int]) -> Iterator[str]:
    """トークン列を、表示できるようになった端から文字列として流す.

    1トークンずつ独立に復号してはいけない。サブワードは byte_fallback で
    語彙にない文字を1バイトずつのトークンに分解するので、
    「🐱」は4トークンになる。1つずつ復号すると各バイトが不正な UTF-8 として
    扱われ、`����` と化ける。

    そこで累積したトークン列を毎回まとめて復号し、前回からの差分だけを出す。
    末尾が U+FFFD (置換文字) のときは、まだ途中のバイト列なので次を待つ。
    """
    buffer: list[int] = []
    shown = ""
    for token_id in token_ids:
        buffer.append(token_id)
        text = tokenizer.decode(buffer)
        if text.endswith("\ufffd"):
            continue
        if len(text) > len(shown):
            yield text[len(shown) :]
            shown = text


def chat_stream(
    model: MiniGPT,
    tokenizer: Tokenizer,
    history: list[tuple[str, str]],
    user_text: str,
    **kwargs,
) -> Iterator[str]:
    prompt = build_chat_prompt(tokenizer, history, user_text, model.cfg.block_size)
    stop_ids = (tokenizer.end_id, tokenizer.user_id)
    token_ids = generate_stream(model, prompt, stop_ids=stop_ids, **kwargs)
    yield from decode_incrementally(tokenizer, token_ids)


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/final")
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--temperature", type=float, default=DEFAULTS["temperature"])
    ap.add_argument("--top-k", type=int, default=DEFAULTS["top_k"])
    ap.add_argument("--repetition-penalty", type=float, default=DEFAULTS["repetition_penalty"])
    ap.add_argument("--max-new-tokens", type=int, default=DEFAULTS["max_new_tokens"])
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()

    if args.seed is not None:
        mx.random.seed(args.seed)
    model, tokenizer = load_bundle(args.ckpt)
    print(f"入力: {args.prompt}")
    print("出力: ", end="", flush=True)
    for piece in chat_stream(
        model,
        tokenizer,
        [],
        args.prompt,
        temperature=args.temperature,
        top_k=args.top_k,
        repetition_penalty=args.repetition_penalty,
        max_new_tokens=args.max_new_tokens,
    ):
        print(piece, end="", flush=True)
    print()


if __name__ == "__main__":
    main()
