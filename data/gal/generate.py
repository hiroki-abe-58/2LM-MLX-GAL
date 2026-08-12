"""ローカルLLMに架空の「ギャルのLINE」会話を書かせてコーパスを作る.

生成には Apache-2.0 のローカルモデル (既定 calm3-22b-chat) だけを使う。
このモデルは、本プロジェクトが既に学習に使っている llm-jp/magpie-sft-v1.0 を
作ったモデル本人でもある。手元で完結するので、出力の扱いに制約が付かない。

3段階に分かれている。

  topics : ジャンルだけ与えて、具体的な話題をモデルに列挙させる
  pairs  : 話題 x 用件 x 機嫌 の組み合わせごとに往復を書かせ、生のまま追記する
  build  : 生の出力を検査してふるいにかけ、学習用の1本にまとめる

日本語の中身をこちらで書かないのは意図的で、話題の語彙まで人間が用意すると
そこが多様性の上限になる。ジャンルという骨組みだけ決めて、肉は全部モデルに付けさせる。

pairs は1バッチごとに追記していく。途中で止まっても、もう一度同じコマンドを
叩けば済んだ組み合わせを飛ばして続きから走る。長時間まわす前提なので、
落ちても失わない形にしてある。

使い方:
    python data/gal/generate.py --stage topics
    python data/gal/generate.py --stage pairs --target 12000
    python data/gal/generate.py --stage build
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
from itertools import product
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import runtime  # noqa: E402
from validate import filter_pairs, report  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
TOPICS_PATH = HERE / "topics.txt"
# 検査前の生の出力。data/raw/ に置くと prepare.py が未検査のまま拾ってしまうので、
# 意図的にこちら側へ置いている。
RAW_PATH = HERE / "raw.jsonl"
DEFAULT_OUT = ROOT / "data" / "raw" / "gal_line.jsonl"

# GQA を持つ世代を選ぶこと。KVヘッドが多いモデルはバッチを増やせず、実用速度が出ない。
# 詳しくは runtime.kv_bytes_per_token のコメント。
DEFAULT_MODEL = "mlx-community/Qwen2.5-32B-Instruct-4bit"
DEFAULT_MEMORY_LIMIT_GB = 30.0

# 文体は「仕様」として渡す。例文を書いて渡すと、モデルはそれを言い換えるだけになり、
# 手本の数だけしか語彙が広がらない。禁止事項だけ具体的に、中身は指定しない。
STYLE = """あなたは日本語のセリフを書くプロの脚本家です。
架空のキャラクター「ギャル」がLINEで返信する場面のセリフを書きます。
実在の人物とは関係のない、完全な創作です。

このキャラクターの話し方:
- 一人称は「うち」
- 敬語を使わない。友達に送るくだけた口調
- 返信は短い。40文字を超えない
- 相手を否定しない。明るくてノリがいい
- 知ったかぶりをせず、わからないことは正直にわからないと言う
- ときどきボケる。まじめに答えすぎない

絶対に守る決まり:
- 絵文字と顔文字を使わない
- 使ってよい記号は 、。！？〜… と w だけ
- 英単語を書かない。カタカナで書く
- 関西弁などの方言にしない。標準語のくだけた話し方で"""

# ジャンルだけを決める。ここから先の具体的な話題はモデルに出させる。
CATEGORIES = (
    "食べ物と飲み物", "学校と勉強", "アルバイト", "恋愛", "友達づきあい",
    "お金", "天気と季節", "旅行とおでかけ", "美容とファッション", "体調と健康",
    "音楽と映画", "ゲームとアニメ", "スマホとインターネット", "家族", "家事と生活",
    "仕事と将来", "スポーツと運動", "動物とペット", "科学と技術", "世の中のできごと",
)

# ユーザー側が何をしてくるか。会話の型が偏らないようにする。
INTENTS = (
    "質問する", "感想を言う", "愚痴をこぼす", "誘う", "報告する",
    "相談する", "頼みごとをする", "あいさつする",
)

# 返す側の機嫌。同じ話題でも返しの温度が変わる。
MOODS = ("機嫌がいい", "眠くてだるい", "テンションが高い", "ちょっと呆れている", "腹をすかせている")

_TOPIC_LINE_RE = re.compile(r"^\s*(?:[-*・]|\d+[.、)）])?\s*(.+?)\s*$")

# 1行に1往復を「｜」で区切って書かせる。本文に出ない全角記号を選んでいる。
SEPARATOR = "｜"
# モデルは指示しても番号・箇条書き・話者名・かぎかっこを付けてくる。
# 何度指示を書き直しても一定の割合で混ざるので、諦めて機械的に剥がす。
_LEAD_NOISE_RE = re.compile(r"^\s*(?:[-*・]|\d+\s*[.、)）:：])?\s*")
_ROLE_RE = re.compile(r"^\s*(?:ユーザー|相手|あなた|ギャル|返信|assistant|user)\s*[:：]\s*", re.I)
_QUOTE_RE = re.compile(r"^[「『\"'（(]+|[」』\"')）]+$")
_DIGITS_ONLY_RE = re.compile(r"^\d+$")


def _clean_side(text: str) -> str:
    """1発言ぶんから、番号・話者名・かぎかっこを剥がす."""
    text = _LEAD_NOISE_RE.sub("", text.strip())
    text = _ROLE_RE.sub("", text)
    for _ in range(2):  # 「「〜」」のように重なっていることがある
        text = _QUOTE_RE.sub("", text.strip())
    return text.strip()


def parse_pairs(text: str) -> list[tuple[str, str]]:
    pairs = []
    for line in text.splitlines():
        # 区切りが2つ以上ある行は、1行に2往復を詰めたか末尾に余分を付けたか判別できない。
        # 最初の2つを採ると往復の対応がずれるので、行ごと捨てる。
        fields = line.split(SEPARATOR)
        if len(fields) != 2:
            continue
        user, assistant = (_clean_side(s) for s in fields)
        # 番号だけの行を往復として拾ってしまう事故があった。数字だけの側は捨てる。
        if not user or not assistant:
            continue
        if _DIGITS_ONLY_RE.match(user) or _DIGITS_ONLY_RE.match(assistant):
            continue
        pairs.append((user, assistant))
    return pairs


def build_prompt(tokenizer, system: str, user: str) -> list[int]:
    text = tokenizer.apply_chat_template(
        [{"role": "system", "content": system}, {"role": "user", "content": user}],
        add_generation_prompt=True,
        tokenize=False,
    )
    return tokenizer.encode(text, add_special_tokens=False)


def run_batch(
    model, tokenizer, prompts: list[list[int]], max_tokens: int, sampler, prefill: int = 2
) -> list[str]:
    from mlx_lm import batch_generate

    return batch_generate(
        model,
        tokenizer,
        prompts=prompts,
        max_tokens=max_tokens,
        sampler=sampler,
        # 既定は32件を同時に走らせる。こちらが渡した数より多く並べさせない。
        completion_batch_size=len(prompts),
        # プリフィルを何本ずつ流すか。ここを大きくすると中間活性が一気に膨らむ。
        prefill_batch_size=min(prefill, len(prompts)),
    ).texts


# --- topics: 話題出し -------------------------------------------------------


def ask_topics(model, tokenizer, sampler, per_category: int, batch_size: int, guard) -> list[str]:
    asks = [
        f"「{category}」に関係する具体的な話題を{per_category}個あげてください。\n"
        "条件: 1行に1つだけ。名詞か短い言い回しで。番号や記号は付けない。説明も書かない。"
        for category in CATEGORIES
    ]
    prompts = [build_prompt(tokenizer, "あなたは日本語の語彙に詳しい編集者です。", a) for a in asks]

    topics: list[str] = []
    for start in range(0, len(prompts), batch_size):
        chunk = prompts[start : start + batch_size]
        for text in run_batch(model, tokenizer, chunk, per_category * 16, sampler):
            for line in text.splitlines():
                match = _TOPIC_LINE_RE.match(line)
                if not match:
                    continue
                topic = match.group(1)
                # 話題として短すぎ・長すぎるものと、見出し行を落とす
                if 2 <= len(topic) <= 14 and "：" not in topic and ":" not in topic:
                    topics.append(topic)
        guard.check()
        guard.release()
        done = min(start + batch_size, len(prompts))
        print(f"  {done}/{len(prompts)} ジャンル  累計 {len(topics)} 件", flush=True)

    unique = sorted(set(topics))
    print(f"  重複を除いて {len(unique)} 話題")
    return unique


# --- pairs: 会話生成 --------------------------------------------------------


def load_done_combos() -> set[tuple[str, str, str]]:
    """すでに生成し終えた組み合わせを、生の出力から読み直す."""
    if not RAW_PATH.exists():
        return set()
    done = set()
    for line in RAW_PATH.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            # 落ちた瞬間に書きかけの行が残ることがある。読めない行は捨てる。
            continue
        done.add((obj["topic"], obj["intent"], obj["mood"]))
    return done


def count_raw() -> int:
    if not RAW_PATH.exists():
        return 0
    return sum(1 for line in RAW_PATH.read_text(encoding="utf-8").splitlines() if line.strip())


def ask_pairs(
    model, tokenizer, sampler, topics: list[str], target: int,
    per_request: int, batch_size: int, rng: random.Random, guard, prefill: int,
) -> None:
    done = load_done_combos()
    combos = [c for c in product(topics, INTENTS, MOODS) if c not in done]
    rng.shuffle(combos)
    print(f"  組み合わせ {len(combos)} 件が未処理 (済み {len(done)} 件)")

    collected = count_raw()
    already = collected
    print(f"  生の出力はすでに {collected} 件ある")
    started = time.time()

    RAW_PATH.parent.mkdir(parents=True, exist_ok=True)
    for start in range(0, len(combos), batch_size):
        if collected >= target:
            print("  目標に達したので打ち切ります")
            break
        chunk = combos[start : start + batch_size]
        prompts = [
            build_prompt(
                tokenizer,
                STYLE,
                f"話題「{topic}」について、独立したLINEのやりとりを{per_request}通り作ってください。\n"
                f"相手は{intent}。ギャルは{mood}という設定です。\n"
                "\n"
                "書き方:\n"
                f"- 1行に1往復。ぜんぶで{per_request}行\n"
                "- 相手の発言とギャルの返信を、全角の縦棒 ｜ ひとつで区切る\n"
                "- 各行は独立した別のやりとり。前の行の続きにしない\n"
                "- 相手の発言はギャル語にしない。ふつうの話し方で\n"
                "\n"
                "書かないこと:\n"
                "- 行番号、箇条書きの記号\n"
                "- 「ユーザー」「ギャル」などの話者名\n"
                "- かぎかっこ、前置き、説明",
            )
            for topic, intent, mood in chunk
        ]
        texts = run_batch(model, tokenizer, prompts, per_request * 40, sampler, prefill)

        # バッチが終わった時点で必ず書き出す。ここで落ちても失うのは1バッチだけ。
        with RAW_PATH.open("a", encoding="utf-8") as f:
            for (topic, intent, mood), text in zip(chunk, texts, strict=True):
                for user, assistant in parse_pairs(text):
                    record = {
                        "topic": topic, "intent": intent, "mood": mood,
                        "user": user, "assistant": assistant,
                    }
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
                    collected += 1

        guard.check()
        guard.release()
        elapsed = time.time() - started
        print(
            f"  {min(start + batch_size, len(combos))}/{len(combos)} 組  "
            f"生 {collected} 件  {elapsed / 60:.1f}分  "
            f"{(collected - already) / elapsed * 60:.0f} 件/分  ピーク {guard.peak_gb:.1f}GB",
            flush=True,
        )


# --- build: 検査してまとめる -------------------------------------------------


def build(out_path: Path, rng: random.Random, preview: int) -> None:
    if not RAW_PATH.exists():
        raise SystemExit(f"{RAW_PATH} がありません。先に --stage pairs を回してください。")

    raw: list[tuple[str, str]] = []
    for line in RAW_PATH.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        raw.append((obj["user"], obj["assistant"]))
    print(f"生のまま {len(raw)} 件\n")

    print("検査")
    pairs, rejections = filter_pairs(raw)
    rejections.show(len(pairs))
    print()
    report(pairs)

    if preview:
        print("\n見本")
        for user, assistant in pairs[:preview]:
            print(f"  user      : {user}")
            print(f"  assistant : {assistant}")
        return

    rng.shuffle(pairs)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for user, assistant in pairs:
            f.write(json.dumps({"user": user, "assistant": assistant}, ensure_ascii=False) + "\n")
    print(f"\n書き出し: {out_path}")


# --- 入り口 -----------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--stage", choices=("topics", "pairs", "build", "all"), default="all")
    ap.add_argument("--target", type=int, default=100000, help="生の出力を何件集めたら止めるか")
    ap.add_argument("--topics-per-category", type=int, default=30)
    ap.add_argument("--pairs-per-request", type=int, default=6)
    ap.add_argument("--batch-size", type=int, default=16, help="同時に走らせる生成数")
    ap.add_argument("--prefill-batch", type=int, default=2, help="プリフィルを何本ずつ流すか")
    ap.add_argument("--memory-limit", type=float, default=DEFAULT_MEMORY_LIMIT_GB, help="GB")
    ap.add_argument("--temp", type=float, default=0.85)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--rng-seed", type=int, default=0)
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--preview", type=int, default=0, help="書き出さずに見本を出す件数")
    args = ap.parse_args()

    rng = random.Random(args.rng_seed)

    # build だけならモデルを読まない。検査のやり直しが速い。
    if args.stage == "build":
        build(Path(args.out), rng, args.preview)
        return

    import mlx.core as mx
    from mlx_lm import load
    from mlx_lm.sample_utils import make_sampler

    print("環境の確認")
    runtime.preflight(required_gb=32.0)
    runtime.configure(args.memory_limit)
    guard = runtime.MemoryGuard(args.memory_limit)
    print()

    mx.random.seed(args.rng_seed)
    sampler = make_sampler(temp=args.temp, top_p=args.top_p)
    print(f"モデルを読み込みます: {args.model}")
    t0 = time.time()
    model, tokenizer = load(args.model)
    print(f"  {time.time() - t0:.1f}秒  常駐 {mx.get_active_memory() / 2**30:.1f}GB")
    # プロンプト約400トークン + 生成ぶんを文脈長として見積もる
    runtime.kv_report(model, args.batch_size, 400 + args.pairs_per_request * 40, args.memory_limit)
    print()

    if args.stage in ("topics", "all"):
        print("1段目: 話題出し")
        topics = ask_topics(
            model, tokenizer, sampler, args.topics_per_category, args.batch_size, guard
        )
        TOPICS_PATH.write_text("\n".join(topics) + "\n", encoding="utf-8")
        print(f"  書き出し: {TOPICS_PATH}\n")
        if args.stage == "topics":
            return
    else:
        topics = [t for t in TOPICS_PATH.read_text(encoding="utf-8").splitlines() if t.strip()]
        print(f"話題を読み込みました: {len(topics)} 件\n")

    print("2段目: 会話生成")
    ask_pairs(
        model, tokenizer, sampler, topics, args.target,
        args.pairs_per_request, args.batch_size, rng, guard, args.prefill_batch,
    )

    if args.stage == "all":
        print("\n3段目: 検査してまとめる")
        build(Path(args.out), rng, args.preview)


if __name__ == "__main__":
    main()
