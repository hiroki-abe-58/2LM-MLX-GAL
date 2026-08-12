"""会話コーパスを作る.

公開データセットから「ユーザーの発言 → アシスタントの返答」のペアを取り出し、
1行1会話のテキストファイルに整形する。

    <|user|>おすすめの本はありますか？<|assistant|>SFがお好きなら...<|end|>

使えるデータセットは SOURCES に登録してある。**すべて Apache-2.0 で、
ShareAlike (継承) 条件を持たないものだけ**を選んである。学習した重みを
Apache-2.0 相当で配布する前提のため、CC BY-SA のデータは意図的に入れていない
(継承条件が重みに及ぶかが不明で、配布ライセンスの整合が崩れるリスクがある)。

data/raw/ に自分のデータを置けば、そのまま混ぜたり、
--no-hf を付けて自分のデータだけで学習させることもできる。
対応形式は次の2つ。

    data/raw/mydata.jsonl : {"user": "...", "assistant": "..."} を1行1件
    data/raw/mydata.tsv   : ユーザー発言<TAB>アシスタント返答 を1行1件

使い方:
    python data/prepare.py                          # 登録済みデータセットを全部使う
    python data/prepare.py --sources oasst1         # 1LM (前作) と同じ1件だけ使う
    python data/prepare.py --exclude eval/holdout.txt  # 固定検証セットを訓練から除く
    python data/prepare.py --no-hf                  # data/raw/ のデータだけ使う
    python data/prepare.py --list-sources           # 登録済みデータセットと出所を表示
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.tokenizer import ASSISTANT, END, USER  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"

_WS_RE = re.compile(r"\s+")


def normalize(text: str) -> str:
    """空白と改行をすべて半角スペース1個に潰し、1会話=1行を保証する.

    箇条書きの改行は失われるが、そのぶん小さなモデルには学習しやすい形になる。
    """
    return _WS_RE.sub(" ", text).strip()


# --- データセットごとの読み込み --------------------------------------------


def load_oasst1() -> list[tuple[str, str]]:
    """kunishou/oasst1-89k-ja. 1メッセージ1レコードの木構造から対を復元する."""
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(
        "kunishou/oasst1-89k-ja", "oasst1_89k_ja_20231027.json", repo_type="dataset"
    )
    messages = json.loads(Path(path).read_text(encoding="utf-8"))
    # parent_id の欠損は None ではなく文字列 "nan" で入っているので get() で拾う。
    by_id = {m["message_id"]: m for m in messages}

    pairs: list[tuple[str, str]] = []
    for msg in messages:
        if msg["role"] != "assistant":
            continue
        parent = by_id.get(msg["parent_id"])
        if parent is None or parent["role"] != "prompter":
            continue
        # ng_translation=="1" は翻訳が破綻していると報告されている行。
        if msg["ng_translation"] == "1" or parent["ng_translation"] == "1":
            continue
        q, a = normalize(parent["text_ja"] or ""), normalize(msg["text_ja"] or "")
        if q and a:
            pairs.append((q, a))
    return pairs


def _pairs_from_turns(turns: Iterable[dict]) -> list[tuple[str, str]]:
    """role/content の並びから、隣り合う user→assistant を対にして取り出す."""
    pairs: list[tuple[str, str]] = []
    pending: str | None = None
    for turn in turns:
        role, content = turn.get("role"), normalize(turn.get("content") or "")
        if not content:
            pending = None
            continue
        if role == "user":
            pending = content
        elif role == "assistant" and pending:
            pairs.append((pending, content))
            pending = None
    return pairs


def _load_messages_dataset(repo: str, field: str) -> list[tuple[str, str]]:
    """conversations / messages 形式のデータセットを共通の手順で読む.

    oasst2-33k-ja、magpie-sft-v1.0、Magpie-Tanuki-8B-97k は
    フィールド名が違うだけで中身は同じ role/content の配列なので、
    アダプタを1本にまとめてある。
    """
    from datasets import load_dataset

    ds = load_dataset(repo, split="train")
    pairs: list[tuple[str, str]] = []
    for row in ds:
        pairs += _pairs_from_turns(row[field])
    return pairs


@dataclass(frozen=True)
class Source:
    key: str
    repo: str
    license: str
    note: str
    loader: Callable[[], list[tuple[str, str]]]


SOURCES: tuple[Source, ...] = (
    Source(
        key="oasst1",
        repo="kunishou/oasst1-89k-ja",
        license="Apache-2.0",
        note="OpenAssistant/oasst1 の機械翻訳。ng_translation で訳崩れを除外",
        loader=load_oasst1,
    ),
    Source(
        key="oasst2",
        repo="llm-jp/oasst2-33k-ja",
        license="Apache-2.0",
        note="OpenAssistant/oasst2 のDeepL翻訳。訳崩れフラグは無いので自前で弾く",
        loader=lambda: _load_messages_dataset("llm-jp/oasst2-33k-ja", "conversations"),
    ),
    Source(
        key="magpie",
        repo="llm-jp/magpie-sft-v1.0",
        license="Apache-2.0",
        note="最初から日本語で作られた合成データ。翻訳由来の崩れが無い",
        loader=lambda: _load_messages_dataset("llm-jp/magpie-sft-v1.0", "conversations"),
    ),
    Source(
        key="tanuki",
        repo="Aratako/Magpie-Tanuki-8B-97k",
        license="Apache-2.0",
        note="Tanuki-8B による合成データ。品質フィルタ未実施と明記されている",
        loader=lambda: _load_messages_dataset("Aratako/Magpie-Tanuki-8B-97k", "messages"),
    ),
)

SOURCES_BY_KEY = {s.key: s for s in SOURCES}


def resolve_sources(spec: str) -> list[Source]:
    if spec == "all":
        return list(SOURCES)
    keys = [k.strip() for k in spec.split(",") if k.strip()]
    unknown = [k for k in keys if k not in SOURCES_BY_KEY]
    if unknown:
        raise SystemExit(
            f"知らないデータセット: {', '.join(unknown)}\n"
            f"使えるのは: {', '.join(SOURCES_BY_KEY)} または all"
        )
    return [SOURCES_BY_KEY[k] for k in keys]


def load_local_pairs(raw_dir: Path = RAW_DIR) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    if not raw_dir.exists():
        return pairs
    for path in sorted(raw_dir.iterdir()):
        if path.suffix == ".jsonl":
            for line in path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                q, a = normalize(obj["user"]), normalize(obj["assistant"])
                if q and a:
                    pairs.append((q, a))
        elif path.suffix == ".tsv":
            for line in path.read_text(encoding="utf-8").splitlines():
                if "\t" not in line:
                    continue
                q, a = (normalize(x) for x in line.split("\t", 1))
                if q and a:
                    pairs.append((q, a))
    return pairs


def load_excluded(paths: list[str]) -> set[tuple[str, str]]:
    """固定検証セットに入っている会話を読み込む.

    eval/holdout.txt は「1行1会話」の整形済みコーパスなので、
    マーカーで割って (質問, 返答) に戻してから照合する。
    ここを忘れると検証セットを暗記したモデルを採点することになる。
    """
    pattern = re.compile(
        re.escape(USER) + "(.*?)" + re.escape(ASSISTANT) + "(.*?)" + re.escape(END)
    )
    excluded: set[tuple[str, str]] = set()
    for path in paths:
        text = Path(path).read_text(encoding="utf-8")
        excluded.update(pattern.findall(text))
    return excluded


def looks_broken(user: str, assistant: str) -> bool:
    """機械翻訳が壊れた行を弾く.

    oasst2-33k-ja には oasst1-89k-ja のような ng_translation 列が無いので、
    崩れ方の特徴で判定するしかない。実データで多かったのは次の2つ。

    - 同じ短い並びが何度も続く (「あなたのためにあなたのためにあなたのために」)
    - 翻訳に失敗して原文がそのまま入り、質問と返答が同一になる
    """
    if user == assistant:
        return True
    if len(assistant) < 12:
        return False
    grams = Counter(assistant[i : i + 6] for i in range(len(assistant) - 5))
    return max(grams.values()) >= 4


def drop_rare_char_pairs(
    pairs: list[tuple[str, str]], min_char_freq: int
) -> tuple[list[tuple[str, str]], int]:
    """出現回数が少ない文字を含む会話を丸ごと捨てる.

    絵文字や一度しか出てこない漢字を残すと、語彙が増えるだけで学習には寄与しない。
    未知文字を <|unk|> に置き換える手もあるが、教材としては「捨てる」方が挙動が読みやすい。
    """
    if min_char_freq <= 1:
        return pairs, 0
    counts: Counter[str] = Counter()
    for q, a in pairs:
        counts.update(q)
        counts.update(a)
    rare = {c for c, n in counts.items() if n < min_char_freq}
    kept = [(q, a) for q, a in pairs if not (rare & set(q)) and not (rare & set(a))]
    return kept, len(rare)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "data" / "corpus.txt"))
    ap.add_argument("--max-q", type=int, default=100, help="ユーザー発言の最大文字数")
    ap.add_argument("--max-a", type=int, default=300, help="アシスタント返答の最大文字数")
    ap.add_argument("--min-a", type=int, default=4, help="アシスタント返答の最小文字数")
    ap.add_argument("--min-char-freq", type=int, default=10)
    ap.add_argument(
        "--sources",
        default="all",
        help="使う公開データセット (カンマ区切り、または all)。--list-sources で一覧",
    )
    ap.add_argument("--no-hf", action="store_true", help="公開データセットを使わない")
    ap.add_argument("--no-local", action="store_true", help="data/raw/ の自前データを使わない")
    ap.add_argument("--raw-dir", default=str(RAW_DIR), help="自前データの置き場を差し替える")
    ap.add_argument("--list-sources", action="store_true", help="登録済みデータセットを表示")
    ap.add_argument(
        "--exclude",
        action="append",
        default=[],
        help="訓練から除く会話のファイル (eval/holdout.txt など). 複数指定可",
    )
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if args.list_sources:
        print("登録済みデータセット (すべて Apache-2.0 / ShareAlike なし)\n")
        for source in SOURCES:
            print(f"  {source.key:<8} {source.repo}")
            print(f"           {source.license} — {source.note}")
        return

    pairs: list[tuple[str, str]] = []
    if not args.no_hf:
        for source in resolve_sources(args.sources):
            loaded = source.loader()
            print(f"{source.repo:<34}: {len(loaded)} 会話")
            pairs += loaded
    local = [] if args.no_local else load_local_pairs(Path(args.raw_dir))
    if local:
        print(f"{args.raw_dir + '/':<34}: {len(local)} 会話")
    pairs += local
    if not pairs:
        raise SystemExit("会話が0件です。--no-hf を外すか data/raw/ にデータを置いてください。")
    print(f"{'合計':<34}: {len(pairs)} 会話\n")

    pairs = [
        (q, a)
        for q, a in pairs
        if len(q) <= args.max_q and args.min_a <= len(a) <= args.max_a
    ]
    print(f"長さフィルタ後   : {len(pairs)} 会話")

    seen: set[tuple[str, str]] = set()
    deduped = []
    for pair in pairs:
        if pair in seen:
            continue
        seen.add(pair)
        deduped.append(pair)
    print(f"重複除去後       : {len(deduped)} 会話")

    before = len(deduped)
    deduped = [(q, a) for q, a in deduped if not looks_broken(q, a)]
    print(f"訳崩れ除去後     : {len(deduped)} 会話 (除いた: {before - len(deduped)})")

    if args.exclude:
        excluded = load_excluded(args.exclude)
        before = len(deduped)
        deduped = [pair for pair in deduped if pair not in excluded]
        print(f"検証セット除外後 : {len(deduped)} 会話 (除いた: {before - len(deduped)})")

    kept, n_rare = drop_rare_char_pairs(deduped, args.min_char_freq)
    print(f"低頻度文字除去後 : {len(kept)} 会話 (捨てた文字種: {n_rare})")

    import random

    random.Random(args.seed).shuffle(kept)

    lines = [f"{USER}{q}{ASSISTANT}{a}{END}" for q, a in kept]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")

    vocab: Counter[str] = Counter()
    for q, a in kept:
        vocab.update(q)
        vocab.update(a)
    n_chars = sum(len(q) + len(a) for q, a in kept)
    meta = {
        "conversations": len(kept),
        "chars": n_chars,
        "unique_chars": len(vocab),
        "avg_user_len": round(sum(len(q) for q, _ in kept) / len(kept), 1),
        "avg_assistant_len": round(sum(len(a) for _, a in kept) / len(kept), 1),
    }
    out.with_suffix(".meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\n書き出し: {out}")
    for k, v in meta.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
