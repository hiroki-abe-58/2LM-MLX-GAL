"""生成した会話を検査してふるいにかける.

テンプレートで作っていた頃は、想定外の文字が出たら「テンプレートの書き間違い」
だったので、見つけ次第そこで止めるのが正しかった。LLM に書かせる今は事情が違う。
モデルは一定の割合で必ず変なものを出すので、止めるのではなく捨てて、
どれくらい捨てたかを数える。捨てた率そのものが生成条件の良し悪しを示す。
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter
from dataclasses import dataclass, field

# コーパスに出てよい文字。ここに無い文字を含む会話は捨てる。
# 文字レベルでも語彙 8k のサブワードでも、1回しか出ない文字は
# 語彙を1つ潰すだけの汚れにしかならない。
_ALLOWED_ASCII = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 ")
_ALLOWED_SYMBOLS = set("、。！？〜ー・（）「」…々w笑")
_ALLOWED_BLOCKS = ("HIRAGANA", "KATAKANA", "CJK UNIFIED")

# 弾く前に直せるものは直す。三点リーダや全角チルダのような「表記の揺れ」で
# 会話を捨てていると、実測では棄却の4割がこれだけで消えていた。
# ただし NFKC は使わない。！ や ？ が半角に化けて、公開データ側の表記と食い違う。
_NORMALIZE = str.maketrans(
    {
        "～": "〜",  # FULLWIDTH TILDE -> WAVE DASH
        "〝": "「", "〟": "」",
        "･": "・",
        "‥": "…",
        **{chr(0xFF10 + i): chr(0x30 + i) for i in range(10)},  # 全角数字 -> 半角
        **{chr(0xFF21 + i): chr(0x41 + i) for i in range(26)},  # 全角英大 -> 半角
        **{chr(0xFF41 + i): chr(0x61 + i) for i in range(26)},  # 全角英小 -> 半角
    }
)

# 方言が混ざると人格がぶれる。ギャル語と広く共用されている「やん」は残し、
# 明らかに関西弁の語尾だけを落とす。
_DIALECT_RE = re.compile(r"(ねん|やけど|あかん|ほんま|せや|へんわ|とちゃう)")

# ラテン文字を全面禁止にはできない (AI や SNS が本文に出る) が、野放しにすると
# 英単語がそのまま紛れ込む。使ってよい綴りを列挙し、それ以外を含む会話は捨てる。
_ALLOWED_WORDS = frozenset(
    {
        "w", "ww", "www", "wwww",
        "AI", "API", "DNA", "GPU", "CPU", "PC", "SNS", "LINE", "URL",
        "NISA", "ATM", "TV", "DVD", "CD", "IT", "OK", "NG",
        "Python", "ChatGPT", "GPT", "YouTube", "X",
    }
)
_ASCII_WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9]*")

# 返答が敬語だとキャラが壊れる。文末の丁寧形だけを見る。
_POLITE_TAIL_RE = re.compile(r"(です|ます|ました|ません|でしょう|ください|ございます)[。！？〜\s]*$")

# 同じ文字が続きすぎるもの (「あああああ」など) を弾く。
_RUN_RE = re.compile(r"(.)\1{4,}")

USER_LEN = (2, 32)
ASSISTANT_LEN = (4, 64)


@dataclass
class Rejections:
    """何を理由に何件捨てたかの集計."""

    counts: Counter[str] = field(default_factory=Counter)
    samples: dict[str, str] = field(default_factory=dict)

    def add(self, reason: str, text: str) -> None:
        self.counts[reason] += 1
        self.samples.setdefault(reason, text)

    @property
    def total(self) -> int:
        return sum(self.counts.values())

    def show(self, kept: int) -> None:
        if not self.counts:
            print("  棄却: なし")
            return
        seen = kept + self.total
        print(f"  棄却: {self.total} / {seen} 件 ({self.total / seen:.1%})")
        for reason, count in self.counts.most_common():
            print(f"    {reason:<16} {count:>6} 件   例: {self.samples[reason][:38]}")


def is_allowed(char: str) -> bool:
    if char in _ALLOWED_ASCII or char in _ALLOWED_SYMBOLS:
        return True
    try:
        name = unicodedata.name(char)
    except ValueError:
        return False
    return any(name.startswith(block) for block in _ALLOWED_BLOCKS)


def bad_characters(text: str) -> list[str]:
    return [c for c in text if not is_allowed(c)]


def bad_words(text: str) -> list[str]:
    return [w for w in _ASCII_WORD_RE.findall(text) if w not in _ALLOWED_WORDS]


def normalize(text: str) -> str:
    return text.translate(_NORMALIZE).strip()


def judge(user: str, assistant: str) -> str | None:
    """捨てる理由を返す. 通ったら None."""
    if not (USER_LEN[0] <= len(user) <= USER_LEN[1]):
        return "ユーザー長"
    if not (ASSISTANT_LEN[0] <= len(assistant) <= ASSISTANT_LEN[1]):
        return "返答長"
    if user == assistant:
        return "同一文"
    for text in (user, assistant):
        if "\n" in text or "\t" in text:
            return "改行混入"
        if bad_characters(text):
            return "文字種"
        if bad_words(text):
            return "英単語"
        if _RUN_RE.search(text):
            return "同字連続"
    if _POLITE_TAIL_RE.search(assistant):
        return "敬語"
    if _DIALECT_RE.search(assistant):
        return "方言"
    return None


def filter_pairs(pairs: list[tuple[str, str]]) -> tuple[list[tuple[str, str]], Rejections]:
    """検査と重複除去をまとめて行う.

    重複は (user, assistant) の完全一致に加えて、返答文だけの一致も落とす。
    LLM は違う話題を振っても同じ返しを書いてくることがあり、それを残すと
    小さいモデルはその1文だけを覚えてしまう。
    """
    rejections = Rejections()
    seen_pairs: set[tuple[str, str]] = set()
    seen_assistant: set[str] = set()
    kept: list[tuple[str, str]] = []
    for user, assistant in pairs:
        user, assistant = normalize(user), normalize(assistant)
        reason = judge(user, assistant)
        if reason is not None:
            rejections.add(reason, f"{user} / {assistant}")
            continue
        if (user, assistant) in seen_pairs:
            rejections.add("重複", f"{user} / {assistant}")
            continue
        if assistant in seen_assistant:
            rejections.add("返答重複", f"{user} / {assistant}")
            continue
        seen_pairs.add((user, assistant))
        seen_assistant.add(assistant)
        kept.append((user, assistant))
    return kept, rejections


def report(pairs: list[tuple[str, str]]) -> None:
    if not pairs:
        print("  会話が1件もありません")
        return
    chars: Counter[str] = Counter()
    for user, assistant in pairs:
        chars.update(user)
        chars.update(assistant)
    total = sum(len(u) + len(a) for u, a in pairs)
    print(f"  会話数        : {len(pairs)}")
    print(f"  総文字数      : {total}")
    print(f"  文字種        : {len(chars)}")
    print(f"  平均ユーザー長: {sum(len(u) for u, _ in pairs) / len(pairs):.1f}")
    print(f"  平均返答長    : {sum(len(a) for _, a in pairs) / len(pairs):.1f}")
    rare = [c for c, n in chars.items() if n < 10]
    if rare:
        print(f"  出現10回未満の文字: {len(rare)} 種類 -> {''.join(sorted(rare))}")
        print("  ※ data/prepare.py の低頻度文字フィルタで、これを含む会話は捨てられます")
