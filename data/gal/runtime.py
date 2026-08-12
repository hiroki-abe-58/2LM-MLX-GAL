"""ローカルLLMを回すときの安全装置.

一度カーネルパニックでMacを落としてから足したもの。落ちたときのログはこう:

    panic(cpu 9): IOGPUGroupMemory.cpp:220 Assertion failed:
    result != kIOReturnSuccess
    Panicked task ...: pid 3625: python3.11

22Bのモデルをバッチ24で回したときに出た。Apple Silicon はユニファイドメモリなので
GPU の割り当ても本体RAMから取る。MLX は既定で上限を持たず、要求が通らなくなると
GPU ドライバ (IOGPUFamily) の側が確保失敗をアサーションで扱ってカーネルごと落とす。
つまりアプリの例外にならずマシンが落ちる。だからアプリ側で先に止める。

やっていることは3つ。

  1. MLX に明示的な上限を与える。超えたら Python の例外になり、カーネルまで行かない
  2. 開始前にスワップと空きメモリを見て、余裕がなければ走らせない
  3. バッチごとにピーク使用量を測り、上限に近づいたら止める
"""

from __future__ import annotations

import subprocess

import mlx.core as mx

GB = 2**30


def device_summary() -> dict:
    info = mx.device_info()
    return {
        "name": info["device_name"],
        "memory_gb": info["memory_size"] / GB,
        # GPU が快適に使える上限。物理メモリより小さい値が返る
        "working_set_gb": info["max_recommended_working_set_size"] / GB,
        "max_buffer_gb": info["max_buffer_length"] / GB,
    }


def swap_used_gb() -> float:
    """使用中のスワップ量をGBで返す. 取れなければ 0 を返す."""
    try:
        out = subprocess.run(
            ["sysctl", "-n", "vm.swapusage"], capture_output=True, text=True, timeout=5
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return 0.0
    # 例: total = 1024.00M  used = 12.25M  free = 1011.75M  (encrypted)
    parts = out.replace("=", " ").split()
    for i, token in enumerate(parts):
        if token == "used" and i + 1 < len(parts):
            value = parts[i + 1]
            scale = {"M": 1 / 1024, "G": 1.0, "K": 1 / 1024**2}.get(value[-1], 0)
            try:
                return float(value[:-1]) * scale
            except ValueError:
                return 0.0
    return 0.0


def configure(limit_gb: float, cache_gb: float = 4.0) -> None:
    """MLX に使用量の上限を与える.

    上限は「GPUが快適に使える上限」よりさらに下に置く。ここを物理メモリいっぱいに
    しても意味はなく、超えた瞬間に落ちる相手はアプリではなくカーネルなので、
    余裕を持たせる側に倒す。
    """
    device = device_summary()
    ceiling = device["working_set_gb"] * 0.75
    if limit_gb > ceiling:
        print(f"  上限 {limit_gb:.0f}GB は高すぎるので {ceiling:.0f}GB に下げます")
        limit_gb = ceiling
    mx.set_memory_limit(int(limit_gb * GB))
    mx.set_cache_limit(int(cache_gb * GB))
    print(
        f"  {device['name']} / 実装 {device['memory_gb']:.0f}GB / "
        f"GPU推奨上限 {device['working_set_gb']:.0f}GB"
    )
    print(f"  MLXの上限を {limit_gb:.0f}GB、キャッシュを {cache_gb:.0f}GB に設定")


def preflight(required_gb: float, max_swap_gb: float = 8.0) -> None:
    """走らせて安全かを開始前に確かめる. 危なければ例外で止める."""
    device = device_summary()
    if device["memory_gb"] < required_gb:
        raise SystemExit(
            f"実装メモリ {device['memory_gb']:.0f}GB では足りません "
            f"(このモデルには {required_gb:.0f}GB 必要)。"
            "--model に小さいモデルを指定してください。"
        )
    swap = swap_used_gb()
    print(f"  スワップ使用量: {swap:.1f}GB")
    if swap > max_swap_gb:
        raise SystemExit(
            f"スワップを {swap:.1f}GB 使っています。この状態でGPUメモリを大量に要求すると\n"
            "カーネルパニックでmacOSごと落ちることがあります。\n"
            "他のアプリを終了するか、再起動してから実行してください。"
        )


def kv_bytes_per_token(model) -> int | None:
    """1トークンぶんのKVキャッシュが何バイトになるかを、モデルの構成から見積もる.

    バッチ生成で足りなくなるのは重みではなくKVキャッシュ側で、その大きさは
    パラメータ数ではなく KVヘッド数で決まる。GQA を持たない世代のモデルは
    KVヘッドがアテンションヘッドと同数あり、同じ規模でも一桁重くなる。

    実測した例:
      calm3-22b-chat  (48層 / KVヘッド48)  1トークン 1,152KB
      Qwen2.5-32B     (64層 / KVヘッド 8)  1トークン   256KB

    パラメータ数は 32B のほうが多いのに、KVは4分の1以下になる。
    バッチを増やせるかどうかは、ここだけで決まる。
    """
    args = getattr(model, "args", None)
    if args is None:
        return None
    layers = getattr(args, "num_hidden_layers", None)
    heads = getattr(args, "num_attention_heads", None)
    if not layers or not heads:
        return None
    kv_heads = getattr(args, "num_key_value_heads", None) or heads
    hidden = getattr(args, "hidden_size", None)
    head_dim = getattr(args, "head_dim", None) or (hidden // heads if hidden else None)
    if not head_dim:
        return None
    return 2 * layers * kv_heads * head_dim * 2  # key と value、fp16


def kv_report(model, batch_size: int, context_tokens: int, limit_gb: float) -> None:
    per_token = kv_bytes_per_token(model)
    if per_token is None:
        print("  KVキャッシュの見積もりはできませんでした")
        return
    weights_gb = mx.get_active_memory() / GB
    kv_gb = batch_size * context_tokens * per_token / GB
    print(
        f"  KVキャッシュ: 1トークン {per_token / 1024:.0f}KB / "
        f"{batch_size}本 x {context_tokens}トークンで {kv_gb:.1f}GB"
    )
    print(f"  見込み合計: 重み {weights_gb:.1f}GB + KV {kv_gb:.1f}GB = {weights_gb + kv_gb:.1f}GB")
    if weights_gb + kv_gb > limit_gb * 0.7:
        advised = max(1, int(limit_gb * 0.5 * GB / (context_tokens * per_token)))
        print(
            f"  ※ 上限 {limit_gb:.0f}GB に対して余裕がありません。"
            f"--batch-size {advised} 以下を勧めます"
        )


class MemoryGuard:
    """バッチごとにピーク使用量を見て、上限に近づいたら止める."""

    def __init__(self, limit_gb: float, headroom: float = 0.85) -> None:
        self.threshold_gb = limit_gb * headroom
        self.peak_gb = 0.0

    def check(self) -> None:
        peak = mx.get_peak_memory() / GB
        self.peak_gb = max(self.peak_gb, peak)
        if peak > self.threshold_gb:
            raise SystemExit(
                f"GPUメモリのピークが {peak:.1f}GB に達しました "
                f"(危険域 {self.threshold_gb:.1f}GB)。\n"
                "--batch-size を半分にして再実行してください。"
                "生成済みのぶんは保存されているので、そのまま再開できます。"
            )

    def release(self) -> None:
        """バッチの区切りでキャッシュを返す. 断片化と積み上がりを防ぐ."""
        mx.clear_cache()
        mx.reset_peak_memory()
