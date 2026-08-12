"""複数の学習ログを重ねて比べる (対照実験の集計用).

    python tools/compare_runs.py runs/exp/loss_a.csv:405万文字 runs/exp/loss_b.csv:949万文字

`パス:表示名` の形で渡す。表示名は省略できる。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.loss_log import Curve, read_curve  # noqa: E402


def parse_spec(spec: str) -> Curve:
    path, _, name = spec.partition(":")
    return read_curve(path, name or None)


def report(curves: list[Curve]) -> None:
    width = max(len(c.name) for c in curves)
    print(f"{'run':<{width}}  {'最終train':>9}  {'最良val':>8}  {'最良step':>8}  {'乖離点':>7}")
    print("-" * (width + 40))
    for c in curves:
        best_step, best_val = c.best_val
        # 0 は「最後まで追い越されなかった」= 予算内では過学習していない、の意味。
        onset = str(c.divergence_step) if c.divergence_step else "なし"
        print(
            f"{c.name:<{width}}  {c.final_train:>9.3f}  {best_val:>8.3f}"
            f"  {best_step:>8d}  {onset:>7}"
        )
    print("\n乖離点 = val loss が train loss を追い越した最初の step。ここから先は暗記が優勢になる。")


def plot(curves: list[Curve], out: Path) -> None:
    """上段に損失、下段に汎化ギャップを描く.

    コーパスが違う run どうしでは val loss の水準そのものは比較できない
    (検証に使う文章が run ごとに違うため)。比べられるのは形、とくに
    「train を追い越すまでに何ステップ持ちこたえたか」なので下段を主役にする。
    """
    from tools.plotting import SERIES_COLORS, dark_figure

    fig, (top, bottom) = dark_figure(2, figsize=(8, 6.4))

    for curve, color in zip(curves, SERIES_COLORS, strict=False):
        top.plot(
            [s for s, _ in curve.train],
            [v for _, v in curve.train],
            color=color, lw=1.2, alpha=0.5, ls="--",
        )
        top.plot(
            [s for s, _ in curve.val],
            [v for _, v in curve.val],
            color=color, lw=1.8, marker="o", ms=3, label=curve.name,
        )
        bottom.plot(
            [s for s, _ in curve.gaps],
            [g for _, g in curve.gaps],
            color=color, lw=1.8, marker="o", ms=3,
        )
        if curve.divergence_step:
            bottom.axvline(curve.divergence_step, color=color, lw=1.0, ls=":", alpha=0.8)
            bottom.annotate(
                f"step {curve.divergence_step}",
                xy=(curve.divergence_step, 0), xytext=(6, -14),
                textcoords="offset points", color=color, fontsize=9,
            )

    top.set_ylabel("cross entropy (nats / char)")
    top.set_title("破線 = train / 実線 = val", loc="left", fontsize=10)
    top.legend(frameon=False, fontsize=9)

    bottom.axhline(0, color="#ffffff", lw=1.0, alpha=0.45)
    bottom.set_xlabel("step")
    bottom.set_ylabel("val - train")
    bottom.set_title("汎化ギャップ。0 を上抜けた地点から暗記が優勢になる", loc="left", fontsize=10)

    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, facecolor=fig.get_facecolor())
    print(f"\n保存: {out}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", help="loss.csv のパス (パス:表示名 の形も可)")
    ap.add_argument("--out", default="", help="重ねた図の保存先 (省略すると作らない)")
    args = ap.parse_args()

    curves = [parse_spec(spec) for spec in args.runs]
    report(curves)
    if args.out:
        plot(curves, Path(args.out))


if __name__ == "__main__":
    main()
