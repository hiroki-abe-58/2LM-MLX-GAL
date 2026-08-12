"""runs/*.csv の学習ログを読んで、曲線と診断値を返す.

train_loss と val_loss は別の行に書き出されるので、step で突き合わせる必要がある。
plot_loss.py と compare_runs.py の両方から使う。
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path


def keep_last_run(points: list[tuple[int, float]]) -> list[tuple[int, float]]:
    """step が単調増加になるように、末尾から遡って行を残す.

    学習を二重起動してしまうと1つのCSVに2つの run が混ざり、step が前後する。
    最後に生き残った run の軌跡だけを取り出すための後始末。
    """
    kept: list[tuple[int, float]] = []
    last = None
    for step, value in reversed(points):
        if last is None or step < last:
            kept.append((step, value))
            last = step
    kept.reverse()
    return kept


@dataclass(frozen=True)
class Curve:
    name: str
    train: list[tuple[int, float]]
    val: list[tuple[int, float]]

    @property
    def gaps(self) -> list[tuple[int, float]]:
        """val 側の step に train を線形補間して合わせ、汎化ギャップを出す."""
        if not self.train:
            return []
        steps = [s for s, _ in self.train]
        values = [v for _, v in self.train]
        out: list[tuple[int, float]] = []
        for step, val in self.val:
            # 検証は log-interval の倍数で走るとは限らないので、近い2点から補間する。
            after = next((i for i, s in enumerate(steps) if s >= step), None)
            if after is None:
                interpolated = values[-1]
            elif after == 0 or steps[after] == step:
                interpolated = values[after]
            else:
                s0, s1 = steps[after - 1], steps[after]
                v0, v1 = values[after - 1], values[after]
                interpolated = v0 + (v1 - v0) * (step - s0) / (s1 - s0)
            out.append((step, val - interpolated))
        return out

    @property
    def best_val(self) -> tuple[int, float]:
        return min(self.val, key=lambda p: p[1])

    @property
    def divergence_step(self) -> int:
        """val loss が train loss を追い越した最初の step.

        「過学習の開始点」を目分量で決めると記事の数字が揺れるので、
        再現できる定義を1つに決めておく。

        学習初期は dropout が効いている分だけ train の方が高く出るので、
        ギャップは負から始まる。それが正に転じた地点を境目とみなす。
        厳密な汎化ギャップのゼロ点ではない（dropout のぶん下駄がある）が、
        同じ設定どうしを比べるかぎりは一貫した目印になる。
        """
        return next((step for step, gap in self.gaps if gap > 0), 0)

    @property
    def final_train(self) -> float:
        return self.train[-1][1] if self.train else float("nan")


def read_curve(path: str | Path, name: str | None = None) -> Curve:
    path = Path(path)
    train: list[tuple[int, float]] = []
    val: list[tuple[int, float]] = []
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            step = int(row["step"])
            if row["train_loss"]:
                train.append((step, float(row["train_loss"])))
            if row["val_loss"]:
                val.append((step, float(row["val_loss"])))
    return Curve(name or path.stem, keep_last_run(train), keep_last_run(val))
