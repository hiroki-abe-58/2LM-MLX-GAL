"""記事・動画用のスクリーンショットを撮る (Chrome DevTools Protocol).

実際の Google Chrome を headless=new で起動し、チャットを実行してから撮影する。
GUI開発時の回帰確認にも使える。教材本体には不要なユーティリティ。

    python tools/screenshot.py --url http://127.0.0.1:8000 --out docs/images
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import shutil
import subprocess
import time
import urllib.error
import urllib.request
from itertools import count
from pathlib import Path

import websockets

CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
PROFILE = Path("/tmp/2lm-chrome-profile")


class CDP:
    """必要最小限の DevTools Protocol クライアント."""

    def __init__(self, ws):
        self.ws = ws
        self.ids = count(1)
        self.problems: list[str] = []

    def _record(self, msg: dict) -> None:
        """コンソールのエラーと未捕捉例外を集めておく."""
        method = msg.get("method")
        if method == "Runtime.exceptionThrown":
            detail = msg["params"]["exceptionDetails"]
            description = detail.get("exception", {}).get("description", "")
            self.problems.append(f"exception: {detail.get('text')} {description}")
        elif method == "Runtime.consoleAPICalled" and msg["params"]["type"] in ("error", "assert"):
            args = " ".join(str(a.get("value", a.get("description", ""))) for a in msg["params"]["args"])
            self.problems.append(f"console.error: {args}")

    async def call(self, method: str, **params):
        msg_id = next(self.ids)
        await self.ws.send(json.dumps({"id": msg_id, "method": method, "params": params}))
        while True:
            msg = json.loads(await self.ws.recv())
            self._record(msg)
            if msg.get("id") == msg_id:
                if "error" in msg:
                    raise RuntimeError(f"{method}: {msg['error']}")
                return msg.get("result", {})

    async def js(self, expression: str):
        result = await self.call(
            "Runtime.evaluate", expression=expression, returnByValue=True, awaitPromise=True
        )
        return result.get("result", {}).get("value")

    async def shot(self, path: Path) -> None:
        result = await self.call("Page.captureScreenshot", format="png", captureBeyondViewport=False)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(base64.b64decode(result["data"]))
        print(f"  撮影: {path}")


def launch_chrome(port: int) -> subprocess.Popen:
    if PROFILE.exists():
        shutil.rmtree(PROFILE)
    proc = subprocess.Popen(
        [
            CHROME,
            "--headless=new",
            f"--remote-debugging-port={port}",
            f"--user-data-dir={PROFILE}",
            "--hide-scrollbars",
            "--no-first-run",
            "--disable-extensions",
            "--force-color-profile=srgb",
            "about:blank",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    for _ in range(60):
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/json/version", timeout=1):
                return proc
        except (urllib.error.URLError, ConnectionError):
            time.sleep(0.5)
    raise SystemExit("Chrome の起動に失敗しました")


def page_ws_url(port: int) -> str:
    with urllib.request.urlopen(f"http://127.0.0.1:{port}/json/list", timeout=5) as res:
        targets = json.loads(res.read())
    for target in targets:
        if target["type"] == "page":
            return target["webSocketDebuggerUrl"]
    raise SystemExit("page ターゲットが見つかりません")


async def wait_replies(cdp: CDP, expected: int, timeout: float = 90.0) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        n = await cdp.js("document.querySelectorAll('.stats').length")
        if n and int(n) >= expected:
            await asyncio.sleep(0.4)
            return
        await asyncio.sleep(0.5)
    raise SystemExit("返答が完了しませんでした (server.py は起動していますか)")


async def run(url: str, out: Path, port: int, width: int, height: int) -> None:
    ws_url = page_ws_url(port)
    async with websockets.connect(ws_url, max_size=64 * 1024 * 1024) as ws:
        cdp = CDP(ws)
        await cdp.call("Page.enable")
        await cdp.call("Runtime.enable")
        await cdp.call(
            "Emulation.setDeviceMetricsOverride",
            width=width,
            height=height,
            deviceScaleFactor=2,
            mobile=False,
        )
        await cdp.call("Page.navigate", url=url)
        await asyncio.sleep(3.0)  # フォントとアイコンの読み込み待ち

        await cdp.shot(out / "gui-welcome.png")

        messages = ["こんにちは", "おすすめの本を教えてください", "AIとは何ですか？"]
        for i, message in enumerate(messages, start=1):
            await cdp.js(
                "(() => {"
                f"  const i = document.getElementById('input'); i.value = {json.dumps(message)};"
                "  document.getElementById('composer')"
                "    .dispatchEvent(new Event('submit', {cancelable: true}));"
                "  return true; })()"
            )
            await asyncio.sleep(1.0)
            if i == 1:
                await cdp.shot(out / "gui-streaming.png")
            await wait_replies(cdp, i)

        await cdp.shot(out / "gui-chat.png")

        await cdp.js("document.getElementById('btn-settings').click()")
        await asyncio.sleep(0.8)
        await cdp.shot(out / "gui-settings.png")

        # 以下は撮影せず、UIの動作確認だけ行う
        await cdp.js("document.getElementById('modal-apply').click()")
        await asyncio.sleep(0.5)
        if await cdp.js("!document.getElementById('modal-backdrop').hidden"):
            raise SystemExit("モーダルが閉じていません")

        await cdp.js("document.getElementById('btn-clear').click()")
        await asyncio.sleep(0.5)
        checks = await cdp.js(
            "JSON.stringify({"
            "  welcome: !!document.querySelector('.welcome'),"
            "  messages: document.querySelectorAll('.msg').length,"
            "  icons: document.querySelectorAll('.welcome svg').length })"
        )
        print(f"  リセット後の状態: {checks}")
        if json.loads(checks)["messages"] != 0 or not json.loads(checks)["welcome"]:
            raise SystemExit("リセットで初期表示に戻っていません")

        if cdp.problems:
            print("\nブラウザ側の問題:")
            for problem in cdp.problems:
                print("  -", problem)
            raise SystemExit(1)
        print("  コンソールエラー: なし")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--out", default="docs/images")
    ap.add_argument("--port", type=int, default=9222)
    ap.add_argument("--width", type=int, default=1180)
    ap.add_argument("--height", type=int, default=860)
    args = ap.parse_args()

    proc = launch_chrome(args.port)
    try:
        asyncio.run(run(args.url, Path(args.out), args.port, args.width, args.height))
    finally:
        proc.terminate()


if __name__ == "__main__":
    main()
