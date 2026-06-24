"""最大同時実行数を固定したまま非同期タスクを流すための雛形。

vLLM の推論など「タスクごとに完了時間がバラつく I/O」をバッチ実行するとき、
リストをチャンクに割って ``asyncio.gather`` を繰り返すと、各チャンクの最遅タスクが
次チャンクの先頭を待たせる（バリア同期）。その結果サーバ側の running batch に
スキマができて GPU / コネクションプールが遊ぶ。

ここでは「常に max_concurrency 本を in-flight に保ち、1本終わった瞬間に次を投入する」
2 つの型を置く:

- map_as_completed: 窓スライド版。遅延イテレータを食えてメモリ有界、終わった順に yield。
  巨大ジョブや「結果を逐次書き出したい」ケース向け（本命）。
- gather_bounded:   セマフォ + gather。入力が全部メモリに乗るなら最小差分。これでも
                    「1本空けば即1本入る」ので継続バッチは飽和する。

どちらも継続バッチ（continuous batching）を埋めっぱなしにするのが狙い。
max_concurrency は vLLM の max_num_seqs 付近に合わせるのが目安（実測で詰める）。

実行:
    uv run python python/async/bounded_concurrency.py
    # stdlib のみなので素の python でも動く
"""

from __future__ import annotations

import asyncio
import random
import time
from collections.abc import AsyncIterator, Callable, Coroutine, Iterable
from dataclasses import dataclass
from typing import Any


async def map_as_completed[T, R](  # T=入力（プロンプト等）, R=結果
    inputs: Iterable[T],
    fn: Callable[[T], Coroutine[Any, Any, R]],
    max_concurrency: int,
) -> AsyncIterator[R]:
    """inputs を最大 max_concurrency 本だけ並行で回し、終わった順に結果を yield する。

    1本終わるたびに次の input を投入するので窓が常に埋まる（= 継続バッチを使い切る）。
    inputs は遅延イテレータで OK。窓のぶんしかメモリに乗らない。

    注意: fn は例外を投げない設計にすること。投げると yield 時に呼び出し側へ伝播し、
    残りの in-flight タスクは finally で cancel される。失敗は戻り値に畳むのが安全
    （下の run_one を参照）。
    """
    if max_concurrency < 1:
        raise ValueError("max_concurrency must be >= 1")

    it = iter(inputs)
    pending: set[asyncio.Task[R]] = set()

    def fill() -> None:  # 窓が空いてるぶんだけ次を投入
        while len(pending) < max_concurrency:
            try:
                item = next(it)
            except StopIteration:
                return
            pending.add(asyncio.create_task(fn(item)))

    fill()
    try:
        while pending:
            done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
            for task in done:
                yield task.result()
            fill()  # 終わったぶんを即補充 → 窓を埋め直す
    finally:
        for task in pending:  # 途中で抜けたら後始末（タスクのリーク防止）
            task.cancel()
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)


async def gather_bounded[T, R](
    inputs: Iterable[T],
    fn: Callable[[T], Coroutine[Any, Any, R]],
    max_concurrency: int,
) -> list[R]:
    """セマフォ + gather。入力が全部メモリに乗るなら最小差分でこれで十分。

    gather それ自体は直列ではない（全部並行に走る）。セマフォで同時 acquire を
    max_concurrency に絞ると、1本 release した瞬間に待機中の1本が acquire するので、
    in-flight は常に max_concurrency に保たれる = 継続バッチは飽和する。
    弱点は (1) 全タスクを一度に生成する (2) 結果が最後にまとめて返る、の2点。
    """
    sem = asyncio.Semaphore(max_concurrency)

    async def guarded(item: T) -> R:
        async with sem:
            return await fn(item)

    return await asyncio.gather(*(guarded(item) for item in inputs))


# --- 以下はデモ。実運用では infer を vLLM 呼び出しに、main の中身を差し替える ---


@dataclass(slots=True)
class Outcome:
    index: int
    prompt: str
    text: str | None
    error: str | None


async def infer(prompt: str) -> str:
    """本番は AsyncLLMEngine.generate か OpenAI 互換 /v1/completions を叩くところ。"""
    await asyncio.sleep(random.uniform(0.05, 0.6))  # 完了時間がバラつく想定
    if random.random() < 0.05:  # たまに一時エラーを起こしてリトライ経路を見せる
        raise TimeoutError("simulated transient error")
    return f"<<{prompt} への応答>>"


async def run_one(index: int, prompt: str, retries: int = 2) -> Outcome:
    """1リクエスト分。例外を Outcome に畳む = 1件コケても全体を止めない。

    一時エラー（timeout / 5xx / 429 相当）だけ指数バックオフで再試行。
    それ以外（4xx 相当）は即あきらめる。
    """
    for attempt in range(retries + 1):
        try:
            return Outcome(index, prompt, await infer(prompt), None)
        except (TimeoutError, ConnectionError) as e:  # 一時エラーだけ再試行
            if attempt == retries:
                return Outcome(index, prompt, None, f"{type(e).__name__}: {e}")
            await asyncio.sleep(0.1 * 2**attempt)
        except Exception as e:  # 想定外は握りつぶして1件だけ失敗扱い
            return Outcome(index, prompt, None, f"{type(e).__name__}: {e}")
    raise AssertionError("unreachable")  # ループは必ず return する（型のため）


async def main(prompts: list[str], max_concurrency: int = 32) -> list[Outcome]:
    results: list[Outcome | None] = [None] * len(prompts)
    inputs = enumerate(prompts)  # 遅延イテレータ（全部を一度に展開しない）

    async def work(item: tuple[int, str]) -> Outcome:
        return await run_one(item[0], item[1])

    started = time.perf_counter()
    done = 0
    async for outcome in map_as_completed(inputs, work, max_concurrency):
        results[outcome.index] = outcome  # 元の順序へ書き戻し
        done += 1
        # ここで jsonl / DB に逐次追記すると中断に強いしメモリも空く
        if done % 20 == 0 or done == len(prompts):
            print(f"  [{done}/{len(prompts)}] elapsed={time.perf_counter() - started:.2f}s")

    return [r for r in results if r is not None]


if __name__ == "__main__":
    demo_prompts = [f"prompt-{i:03d}" for i in range(200)]
    started = time.perf_counter()
    outcomes = asyncio.run(main(demo_prompts, max_concurrency=32))
    elapsed = time.perf_counter() - started

    ok = sum(1 for o in outcomes if o.error is None)
    failed = len(outcomes) - ok
    print(
        f"\ndone: {len(outcomes)} 件 / ok={ok} failed={failed} / "
        f"{elapsed:.2f}s / {len(outcomes) / elapsed:.1f} req/s"
    )
