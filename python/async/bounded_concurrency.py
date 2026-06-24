"""最大同時実行数を固定したまま非同期タスクを流すための雛形。

vLLM の推論など「タスクごとに完了時間がバラつく I/O」をバッチ実行するとき、
リストをチャンクに割って ``asyncio.gather`` を繰り返すと、各チャンクの最遅タスクが
次チャンクの先頭を待たせる（バリア同期）。その結果サーバ側の running batch に
スキマができて GPU / コネクションプールが遊ぶ。

ここでは「常に max_concurrency 本を in-flight に保ち、1本終わった瞬間に次を投入する」
2 つの型を置く:

- map_as_completed: 窓スライド版。遅延イテレータを食えてメモリ有界、終わった順に
  (item, 結果 or 例外) を yield。巨大ジョブや「結果を逐次書き出したい」ケース向け（本命）。
- gather_bounded:   セマフォ + gather。入力が全部メモリに乗るなら最小差分。これでも
                    「1本空けば即1本入る」ので継続バッチは飽和する。

エラーは握りつぶさない: fn は失敗時に素直に例外を投げてよく、ドライバ側が例外を
「値」として拾って呼び出し側へ渡す（asyncio.gather(return_exceptions=True) と同じ流儀）。
呼び出し側は ``async for item, res in ...:`` と回し ``isinstance(res, Exception)`` で
成功/失敗を振り分ける（失敗時も item で「どの入力か」を辿れる）。
リトライは opt-in: 要るときだけ with_retry で fn を包む（最終失敗時は例外が surfaced される）。

timeout はこのコードに既定値を持たない（小さすぎる隠れデフォルトは無い）。タスク依存なので
fn 側で設定する（httpx の timeout / asyncio.wait_for など）。max_concurrency は vLLM の
max_num_seqs 付近に合わせるのが目安（実測で詰める）。

実行:
    uv run python python/async/bounded_concurrency.py
    # stdlib のみなので素の python でも動く
"""

from __future__ import annotations

import asyncio
import random
import time
from collections.abc import AsyncIterator, Callable, Coroutine, Iterable
from typing import Any


async def map_as_completed[T, R](  # T=入力（プロンプト等）, R=成功時の結果
    inputs: Iterable[T],
    fn: Callable[[T], Coroutine[Any, Any, R]],
    max_concurrency: int,
) -> AsyncIterator[tuple[T, R | Exception]]:
    """inputs を最大 max_concurrency 本だけ並行で回し、(item, 結果 or 例外) を終わった順に yield。

    1本終わるたびに次の input を投入するので窓が常に埋まる（= 継続バッチを使い切る）。
    inputs は遅延イテレータで OK。窓のぶんしかメモリに乗らない。

    エラーは握りつぶさない。fn が例外を投げたら (item, その例外) を yield するので、
    呼び出し側で ``isinstance(res, Exception)`` で振り分けて handling する。失敗時も item
    （元の入力）が手に入るので「どの入力でコケたか」を辿れる（順不同でも識別子を失わない）。
    raise する fn は何も返さないので、失敗に入力を貼れるのはドライバだけ＝ここで持ち回る。
    CancelledError などの BaseException は捕まえず素通しする（cancel が効くように）。
    """
    if max_concurrency < 1:
        raise ValueError("max_concurrency must be >= 1")

    it = iter(inputs)
    in_flight: dict[asyncio.Task[R], T] = {}  # task -> 元の入力（失敗時の対応付け用）

    def fill() -> None:  # 窓が空いてるぶんだけ次を投入
        while len(in_flight) < max_concurrency:
            try:
                item = next(it)
            except StopIteration:
                return
            in_flight[asyncio.create_task(fn(item))] = item

    fill()
    try:
        while in_flight:
            done, _ = await asyncio.wait(in_flight.keys(), return_when=asyncio.FIRST_COMPLETED)
            for task in done:
                item = in_flight.pop(task)
                # task.result() だけを try で包む（yield を外に出して、消費側で起きた
                # 例外を「タスクの失敗」と取り違えないようにする）。中身は下と等価:
                #   try: yield item, task.result()
                #   except Exception as e: yield item, e
                try:
                    result: R | Exception = task.result()
                except Exception as exc:  # 失敗は握りつぶさず値として返す
                    result = exc
                yield item, result
            fill()  # 終わったぶんを即補充 → 窓を埋め直す
    finally:
        for task in in_flight:  # 途中で抜けたら後始末（タスクのリーク防止）
            task.cancel()
        if in_flight:
            await asyncio.gather(*in_flight, return_exceptions=True)


async def gather_bounded[T, R](
    inputs: Iterable[T],
    fn: Callable[[T], Coroutine[Any, Any, R]],
    max_concurrency: int,
    *,
    return_exceptions: bool = False,
) -> list[R | BaseException]:
    """セマフォ + gather。入力が全部メモリに乗るなら最小差分でこれで十分。

    gather それ自体は直列ではない（全部並行に走る）。セマフォで同時 acquire を
    max_concurrency に絞ると、in-flight は常に max_concurrency に保たれる = 継続バッチは飽和。
    return_exceptions=True なら失敗は例外オブジェクトとして結果リストに（入力順で）入る。
    こちらは結果が入力順に並ぶので、識別子は zip(inputs, results) で取れる（map_as_completed
    と違って item を持ち回る必要が無い）。弱点は (1) 全タスクを一度に生成する
    (2) 結果が最後にまとめて返る、の2点。
    """
    sem = asyncio.Semaphore(max_concurrency)

    async def guarded(item: T) -> R:
        async with sem:
            return await fn(item)

    return await asyncio.gather(
        *(guarded(item) for item in inputs), return_exceptions=return_exceptions
    )


def with_retry[T, R](
    fn: Callable[[T], Coroutine[Any, Any, R]],
    *,
    retries: int = 2,
    transient: tuple[type[Exception], ...] = (TimeoutError, ConnectionError),
    base_delay: float = 0.1,
) -> Callable[[T], Coroutine[Any, Any, R]]:
    """fn を「一時エラーだけ指数バックオフで retries 回まで再試行する fn」に包む（opt-in）。

    最終的に失敗したら例外をそのまま投げる（= map_as_completed 側で値として拾われる）。
    transient 以外（4xx 相当）は再試行せず即 surfaced。リトライ不要なら素の fn を渡せばよい。
    使用例: map_as_completed(inputs, with_retry(fn, retries=2), max_concurrency)
    """

    async def wrapped(item: T) -> R:
        for attempt in range(retries + 1):
            try:
                return await fn(item)
            except transient:  # 一時エラーのみ再試行
                if attempt == retries:
                    raise  # 使い切ったら例外をそのまま投げる
                await asyncio.sleep(base_delay * 2**attempt)
        raise AssertionError("unreachable")  # ループは必ず return か raise する（型のため）

    return wrapped


# --- 以下はデモ。実運用では infer を vLLM 呼び出しに、main の中身を差し替える ---


async def infer(prompt: str) -> str:
    """本番は AsyncLLMEngine.generate か OpenAI 互換 /v1/completions を叩くところ。

    成功時だけ結果を返し、失敗時は素直に例外を投げる（握りつぶさない）。
    timeout を入れるならこの中で（httpx の timeout か asyncio.wait_for で包む）。
    """
    await asyncio.sleep(random.uniform(0.05, 0.6))  # 完了時間がバラつく想定
    if random.random() < 0.05:  # たまに失敗（例外はドライバが拾って消費側へ渡す）
        raise TimeoutError("simulated transient error")
    return f"<<{prompt} への応答>>"


async def main(prompts: list[str], max_concurrency: int = 32) -> None:
    async def work(prompt: str) -> str:
        return await infer(prompt)  # 失敗時は例外がそのまま伝播 → ドライバが拾う

    # リトライを入れたいときは work を包むだけ（最終失敗時は例外が surfaced される）:
    #   map_as_completed(prompts, with_retry(work, retries=2), max_concurrency)
    started = time.perf_counter()
    ok = 0
    failures: list[tuple[str, Exception]] = []
    async for prompt, res in map_as_completed(prompts, work, max_concurrency):
        if isinstance(res, Exception):
            failures.append((prompt, res))  # 入力(prompt)も一緒に → どれがコケたか分かる
        else:
            ok += 1  # 成功（res が結果。実運用では集約 dict をここで逐次書き出す）

    elapsed = time.perf_counter() - started
    print(
        f"done: {len(prompts)} 件 / ok={ok} failed={len(failures)} / "
        f"{elapsed:.2f}s / {len(prompts) / elapsed:.1f} req/s"
    )
    for prompt, exc in failures[:3]:
        print(f"  failed: {prompt}: {type(exc).__name__}: {exc}")


if __name__ == "__main__":
    asyncio.run(main([f"prompt-{i:03d}" for i in range(200)], max_concurrency=32))
