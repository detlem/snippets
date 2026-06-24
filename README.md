# snippets

個人用の汎用スニペット / スクリプト置き場。再利用できる小さなパーツを、言語・トピックごとに置いておくところ。

## 収録スニペット

### Python

| パス | 概要 |
| --- | --- |
| [`python/async/bounded_concurrency.py`](python/async/bounded_concurrency.py) | 非同期 I/O（vLLM 推論など完了時間がバラつくタスク）を **最大同時実行数を固定したまま** 流す雛形。1本終わるごとに次を投入し、継続バッチ（continuous batching）/ コネクションプールを飽和させる。窓スライド版（`map_as_completed`）＋セマフォ版（`gather_bounded`）＋ fail-soft / 一時エラーのみ指数バックオフ再試行つき。 |

## 使い方

stdlib だけで動くスニペットが多いので素の `python` でも走るが、ツール込みなら uv 推奨。

```bash
uv sync                                            # dev ツール（ruff / mypy）を入れる
uv run python python/async/bounded_concurrency.py  # デモ実行
uv run ruff check .                                # lint
uv run mypy .                                      # 型チェック
```

## 技術スタック / 規約

- Python 3.12（`.python-version` で固定）、依存・実行は **uv**。
- 型ヒント必須、**ruff** + **mypy (strict)** を通す。line-length 100。和文コメント前提なので RUF001/002/003 は無効。
- タイムスタンプは UTC・ISO 8601・Z。秘密は `.env`（コミットしない）。生成物は gitignore。

## License

MIT — see [LICENSE](LICENSE).
