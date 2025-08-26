import argparse
import importlib
import statistics as stats
import time
from typing import Any, Callable, Tuple


# ---------- 共通ユーティリティ ----------
def try_import_fast():
    try:
        return importlib.import_module("solitier_game_lw_fast")
    except Exception:
        return None


def timeit_recreate(
    builder: Callable[[], Tuple[Any, Callable[[], Any]]], warmup: int, repeat: int
):
    """
    builder() -> (context_obj, target_fn)
      * 計測ごとに新しいオブジェクト群を作る（←キャッシュ対策）
      * ただし '作る時間' はカウントしない
    """
    # ウォームアップ（作成+実行。ただし作成時間は無視）
    for _ in range(warmup):
        _, fn = builder()
        fn()

    # 計測
    ts = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        _, fn = builder()  # 作成
        t1 = time.perf_counter()
        fn()  # 実行
        t2 = time.perf_counter()
        ts.append(t2 - t1)  # 実行のみを加算（作成時間は除外）
    ts.sort()
    return {
        "avg_ms": 1000 * stats.mean(ts),
        "p50_ms": 1000 * stats.median(ts),
        "p95_ms": 1000 * ts[int(0.95 * (len(ts) - 1))],
        "runs": repeat,
    }


def format_row(name, res):
    return f"{name:50s}  avg {res['avg_ms']:8.3f} ms  p50 {res['p50_ms']:8.3f}  p95 {res['p95_ms']:8.3f}  (n={res['runs']})"


# ---------- ベンチひな型 ----------
def make_builders(mod, game):
    """各計測対象の builder を返す"""
    LWState = getattr(mod, "SolitireLightWeightState")
    LWGame = getattr(mod, "SolitireLightWeightGame")

    # 1) State.enumerate_valid_moves
    def build_state_enum():
        lw_game = LWGame.from_solitire_game(game)  # 作り直す（非計測）
        lw_state = lw_game.get_last_state()
        return lw_state, (lambda: lw_state.enumerate_valid_moves())

    # 2) Game.enumerate_valid_moves_excluding_same_state（毎回新しい Game）
    def build_game_enum_excl():
        lw_game = LWGame.from_solitire_game(game)  # 作り直す（非計測）
        return lw_game, (lambda: lw_game.enumerate_valid_moves_excluding_same_state())

    # 3) Game.move_uncertain_states（全手について合計・毎回新しい Game）
    def build_game_move_all():
        lw_game = LWGame.from_solitire_game(game)  # 作り直す（非計測）

        def run():
            moves = lw_game.enumerate_valid_moves_excluding_same_state()
            total_children = 0
            for from_pos, to_pos in moves:
                children = lw_game.move_uncertain_states(from_pos, to_pos)
                total_children += len(children)
            return total_children

        return lw_game, run

    return build_state_enum, build_game_enum_excl, build_game_move_all


def bench_impl(label: str, mod, game_path: str, warmup: int, repeat: int):
    sg = importlib.import_module("solitier_game")
    game = sg.SolitireGame()

    b1, b2, b3 = make_builders(mod, game)
    res1 = timeit_recreate(b1, warmup, repeat)
    res2 = timeit_recreate(b2, warmup, repeat)
    res3 = timeit_recreate(b3, warmup, repeat)

    print(f"\n[{label}]  file: {game_path}")
    print(format_row("State.enumerate_valid_moves", res1))
    print(format_row("Game.enumerate_valid_moves_excluding_same_state", res2))
    print(format_row("Game.move_uncertain_states (sum over moves)", res3))


def main():
    ap = argparse.ArgumentParser(
        description="Benchmark Python vs Cython LW implementations (recreate per run)."
    )
    ap.add_argument(
        "--game",
        required=True,
        help="Path to a serialized SolitireGame (load_from_file).",
    )
    ap.add_argument("--repeat", type=int, default=5)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--impl", choices=["py", "cy", "both"], default="both")
    args = ap.parse_args()

    py_mod = importlib.import_module("solitier_game_lw")  # Python版
    cy_mod = try_import_fast()  # Cython版（あれば）

    if args.impl in ("py", "both"):
        bench_impl("python", py_mod, args.game, args.warmup, args.repeat)
    if args.impl in ("cy", "both"):
        if cy_mod is None:
            print(
                "\n[cython] solitier_game_lw_fast が見つかりません（ビルド未完了の可能性）。"
            )
        else:
            bench_impl("cython", cy_mod, args.game, args.warmup, args.repeat)


if __name__ == "__main__":
    main()
