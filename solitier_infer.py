import asyncio
import datetime
import math
import os
import random
from typing import Awaitable, Callable, Optional, Tuple, Union

import torch
from solitier_dataset import DEFAULT_SCORE_ARGS
from solitier_game import (
    SolitireCardPositionFrom,
    SolitireCardPositionTo,
    SolitireGame,
    SolitireState,
)
from solitier_token import TOKEN_INDEX_LEN, TOKENS_SEQ_LEN

try:
    from solitier_game_lw_fast import SolitireLightWeightState
except Exception:
    print("Failed to import solitier_game_lw_fast. Make sure it is compiled.")
    from solitier_game_lw import SolitireLightWeightState

from solitier_model import SolitireAbstractEndToEndValueModel
from solitier_token import state_to_token_indices

AbstractSolitireState = Union[SolitireState, SolitireLightWeightState]


def abstract_solitire_state_to_token_indices(
    state: AbstractSolitireState,
) -> list[int]:
    match state:
        case SolitireState():
            return state_to_token_indices(state)
        case SolitireLightWeightState():
            return state.to_token_indices()
        case _:
            raise ValueError("Unsupported state type")


class SolitireValueExecuter:
    def __init__(self, model: SolitireAbstractEndToEndValueModel):
        self.model = model
        self.model.eval()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    async def execute(self, states: list[AbstractSolitireState]) -> list[float]:
        """
        Execute the value model on the given states.
        """
        token_indices = [
            abstract_solitire_state_to_token_indices(state) for state in states
        ]
        token_indices_tensor = torch.tensor(token_indices, dtype=torch.long)
        token_indices_tensor = token_indices_tensor.to(self.device)
        # print("Executing model on states:", token_indices_tensor.shape)
        with torch.no_grad():
            with torch.autocast("cuda", dtype=torch.bfloat16):
                outputs = self.model(token_indices_tensor).float()
        return outputs.cpu().numpy().tolist()


class SolitireValueBucketGraphExecutor:
    def __init__(self, model: SolitireAbstractEndToEndValueModel, batch_size: int = 64):
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
        self.model = model
        self.model.eval()
        self.device = "cuda"
        self.model.to(self.device)

        self._build_buckets(batch_size)
        self._static_inputs = {}
        self._static_outputs = {}
        self._graph = {}
        for b in self._buckets:
            self._build_graph(b)

    def _build_buckets(self, batch_size: int):
        self._buckets = []
        b = 8
        while b < batch_size:
            self._buckets.append(b)
            b *= 2
        self._buckets.append(batch_size)

    def _pick_bucket(self, batch_size: int) -> int:
        assert batch_size <= self._buckets[-1]
        for b in self._buckets:
            if b >= batch_size:
                return b

    def _build_graph(self, batch_size: int):
        x = torch.randint(
            TOKEN_INDEX_LEN,
            (batch_size, TOKENS_SEQ_LEN),
            device="cuda",
            dtype=torch.long,
        )
        static_in = torch.empty_like(x, device=self.device)  # [B, T] long
        # 出力形状はモデルに合わせて（ここでは [B] を想定）
        static_out = torch.empty((batch_size,), dtype=torch.float32, device=self.device)

        # ウォームアップ（必ず妥当な値で埋めてから）
        static_in.copy_(x)
        with (
            torch.inference_mode(),
            torch.autocast(device_type="cuda", dtype=torch.bfloat16),
        ):
            for _ in range(3):
                y = self.model(static_in).float()
                static_out.copy_(y)
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            torch.cuda.synchronize()
            with torch.cuda.graph(g):
                static_out.copy_(self.model(static_in).float())
        self._static_inputs[batch_size] = static_in
        self._static_outputs[batch_size] = static_out
        self._graph[batch_size] = g

    def run(self, x: torch.Tensor) -> torch.Tensor:
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            b = x.size(0)
            B = self._pick_bucket(b)
            static_in = self._static_inputs[B]
            static_out = self._static_outputs[B]
            g = self._graph[B]
            # 入力コピー（CPU->GPUなら .to(device, non_blocking=True) で先にGPUへ）
            static_in[:b].copy_(x[:b], non_blocking=True)
            # 余り領域はゼロ/パディング（必要ならマスクも静的テンソルに書き込み）
            if b < B:
                static_in[b:].zero_()
            g.replay()
            return static_out[:b].clone()  # 取り出し

    async def execute(self, states: list[AbstractSolitireState]) -> list[float]:
        """
        Execute the value model on the given states.
        """
        token_indices = [
            abstract_solitire_state_to_token_indices(state) for state in states
        ]
        token_indices_tensor = torch.tensor(token_indices, dtype=torch.long)
        token_indices_tensor = token_indices_tensor.to(self.device)
        # print("Executing model on states:", token_indices_tensor.shape)
        outputs = self.run(token_indices_tensor)
        return outputs.cpu().numpy().tolist()


class SolitireValueBatchedExecuter:
    def __init__(self, model: SolitireAbstractEndToEndValueModel, batch_size: int = 64):
        self.model = model
        self.compiled_model = torch.compile(model, mode="reduce-overhead")
        self.model.eval()
        self.device = "cuda"
        self.model.to(self.device)

        self.result_futures = {}
        self.job_queue = asyncio.Queue(maxsize=batch_size * 8)
        self.batch_size = batch_size

    def start(self):
        """
        Start the worker to process jobs from the job queue.
        """
        self.worker_task = asyncio.create_task(self.worker())

    async def execute_state(self, state: AbstractSolitireState) -> float:
        """
        Execute the value model on a single state.
        """
        token_indices = abstract_solitire_state_to_token_indices(state)
        loop = asyncio.get_running_loop()
        fut = loop.create_future()
        self.result_futures[id(state)] = fut
        await self.job_queue.put((id(state), token_indices))
        return await fut

    async def execute(self, states: list[AbstractSolitireState]) -> list[float]:
        """
        Execute the value model on the given states.
        """
        return await asyncio.gather(*[self.execute_state(state) for state in states])

    async def worker(self) -> float:
        """
        Worker to process jobs from the job queue in batches.
        """
        while True:
            token_indicess = []
            job_ids = []
            timeout = 0.1  # Adjust as needed
            start = asyncio.get_event_loop().time()
            while len(token_indicess) < self.batch_size:
                remaining_time = timeout - (asyncio.get_event_loop().time() - start)
                if remaining_time <= 0:
                    break
                try:
                    state_id, token_indices = await asyncio.wait_for(
                        self.job_queue.get(), remaining_time
                    )
                except asyncio.TimeoutError:
                    break
                if state_id is None:
                    break
                token_indicess.append(token_indices)
                job_ids.append(state_id)

            if not token_indicess:
                continue

            model = (
                self.compiled_model
                if len(token_indicess) == self.batch_size
                else self.model
            )
            token_indices_tensor = torch.tensor(token_indicess, dtype=torch.long)
            token_indices_tensor = token_indices_tensor.to(self.device)
            with torch.no_grad():
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    outputs = model(token_indices_tensor).float()
            for job_id, output in zip(job_ids, outputs.cpu().numpy().tolist()):
                self.result_futures.pop(job_id).set_result(output)


AbstractSolitireValueExecutor = Union[
    SolitireValueExecuter,
    SolitireValueBucketGraphExecutor,
    SolitireValueBatchedExecuter,
]


async def estimate_move_of_game_greedy(
    game: SolitireGame,
    executor: AbstractSolitireValueExecutor,
    epsilon: float = 0.0,
    is_verbose: bool = False,
    score_args: Optional[dict] = None,
    cvar_alpha: Optional[float] = None,
    is_allow_same_state: bool = False,
) -> Optional[Tuple[SolitireCardPositionFrom, SolitireCardPositionTo]]:
    """
    Estimate the value of the next move in the game using the provided executer.
    """
    if score_args is None:
        score_args = DEFAULT_SCORE_ARGS
    if not is_allow_same_state:
        moves = game.enumerate_valid_moves_excluding_same_state()
    else:
        moves = game.state.enumerate_valid_moves()
    if not moves:
        if is_verbose:
            print("No valid moves available.")
        return None
    if epsilon > random.random():
        # Randomly select a move
        move = random.choice(moves)
        return move
    uncertain_states = {move: game.state.move_uncertain_states(*move) for move in moves}
    flattened_states = []
    for move, states in uncertain_states.items():
        flattened_states.extend([(move, state) for state in states])
    for move, state in flattened_states:
        if state.is_all_open():
            if is_verbose:
                print(f"Game already completed with move: {move}")
            return move
    execute_states = [state for move, state in flattened_states]

    values = await executor.execute(execute_states)

    base = game.state  # 現在の状態 s

    def opened_cards(st):
        return st.open_count()

    def foundation_cards(st):
        return sum(len(v) for v in st.foundation.values())

    base_open = opened_cards(base)
    base_found = foundation_cards(base)
    base_stock = base.stock_cycle_count

    value_map = {}
    for (move, sp), v in zip(flattened_states, values):
        if score_args["is_delta"]:
            # 即時報酬 r(s→s') を学習時と同じ定義で
            sp_open = opened_cards(sp)
            sp_found = foundation_cards(sp)
            d_open, d_found = sp_open - base_open, sp_found - base_found
            r = score_args["w_open"] * d_open
            r += score_args["w_foundation"] * d_found
            if score_args["penalty_stagnation"] != 0.0 and (
                opened_cards(sp) == base_open and foundation_cards(sp) == base_found
            ):
                r += score_args["penalty_stagnation"]
            if sp.stock_cycle_count > base_stock:
                r += score_args["penalty_stock_cycle"]
            q = r + score_args["gamma"] * v
        else:
            q = v  # 旧方式（互換用）

        if move not in value_map:
            value_map[move] = []
        value_map[move].append(q)
    if cvar_alpha is not None:

        def cvar(vals: list[float]) -> float:
            sv = list(sorted(vals))
            m = max(1, int(math.ceil(cvar_alpha * len(sv))))
            return sum(sv[:m]) / m

        move_values = {move: cvar(values) for move, values in value_map.items()}
    else:
        move_values = {
            move: sum(values) / len(values) for move, values in value_map.items()
        }
    if is_verbose:
        print("Move values:", move_values)
    best_move = max(move_values, key=move_values.get)
    return best_move


async def play_game_with_move_estimator(
    game: SolitireGame,
    estimate_move: Callable[
        [SolitireGame],
        Awaitable[Optional[Tuple[SolitireCardPositionFrom, SolitireCardPositionTo]]],
    ],
    max_moves: int = 200,
    is_verbose: bool = False,
) -> bool:
    """
    Play the game using the executor to estimate the best moves.
    """
    for i in range(max_moves):
        move = await estimate_move(
            game,
        )
        if move is None:
            if is_verbose:
                print("No more valid moves. Game over.")
            return False
        if is_verbose:
            print(f"Executing move[{i}]: {move}")
        is_moved = game.checked_move(*move)
        if game.state.is_all_open():
            if is_verbose:
                print(f"Game completed in {i} moves.")
            return True
        if not is_moved:
            if is_verbose:
                print(f"Move[{i}] failed: {move}")
            return False
    return False  # Game did not complete within max_moves


async def play_game_with_executor_greedy(
    game: SolitireGame,
    executor: SolitireValueExecuter,
    max_moves: int = 200,
    epsilon: float = 0.0,
    is_verbose: bool = False,
    score_args: Optional[dict] = None,
    cvar_alpha: Optional[float] = None,
    is_allow_same_state: bool = False,
) -> bool:
    """
    Play the game using the executor to estimate the best moves.
    """

    async def estimate_move(
        game: SolitireGame,
    ) -> Optional[Tuple[SolitireCardPositionFrom, SolitireCardPositionTo]]:
        return await estimate_move_of_game_greedy(
            game,
            executor,
            epsilon=epsilon,
            is_verbose=is_verbose,
            score_args=score_args,
            cvar_alpha=cvar_alpha,
            is_allow_same_state=is_allow_same_state,
        )

    return await play_game_with_move_estimator(
        game,
        estimate_move,
        max_moves=max_moves,
        is_verbose=is_verbose,
    )


def create_datetime_str():
    date = datetime.datetime.now()
    return "{0:%Y%m%d%H%M%S_%f}".format(date)


async def loop_log_play_game_by_move_estimator(
    move_estimate: Callable[
        [SolitireGame],
        Awaitable[Optional[Tuple[SolitireCardPositionFrom, SolitireCardPositionTo]]],
    ],
    model_name: str,
    log_dir: str = r"/mnt/c/log/solitire/generated_log",
    max_moves: int = 200,
    is_verbose: bool = False,
    post_fix: str = "",
) -> None:
    path = os.path.join(log_dir, model_name)
    path += post_fix
    os.makedirs(path, exist_ok=True)
    while True:
        game = SolitireGame()
        is_successed = await play_game_with_move_estimator(
            game,
            move_estimate,
            max_moves=max_moves,
            is_verbose=is_verbose,
        )
        filename = os.path.join(path, create_datetime_str() + ".pickle")
        game.save_to_file(filename)
        open_count = game.state.open_count()
        print(is_successed, open_count)


async def loop_log_play_game_with_executor_greedy(
    executor: AbstractSolitireValueExecutor,
    model_name: str,
    log_dir: str = r"/mnt/c/log/solitire/generated_log",
    max_moves: int = 200,
    epsilon: float = 0.0,
    is_verbose: bool = False,
    score_args: Optional[dict] = None,
    cvar_alpha: Optional[float] = None,
    is_allow_same_state: bool = False,
) -> None:
    async def estimate_move(
        game: SolitireGame,
    ) -> Optional[Tuple[SolitireCardPositionFrom, SolitireCardPositionTo]]:
        return await estimate_move_of_game_greedy(
            game,
            executor,
            epsilon=epsilon,
            is_verbose=is_verbose,
            score_args=score_args,
            cvar_alpha=cvar_alpha,
            is_allow_same_state=is_allow_same_state,
        )

    post_fix = ""
    if cvar_alpha is not None:
        post_fix += f"_cvar{cvar_alpha:.2f}"
    if is_allow_same_state:
        post_fix += "_ss"
    return await loop_log_play_game_by_move_estimator(
        estimate_move,
        model_name=model_name,
        log_dir=log_dir,
        max_moves=max_moves,
        is_verbose=is_verbose,
        post_fix=post_fix,
    )


def batched_loop_log_play_game_greedy(
    model: SolitireAbstractEndToEndValueModel,
    model_name: str,
    log_dir: str = r"/mnt/c/log/solitire/generated_log",
    batch_size: int = 64,
    max_moves: int = 200,
    epsilon: float = 0.0,
    is_verbose: bool = False,
    score_args: Optional[dict] = None,
    cvar_alpha: Optional[float] = None,
    is_allow_same_state: bool = False,
) -> None:
    executor = SolitireValueBatchedExecuter(model, batch_size=batch_size)
    executor.start()
    for i in range(batch_size):
        asyncio.create_task(
            loop_log_play_game_with_executor_greedy(
                executor,
                model_name=model_name,
                log_dir=log_dir,
                max_moves=max_moves,
                epsilon=epsilon,
                score_args=score_args,
                is_verbose=is_verbose,
                cvar_alpha=cvar_alpha,
                is_allow_same_state=is_allow_same_state,
            )
        )


def profile_model(model, batch_size=64):
    import torch.profiler

    model = model.cuda().eval()
    x = torch.randint(
        TOKEN_INDEX_LEN, (batch_size, TOKENS_SEQ_LEN), device="cuda", dtype=torch.long
    )

    # 推論前の一般的な設定（任意）
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    # 事前ウォームアップ（JIT/Autotune/メモリ確保を済ませる）
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        for _ in range(3):
            _ = model(x)
    torch.cuda.synchronize()

    # プロファイラ
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        schedule=torch.profiler.schedule(wait=0, warmup=1, active=5),
        on_trace_ready=None,
        record_shapes=False,
        profile_memory=False,
        with_stack=False,
    ) as prof:
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            # 計測区間（10ステップ分）
            for step in range(10):
                _ = model(x)
                prof.step()  # スケジューラ進行
        torch.cuda.synchronize()

    # 上位オペレーションを表示（自己GPU時間でソート）
    print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=15))


def compare_inference_time(model, batch_size=64, iters=200, device="cuda"):
    """
    d=512 l=18 の Transformer（Embedding入力→値出力）想定。
    Graphなし（素の forward）と、CUDA Graph（replay）の 1step 平均時間を ms で比較します。
    - baseline: model(x) のみ
    - graph(replay_only): g.replay() のみ（静的入力は事前に埋め済み）
    - graph(copy+replay): static_in.copy_(x) を含めた現実的コスト

    返り値: dict
    """
    from torch.cuda import Event

    assert torch.cuda.is_available(), "CUDAが有効な環境で実行してください。"
    device = torch.device(device)

    model = model.to(device).eval()
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    # 語彙サイズはモデル側を優先（安全）
    vocab_size = TOKEN_INDEX_LEN

    # 実行用テンソル
    x = torch.randint(
        vocab_size, (batch_size, TOKENS_SEQ_LEN), device=device, dtype=torch.long
    )

    # ==== ベースライン（Graphなし） ====
    with (
        torch.inference_mode(),
        torch.autocast(device_type="cuda", dtype=torch.bfloat16),
    ):
        # 軽いウォームアップ
        for _ in range(5):
            _ = model(x)

        torch.cuda.synchronize()
        start, end = Event(True), Event(True)
        times = []
        for _ in range(iters):
            start.record()
            _ = model(x)
            end.record()
            torch.cuda.synchronize()
            times.append(start.elapsed_time(end))  # ms
        baseline_ms = sum(times) / len(times)

    # ==== CUDA Graph 準備 ====
    static_in = torch.empty_like(x, device=device)  # [B, T] long
    # 出力形状はモデルに合わせて（ここでは [B] を想定）
    static_out = torch.empty((batch_size,), dtype=torch.float32, device=device)

    # ウォームアップ（必ず妥当な値で埋めてから）
    static_in.copy_(x)
    with (
        torch.inference_mode(),
        torch.autocast(device_type="cuda", dtype=torch.bfloat16),
    ):
        for _ in range(3):
            y = model(static_in).float()
            static_out.copy_(y)
    torch.cuda.synchronize()

    # キャプチャ（1ステップのみ）
    g = torch.cuda.CUDAGraph()
    with (
        torch.inference_mode(),
        torch.autocast(device_type="cuda", dtype=torch.bfloat16),
    ):
        torch.cuda.synchronize()
        with torch.cuda.graph(g):
            y = model(static_in).float()
            static_out.copy_(y)

    # ==== Graph: replayのみ（copyを含めない純粋forward時間） ====
    with (
        torch.inference_mode(),
        torch.autocast(device_type="cuda", dtype=torch.bfloat16),
    ):
        torch.cuda.synchronize()
        start, end = Event(True), Event(True)
        times = []
        for _ in range(iters):
            # 入力はすでに static_in に入っている想定
            start.record()
            g.replay()
            end.record()
            torch.cuda.synchronize()
            times.append(start.elapsed_time(end))
        graph_replay_ms = sum(times) / len(times)

    # ==== Graph: 入力コピー込み（現実的） ====
    with (
        torch.inference_mode(),
        torch.autocast(device_type="cuda", dtype=torch.bfloat16),
    ):
        torch.cuda.synchronize()
        start, end = Event(True), Event(True)
        times = []
        for _ in range(iters):
            static_in.copy_(
                x, non_blocking=True
            )  # 実運用で CPU->GPU なら x は pin_memory から to(cuda, non_blocking=True)
            start.record()
            g.replay()
            end.record()
            torch.cuda.synchronize()
            times.append(start.elapsed_time(end))
        graph_copy_replay_ms = sum(times) / len(times)

    result = {
        "baseline_ms": baseline_ms,
        "graph_replay_only_ms": graph_replay_ms,
        "graph_copy_plus_replay_ms": graph_copy_replay_ms,
        "speedup_vs_baseline_replay_only": baseline_ms / graph_replay_ms,
        "speedup_vs_baseline_copy_plus_replay": baseline_ms / graph_copy_replay_ms,
    }

    print(
        f"[batch={batch_size}] baseline: {baseline_ms:.3f} ms | "
        f"graph(replay): {graph_replay_ms:.3f} ms | "
        f"graph(copy+replay): {graph_copy_replay_ms:.3f} ms | "
        f"speedup(replay): x{baseline_ms/graph_replay_ms:.2f} | "
        f"speedup(copy+replay): x{baseline_ms/graph_copy_replay_ms:.2f}"
    )
    return result


def compare_latency_e2e(model, batch_size=32, iters=200, device="cuda"):
    import time

    dev = torch.device(device)
    model = model.to(dev).eval()
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    vocab = TOKEN_INDEX_LEN
    x = torch.randint(vocab, (batch_size, TOKENS_SEQ_LEN), device=dev, dtype=torch.long)

    def bench_baseline():
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            for _ in range(5):
                _ = model(x)
        torch.cuda.synchronize()
        e2e, gpu = [], []
        start_ev = torch.cuda.Event(True)
        end_ev = torch.cuda.Event(True)
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            for _ in range(iters):
                t0 = time.perf_counter()
                start_ev.record()
                _ = model(x)  # 途中で同期があってもOK
                end_ev.record()
                torch.cuda.synchronize()
                t1 = time.perf_counter()
                e2e.append((t1 - t0) * 1000)
                gpu.append(start_ev.elapsed_time(end_ev))  # GPU時間(ms)
        e2e_ms = sum(e2e) / len(e2e)
        gpu_ms = sum(gpu) / len(gpu)
        host_ms = max(e2e_ms - gpu_ms, 0.0)  # 残りをHost側とみなす
        return {"e2e_ms": e2e_ms, "gpu_ms": gpu_ms, "host_ms": host_ms}

    def bench_graph():
        static_in = torch.empty_like(x)
        static_out = torch.empty((batch_size,), dtype=torch.float32, device=dev)
        static_in.copy_(x)
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            for _ in range(3):
                static_out.copy_(model(static_in).float())
        g = torch.cuda.CUDAGraph()
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            torch.cuda.synchronize()
            with torch.cuda.graph(g):
                static_out.copy_(model(static_in).float())
        e2e, gpu = [], []
        start_ev = torch.cuda.Event(True)
        end_ev = torch.cuda.Event(True)
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            for _ in range(iters):
                t0 = time.perf_counter()
                static_in.copy_(x, non_blocking=True)
                start_ev.record()
                g.replay()
                end_ev.record()
                torch.cuda.synchronize()
                t1 = time.perf_counter()
                e2e.append((t1 - t0) * 1000)
                gpu.append(start_ev.elapsed_time(end_ev))
        e2e_ms = sum(e2e) / len(e2e)
        gpu_ms = sum(gpu) / len(gpu)
        host_ms = max(e2e_ms - gpu_ms, 0.0)
        return {"e2e_ms": e2e_ms, "gpu_ms": gpu_ms, "host_ms": host_ms}

    base = bench_baseline()
    graph = bench_graph()
    print(
        f"[B={batch_size}] E2E: base {base['e2e_ms']:.3f}→graph {graph['e2e_ms']:.3f} ms "
        f"(x{base['e2e_ms']/graph['e2e_ms']:.2f}) | "
        f"GPU: {base['gpu_ms']:.3f}→{graph['gpu_ms']:.3f} ms "
        f"(x{(base['gpu_ms']/graph['gpu_ms']) if graph['gpu_ms']>0 else float('inf'):.2f}) | "
        f"Host: {base['host_ms']:.3f}→{graph['host_ms']:.3f} ms"
    )
    return {"baseline": base, "graph": graph}
