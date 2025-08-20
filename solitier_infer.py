import asyncio
import datetime
import math
import os
import random
from typing import Optional, Tuple

import torch
from solitier_dataset import DEFAULT_SCORE_ARGS
from solitier_game import (
    SolitireCardPositionFrom,
    SolitireCardPositionTo,
    SolitireGame,
    SolitireState,
)
from solitier_model import SolitireAbstractEndToEndValueModel
from solitier_token import state_to_token_indices


class SolitireValueExecuter:
    def __init__(self, model: SolitireAbstractEndToEndValueModel):
        self.model = model
        self.model.eval()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    async def execute(self, states: list[SolitireState]) -> list[float]:
        """
        Execute the value model on the given states.
        """
        token_indices = [state_to_token_indices(state) for state in states]
        token_indices_tensor = torch.tensor(token_indices, dtype=torch.long)
        token_indices_tensor = token_indices_tensor.to(self.device)
        # print("Executing model on states:", token_indices_tensor.shape)
        with torch.no_grad():
            with torch.autocast("cuda", dtype=torch.bfloat16):
                outputs = self.model(token_indices_tensor).float()
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

    async def execute_state(self, state: SolitireState) -> float:
        """
        Execute the value model on a single state.
        """
        token_indices = state_to_token_indices(state)
        loop = asyncio.get_running_loop()
        fut = loop.create_future()
        self.result_futures[id(state)] = fut
        await self.job_queue.put((id(state), token_indices))
        return await fut

    async def execute(self, states: list[SolitireState]) -> list[float]:
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


async def estimate_move_of_game(
    game: SolitireGame,
    executor: SolitireValueExecuter,
    epsilon: float = 0.0,
    is_verbose: bool = False,
    score_args: Optional[dict] = None,
    cvar_alpha: Optional[float] = None,
) -> Optional[Tuple[SolitireCardPositionFrom, SolitireCardPositionTo]]:
    """
    Estimate the value of the next move in the game using the provided executer.
    """
    if score_args is None:
        score_args = DEFAULT_SCORE_ARGS
    moves = game.enumerate_valid_moves_excluding_same_state()
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


async def play_game_with_executor(
    game: SolitireGame,
    executor: SolitireValueExecuter,
    max_moves: int = 200,
    epsilon: float = 0.0,
    is_verbose: bool = False,
    score_args: Optional[dict] = None,
    cvar_alpha: Optional[float] = None,
) -> bool:
    """
    Play the game using the executor to estimate the best moves.
    """
    for i in range(max_moves):
        move = await estimate_move_of_game(
            game,
            executor,
            epsilon=epsilon,
            is_verbose=is_verbose,
            score_args=score_args,
            cvar_alpha=cvar_alpha,
        )
        if move is None:
            if is_verbose:
                print("No more valid moves. Game over.")
            return False
        if is_verbose:
            print(f"Executing move[{i}]: {move}")
        is_moved = game.checked_move_excluding_same_state(*move)
        if game.state.is_all_open():
            if is_verbose:
                print(f"Game completed in {i} moves.")
            return True
        if not is_moved:
            if is_verbose:
                print(f"Move[{i}] failed: {move}")
            return False
    return False  # Game did not complete within max_moves


def create_datetime_str():
    date = datetime.datetime.now()
    return "{0:%Y%m%d%H%M%S_%f}".format(date)


async def loop_log_play_game_with_executor(
    executor: SolitireValueExecuter,
    model_name: str,
    log_dir: str = r"/mnt/c/log/solitire/generated_log",
    max_moves: int = 200,
    epsilon: float = 0.0,
    is_verbose: bool = False,
    score_args: Optional[dict] = None,
    cvar_alpha: Optional[float] = None,
) -> None:
    path = os.path.join(log_dir, model_name)
    if cvar_alpha is not None:
        path += f"_cvar{cvar_alpha:.2f}"
    os.makedirs(path, exist_ok=True)
    while True:
        game = SolitireGame()
        is_successed = await play_game_with_executor(
            game,
            executor,
            epsilon=epsilon,
            max_moves=max_moves,
            is_verbose=is_verbose,
            score_args=score_args,
            cvar_alpha=cvar_alpha,
        )
        filename = os.path.join(path, create_datetime_str() + ".pickle")
        game.save_to_file(filename)
        open_count = game.state.open_count()
        print(is_successed, open_count)


def batched_loop_log_play_game(
    model: SolitireAbstractEndToEndValueModel,
    model_name: str,
    log_dir: str = r"/mnt/c/log/solitire/generated_log",
    batch_size: int = 64,
    max_moves: int = 200,
    epsilon: float = 0.0,
    is_verbose: bool = False,
    score_args: Optional[dict] = None,
    cvar_alpha: Optional[float] = None,
) -> None:
    executor = SolitireValueBatchedExecuter(model, batch_size=batch_size)
    executor.start()
    for i in range(batch_size):
        asyncio.create_task(
            loop_log_play_game_with_executor(
                executor,
                model_name=model_name,
                log_dir=log_dir,
                max_moves=max_moves,
                epsilon=epsilon,
                score_args=score_args,
                is_verbose=is_verbose,
                cvar_alpha=cvar_alpha,
            )
        )
