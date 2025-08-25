import argparse
import asyncio
import os
from typing import Optional, Tuple

from huggingface_hub import hf_hub_download
from solitier_game import SolitireCardPositionFrom, SolitireCardPositionTo, SolitireGame
from solitier_infer import SolitireValueExecuter, estimate_move_of_game_greedy
from solitier_model import SolitireEndToEndValueModel
from solitier_search import estimate_move_of_game_by_mcts
from solitier_visualize import SolitireGameVisualizer

repo_id = "piyohogeo/deep_solitire"
# 28.8% @ one-step e-greedy(0.1)
pth = hf_hub_download(
    repo_id,
    filename="value/solitire_endtoend_value_m512_l18_h8_n74_cb0.00_scr0.1445_dss_20250820_012942/best_model.pth",
)  # パスはアップロードした場所に合わせて
jpath = hf_hub_download(
    repo_id,
    filename="value/solitire_endtoend_value_m512_l18_h8_n74_cb0.00_scr0.1445_dss_20250820_012942/best_model_params.json",
)
print(f"Model path: {pth}")
print(f"Model params path: {jpath}")


async def demo_one_step_greedy():
    model = SolitireEndToEndValueModel.load_from_file(os.path.split(pth)[0])
    game = SolitireGame()
    visualizer = SolitireGameVisualizer(game)
    executer = SolitireValueExecuter(model)

    async def estimate_move(
        game: SolitireGame,
    ) -> Optional[Tuple[SolitireCardPositionFrom, SolitireCardPositionTo]]:
        return await estimate_move_of_game_greedy(
            game, executer, epsilon=0.1, is_verbose=False
        )

    await visualizer.run_by_move_estimator(
        estimate_move,
        is_loop=True,
    )


# experimentally implemented
async def demo_mcts():
    model = SolitireEndToEndValueModel.load_from_file(os.path.split(pth)[0])
    game = SolitireGame()
    visualizer = SolitireGameVisualizer(game)
    executer = SolitireValueExecuter(model)

    async def estimate_move(
        game: SolitireGame,
    ) -> Optional[Tuple[SolitireCardPositionFrom, SolitireCardPositionTo]]:
        return await estimate_move_of_game_by_mcts(
            game,
            executer,
            iterations=1000,
            batch_size=64,
            c_ucb=1.4,
            epsilon=0.1,
            is_verbose=False,
            is_tqdm=False,
        )

    await visualizer.run_by_move_estimator(
        estimate_move,
        is_loop=True,
    )


async def main(is_mcts: bool = False):
    if is_mcts:
        await demo_mcts()
    else:
        await demo_one_step_greedy()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Deep Solitire demo runner")
    g = parser.add_mutually_exclusive_group()
    g.add_argument(
        "--mcts",
        dest="is_mcts",
        action="store_true",
        help="Use Monte Carlo Tree Search instead of one-step greedy.",
    )
    g.add_argument(
        "--greedy",
        dest="is_mcts",
        action="store_false",
        help="Force one-step epsilon-greedy (default).",
    )
    parser.set_defaults(is_mcts=False)
    args = parser.parse_args()

    asyncio.run(main(is_mcts=args.is_mcts))
