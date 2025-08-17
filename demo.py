import asyncio
import os
from typing import Optional, Tuple

from huggingface_hub import hf_hub_download
from solitier_game import SolitireCardPositionFrom, SolitireCardPositionTo, SolitireGame
from solitier_infer import SolitireValueExecuter, estimate_move_of_game
from solitier_model import SolitireEndToEndValueModel
from solitier_visualize import SolitireGameVisualizer

repo_id = "piyohogeo/deep_solitire"
pth = hf_hub_download(
    repo_id,
    filename="value/solitire_endtoend_value_m512_l18_h8_n74_cb10.0_20250811_052139/best_model.pth",
)  # パスはアップロードした場所に合わせて
jpath = hf_hub_download(
    repo_id,
    filename="value/solitire_endtoend_value_m512_l18_h8_n74_cb10.0_20250811_052139/best_model_params.json",
)
print(f"Model path: {pth}")
print(f"Model params path: {jpath}")


async def main():
    model = SolitireEndToEndValueModel.load_from_file(os.path.split(pth)[0])
    game = SolitireGame()
    visualizer = SolitireGameVisualizer(game)
    executer = SolitireValueExecuter(model)

    async def estimate_move(
        game: SolitireGame,
    ) -> Optional[Tuple[SolitireCardPositionFrom, SolitireCardPositionTo]]:
        return await estimate_move_of_game(
            game, executer, epsilon=0.0, is_verbose=False
        )

    await visualizer.run_by_move_estimator(
        estimate_move,
        is_loop=True,
    )


if __name__ == "__main__":
    asyncio.run(main())
