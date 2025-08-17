import gc
import multiprocessing as mp
import pickle
import random

import matplotlib.pyplot as plt
import torch
import tqdm
from pt_utils import build_mask_like, nearest_embedding_indices_from_embedding_layer
from solitier_game import SolitireGame
from solitier_model import SolitireMAEModel
from solitier_token import (
    SPECIAL_TOKEN_BEGIN,
    flattened_card_state_tokens_and_tokens_to_map_card_states,
    index_to_token,
    state_flatten,
    state_to_token_indices,
)
from solitier_visualize import SolitiereStateVisualizer


def evaluate_mae_model_to_show(
    model: SolitireMAEModel,
    paths: list[str],
    evaluate_count: int = 5,
    mask_ratio: float = 0.15,
) -> float:
    device = torch.device("cuda")
    model.eval()
    model.to(device)
    visualizer = SolitiereStateVisualizer()
    for i in range(evaluate_count):
        path = random.choice(paths)
        game = SolitireGame.load_from_file(path)
        game.deduplicate_same_states()
        state = random.choice(game.states)

        card_state_tokens = state_flatten(state)
        token_indices = state_to_token_indices(state)
        token_indices = torch.tensor(token_indices, dtype=torch.long).unsqueeze(0)

        with torch.no_grad():
            token_indices = token_indices.to(device)

            # モデルの出力を取得
            inputs = model.embeddings(token_indices)
            normal_token_mask = token_indices < SPECIAL_TOKEN_BEGIN
            normal_token_mask = normal_token_mask.to(device)
            mask = build_mask_like(token_indices, mask_ratio)
            mask = mask.to(device)
            mask = mask & normal_token_mask

            with torch.autocast("cuda", dtype=torch.bfloat16):
                latents = model.encode(inputs, mask=mask)
                nearest_indices = model.decode_to_indices(latents)

            nearest_indices = nearest_indices.squeeze(0).cpu().numpy()
        nearest_tokens = [index_to_token(idx) for idx in nearest_indices]
        card_dif_map = flattened_card_state_tokens_and_tokens_to_map_card_states(
            card_state_tokens, nearest_tokens
        )
        masked_card = [
            card_state.card
            for (card_state, token), is_mask in zip(
                card_state_tokens, mask.squeeze(0).cpu().numpy()
            )
            if is_mask
        ]
        visualizer.unselect_all()
        visualizer.select_cards(masked_card)
        original_state_visualization = visualizer.visualize_state_to_numpy_image(
            state,
        )
        reconstructed_state_visualization = visualizer.visualize_state_to_numpy_image(
            state, card_to_card_state_map=card_dif_map
        )
        plt.figure(figsize=(14, 14))
        plt.subplot(2, 1, 1)
        plt.imshow(original_state_visualization)
        plt.title("Original State")
        plt.axis("off")

        plt.subplot(2, 1, 2)
        plt.imshow(reconstructed_state_visualization)
        plt.title("Reconstructed State")
        plt.axis("off")

        plt.show()


def show_statistics_of_games(paths: list[str]) -> float:
    """
    Show statistics of the games in the given paths.
    """
    game_count = len(paths)
    total_states = []
    success_states = []
    opened_counts = []
    for path in tqdm.tqdm(paths):
        game = SolitireGame.load_from_file(path)
        total_states.append(len(game.states))
        if game.state.is_all_open():
            success_states.append(len(game.states))
        opened_counts.append(game.state.open_count())
    if game_count == 0:
        print("No games found.")
        return
    avg_states = sum(total_states) / game_count if total_states else 0
    avg_success_states = (
        sum(success_states) / len(success_states) if success_states else 0
    )
    avg_non_success_states = (sum(total_states) - sum(success_states)) / game_count
    avg_opened_count = sum(opened_counts) / game_count if opened_counts else 0
    success_rate = len(success_states) / game_count * 100

    print(f"Total games: {game_count}")
    print(f"Success games: {len(success_states)}")
    print(f"Success game rate: {success_rate:.2f}%")
    print(f"Average opened cards per game: {avg_opened_count:.2f}")
    print(f"Average states per game: {avg_states:.2f}")
    print(f"Average success states per game: {avg_success_states:.2f}")
    print(f"Average non-success states per game: {avg_non_success_states:.2f}")
    print(f"Max states in a game: {max(total_states) if total_states else 0}")
    print(
        f"Max success states in a game: {max(success_states) if success_states else 0}"
    )
    return success_rate


def _compute_game_stats(path: str):
    """
    1ゲームファイル分の統計を計算して返す（並列ワーカーで実行する関数）
    """
    try:
        gc.disable()  # Disable garbage collection for performance
        game = SolitireGame.load_from_file(path)
    except (EOFError, pickle.UnpicklingError, OSError) as e:
        print(f"Error loading game from {path}: {e}")
        return False, 0, 0, 0, 0
    finally:
        gc.enable()
    total_states = len(game.states)
    is_success = 1 if game.state.is_all_open() else 0
    success_states = total_states if is_success else 0
    opened_count = game.state.open_count()
    return True, total_states, success_states, opened_count, is_success


# 追加：並列版の関数
def show_statistics_of_games_parallel(
    paths: list[str], processes: int | None = None, chunksize: int = 8
) -> float:
    """
    Show statistics of the games in the given paths using multiprocessing.
    """
    game_count = len(paths)
    if game_count == 0:
        print("No games found.")
        return 0.0

    totals = 0
    success_totals = 0
    opened_totals = 0
    success_games = 0

    with mp.Pool(processes=processes) as pool:
        # imap_unorderedで順不同に受け取りつつ進捗を出す
        for is_ok, total_states, success_states, opened_count, is_success in tqdm.tqdm(
            pool.imap_unordered(_compute_game_stats, paths, chunksize=chunksize),
            total=game_count,
        ):
            if not is_ok:
                continue
            totals += total_states
            success_totals += success_states
            opened_totals += opened_count
            success_games += is_success

    avg_states = totals / game_count
    avg_success_states = (success_totals / success_games) if success_games > 0 else 0
    avg_non_success_states = (totals - success_totals) / game_count
    avg_opened_count = opened_totals / game_count
    success_rate = success_games / game_count * 100

    print(f"Total games: {game_count}")
    print(f"Success games: {success_games}")
    print(f"Success game rate: {success_rate:.2f}%")
    print(f"Average opened cards per game: {avg_opened_count:.2f}")
    print(f"Average states per game: {avg_states:.2f}")
    print(f"Average success states per game: {avg_success_states:.2f}")
    print(f"Average non-success states per game: {avg_non_success_states:.2f}")
    print(f"Max states in a game: (tracked per-file if needed)")
    print(f"Max success states in a game: (tracked per-file if needed)")

    return success_rate
