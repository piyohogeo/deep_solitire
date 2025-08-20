import gc
import glob
import hashlib
import math
import multiprocessing as mp
import os
import pickle
import random
from functools import partial
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch
from solitier_game import SolitireGame
from solitier_token import state_to_token_indices
from torch.utils.data import ConcatDataset, DataLoader, Dataset, Sampler
from tqdm import tqdm

DEFAULT_SCORE_ARGS = {
    "complete_bonus": 10.0,
    "gamma": 0.99,
    "is_delta": False,
    "w_open": 0.05,  # 開札の増分に対する重み
    "w_foundation": 0.20,  # Foundationに積んだ増分に対する重み
    "penalty_stagnation": -0.01,  # 進捗なし（開札/基礎増分ゼロ）の微ペナルティ
    "penalty_stock_cycle": -0.02,  # ストックを1巡させたときの微ペナルティ
}


def score_game(game: SolitireGame, score_args: dict) -> float:
    """
    Calculate the score of the current state.
    The score is the number of cards plus bonus for completing the game.
    """
    score = 0.0
    if game.state.is_all_open():
        score += score_args["complete_bonus"]
    score += game.state.open_count() * 0.1
    # discount the score by the number of moves
    return score


# 新: 各遷移 t->t+1 の報酬 r_t を返す
def score_game_deltas(game: SolitireGame, score_args: dict) -> list[float]:
    """
    各遷移の差分報酬列 r_t を返す。
    - Δopen = open_count(s_{t+1}) - open_count(s_t)
    - Δfound = foundation 枚数の増分（open_count に現れない「基礎に積むだけ」の行為も加点）
    - 進捗ゼロやストック巡回に微ペナルティ
    - クリア時は最終遷移に complete_bonus を付与
    """
    states = game.states
    if len(states) < 2:
        return []

    def opened_cards(st):  # 開いているカード総数（既存の open_count と同義）
        return st.open_count()

    def foundation_cards(st):  # Foundation 上の枚数合計
        # state.foundation は {Suit: list[CardState]} の辞書
        return sum(len(lst) for lst in st.foundation.values())

    rewards: list[float] = []
    for t in range(len(states) - 1):
        s, sp = states[t], states[t + 1]
        r = 0.0

        # 1) 開札増分（裏がめくれたときに効く）
        r += score_args["w_open"] * (opened_cards(sp) - opened_cards(s))

        # 2) Foundation 増分（開札枚数が変わらなくても、基礎に積む価値を加点）
        r += score_args["w_foundation"] * (foundation_cards(sp) - foundation_cards(s))

        # 3) 進捗なしの微ペナルティ
        if r == 0.0:
            r += score_args["penalty_stagnation"]

        # 4) ストック巡回の微ペナルティ（在庫を戻すだけの手を少しだけ嫌う）
        if sp.stock_cycle_count > s.stock_cycle_count:
            r += score_args["penalty_stock_cycle"]

        rewards.append(r)

    # 5) 完了ボーナス（最終遷移に付与）
    if game.state.is_all_open() and rewards:
        rewards[-1] += score_args["complete_bonus"]

    return rewards


def build_training_data(
    path: str, score_args: dict
) -> tuple[list[tuple[np.ndarray, float]], bool]:
    try:
        gc.disable()  # Disable garbage collection for performance
        game = SolitireGame.load_from_file(path)
    except (EOFError, pickle.UnpicklingError, OSError) as e:
        print(f"Error loading game from {path}: {e}")
        return ([], False)
    finally:
        gc.enable()
    score = score_game(game, score_args=score_args)
    discounted = [
        score * (score_args["gamma"] ** i) for i in reversed(range(len(game.states)))
    ]
    is_complete = game.state.is_all_open()
    tokenized = [
        np.array(state_to_token_indices(s), dtype=np.uint8) for s in game.states
    ]
    return (list(zip(tokenized, discounted)), is_complete)


def build_training_data_deltas(
    path: str, score_args: dict
) -> tuple[list[tuple[np.ndarray, float]], bool]:
    try:
        gc.disable()
        game = SolitireGame.load_from_file(path)
    except (EOFError, pickle.UnpicklingError, OSError) as e:
        print(f"Error loading game from {path}: {e}")
        return ([], False)
    finally:
        gc.enable()

    # 変更点: 各遷移の報酬列 r_t を作る
    rewards = score_game_deltas(game, score_args=score_args)

    # r_t から G_t = r_t + γ G_{t+1} を後ろから累積（長さ: len(states)-1）
    returns: list[float] = []
    G = 0.0
    for r in reversed(rewards):
        G = r + score_args["gamma"] * G
        returns.append(G)
    returns.reverse()

    # 学習ターゲットは「状態 t の価値 = G_t」。
    # 終端状態（最後の state）はこれ以上の手がないため 0 を割り当てて揃える。
    targets = returns + [0.0]

    tokenized = [
        np.array(state_to_token_indices(s), dtype=np.uint8) for s in game.states
    ]
    token_scores = list(zip(tokenized, targets))
    is_complete = game.state.is_all_open()
    return (token_scores, is_complete)


def flatten(xss, ys):
    for xs, y in zip(xss, ys):
        for x in xs:
            yield x, y


def load_all(
    pickled_game_paths, score_args: dict
) -> tuple[list[list[tuple[np.ndarray, float]]], list[bool]]:
    if score_args["is_delta"]:
        worker = partial(build_training_data_deltas, score_args=score_args)
    else:
        worker = partial(build_training_data, score_args=score_args)
    with mp.Pool(mp.cpu_count()) as pool:
        # 進捗を逐次出したいなら imap_unordered
        per_game = list(
            tqdm(
                pool.imap_unordered(worker, pickled_game_paths, chunksize=10),
                total=len(pickled_game_paths),
            )
        )
    token_scores = [x for x, is_complete in per_game]  # filter out empty results
    is_completes = [is_complete for _, is_complete in per_game]

    return token_scores, is_completes


def extract_filename(path):
    return os.path.split(path)[1]


def hash_string_list(strings: list[str]) -> str:
    hasher = hashlib.sha256()
    for s in strings:
        # ファイル順序を反映したい場合はそのまま
        hasher.update(s.encode("utf-8"))
        hasher.update(b"\0")  # 区切り用
    return hasher.hexdigest()


class SolitireTokenScoresDirectoryCache:
    def __init__(
        self,
        directory: str,
        score_args: dict,
        is_skip_hash_check: bool = False,
    ):
        self.directory = directory
        self.score_args = score_args
        paths = glob.glob(os.path.join(directory, "*.pickle"))
        filenames = [extract_filename(p) for p in paths]
        filenames.sort()  # Ensure consistent order
        self.hash = hash_string_list(filenames)
        if score_args["is_delta"]:
            filename = f"token_scores_cb{score_args['complete_bonus']:.1f}"
            filename += f"_dl_gm{score_args['gamma']:.2f}"
            filename += f"_wo{score_args['w_open']:.2f}"
            filename += f"_wf{score_args['w_foundation']:.2f}"
            filename += f"_psta{score_args['penalty_stagnation']:.2f}"
            filename += f"_psto{score_args['penalty_stock_cycle']:.2f}.cache"
            cache_path = os.path.join(
                directory,
                filename,
            )
        else:
            cache_path = os.path.join(
                directory, f"token_scores_cb{score_args['complete_bonus']:.1f}.cache"
            )
        if os.path.exists(cache_path):
            try:
                gc.disable()  # Disable garbage collection for performance
                with open(cache_path, "rb") as f:
                    token_scores_cache = pickle.load(f)
            finally:
                gc.enable()
        else:
            token_scores_cache = None
        if token_scores_cache is None or (
            token_scores_cache["hash"] != self.hash and not is_skip_hash_check
        ):
            token_scoress, is_completes = load_all(paths, score_args=score_args)
            self.token_scores = {
                filename: (token_scores, is_complete)
                for filename, (token_scores, is_complete) in zip(
                    filenames, zip(token_scoress, is_completes)
                )
                if len(token_scores) > 0
            }
            token_scores_cache = {
                "hash": self.hash,
                "token_scores": self.token_scores,
            }
            with open(cache_path, "wb") as f:
                pickle.dump(token_scores_cache, f)
        self.token_scores = token_scores_cache["token_scores"]

    def get_filenames(self):
        return list(self.token_scores.keys())


class SolitireDataset(Dataset):
    @classmethod
    def load_from_paths(
        cls,
        pickled_game_paths: list[str],
        score_args: dict,
    ):
        # build parallel data loading
        data, is_completes = load_all(pickled_game_paths, score_args=score_args)
        return cls(data, is_completes)

    @classmethod
    def load_from_directory_cache(
        cls,
        directory_cache: SolitireTokenScoresDirectoryCache,
        filenames: list[str],
    ):
        token_scores_is_completes = [
            directory_cache.token_scores[filename]
            for filename in filenames
            if filename in directory_cache.token_scores
        ]
        token_scores = [x[0] for x in token_scores_is_completes]
        is_completes = [x[1] for x in token_scores_is_completes]

        return cls(token_scores, is_completes)

    @classmethod
    def load_from_directory(
        cls,
        directory: str,
        filenames: list[str],
        score_args: dict,
        is_skip_hash_check: bool = False,
    ):
        cache = SolitireTokenScoresDirectoryCache(
            directory,
            score_args=score_args,
            is_skip_hash_check=is_skip_hash_check,
        )
        return cls.load_from_directory_cache(
            cache,
            filenames,
        )

    def __init__(
        self, data: list[list[tuple[np.ndarray, float]]], is_completes: list[bool]
    ):
        self.data = list(flatten(data, is_completes))
        self.complete_rate = (
            sum(is_completes) / len(is_completes) if is_completes else 0.0
        )
        self.is_completes = [x[1] for x in self.data]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        (tokenized_state, score), is_complete = self.data[idx]
        return (
            torch.tensor(tokenized_state, dtype=torch.long),
            torch.tensor(score, dtype=torch.float32),
            torch.tensor(is_complete, dtype=torch.bool),
        )


def load_caches_from_directories(
    directories: list[str],
    score_args: Optional[dict] = None,
):
    if score_args is None:
        score_args = DEFAULT_SCORE_ARGS
    for directory in directories:
        print(f"Loading cache from {directory}...")
        _ = SolitireTokenScoresDirectoryCache(
            directory,
            score_args=score_args,
        )


def load_solitire_dataset(
    directories: list[str],
    score_args: Optional[dict] = None,
    train_ratio: float = 0.9,
    is_skip_hash_check: bool = False,
) -> tuple[Dataset, Dataset, int, list[bool], list[int], list[float]]:
    """
    Load a Solitire dataset from a directory with specified filenames.
    """
    if score_args is None:
        score_args = DEFAULT_SCORE_ARGS
    train_datasets = []
    val_datasets = []
    train_dataset_lengths = []
    complete_rates = []
    is_completes = []
    total_train_size = 0
    for directory in directories:
        print(f"Loading dataset from {directory}...")
        cache = SolitireTokenScoresDirectoryCache(
            directory,
            score_args=score_args,
            is_skip_hash_check=is_skip_hash_check,
        )
        filenames = cache.get_filenames()
        random.shuffle(filenames)
        train_size = int(len(filenames) * train_ratio)
        total_train_size += train_size
        train_filenames = filenames[:train_size]
        val_filenames = filenames[train_size:]

        train_dataset = SolitireDataset.load_from_directory_cache(
            cache,
            train_filenames,
        )
        train_datasets.append(train_dataset)
        complete_rates.append(train_dataset.complete_rate)
        is_completes.extend(train_dataset.is_completes)
        train_dataset_lengths.append(len(train_dataset))
        val_dataset = SolitireDataset.load_from_directory_cache(
            cache,
            val_filenames,
        )
        val_datasets.append(val_dataset)
    avg_complete_rate = np.array(complete_rates) * np.array(train_dataset_lengths)
    avg_complete_rate = avg_complete_rate.sum() / sum(train_dataset_lengths)

    train_dataset = ConcatDataset(train_datasets)
    train_dataset.complete_rate = avg_complete_rate
    val_dataset = ConcatDataset(val_datasets)
    return (
        train_dataset,
        val_dataset,
        total_train_size,
        is_completes,
        train_dataset_lengths,
        complete_rates,
    )


class BalancedBatchSampler(Sampler):
    def __init__(self, win_idx, lose_idx, batch_size, ratio=0.5, shuffle=True):
        assert 0 < ratio < 1
        self.win_idx = list(win_idx)
        self.lose_idx = list(lose_idx)
        self.batch_size = batch_size
        self.k_win = int(round(batch_size * ratio))
        self.k_lose = batch_size - self.k_win
        self.shuffle = shuffle
        self._reset_iters()

    def _reset_iters(self):
        if self.shuffle:
            random.shuffle(self.win_idx)
            random.shuffle(self.lose_idx)
        self.win_ptr = 0
        self.lose_ptr = 0

    def __iter__(self):
        self._reset_iters()
        n_batches = max(
            (len(self.win_idx) + self.k_win - 1) // self.k_win,
            (len(self.lose_idx) + self.k_lose - 1) // self.k_lose,
        )
        for _ in range(n_batches):
            batch = []

            # 勝ち側
            if self.win_ptr + self.k_win <= len(self.win_idx):
                batch += self.win_idx[self.win_ptr : self.win_ptr + self.k_win]
                self.win_ptr += self.k_win
            else:
                need = self.k_win
                remain = len(self.win_idx) - self.win_ptr
                if remain > 0:
                    batch += self.win_idx[self.win_ptr :]
                    need -= remain
                # 足りない分はリプレースありで補充
                batch += random.choices(self.win_idx, k=need)
                self.win_ptr = len(self.win_idx)

            # 負け側
            if self.lose_ptr + self.k_lose <= len(self.lose_idx):
                batch += self.lose_idx[self.lose_ptr : self.lose_ptr + self.k_lose]
                self.lose_ptr += self.k_lose
            else:
                need = self.k_lose
                remain = len(self.lose_idx) - self.lose_ptr
                if remain > 0:
                    batch += self.lose_idx[self.lose_ptr :]
                    need -= remain
                batch += random.choices(self.lose_idx, k=need)
                self.lose_ptr = len(self.lose_idx)

            random.shuffle(batch)
            yield batch

    def __len__(self):
        # 概算：両クラスを使い切るまで
        return max(
            (len(self.win_idx) + self.k_win - 1) // self.k_win,
            (len(self.lose_idx) + self.k_lose - 1) // self.k_lose,
        )


def load_solitire_balanced_dataloader(
    directories: list[str],
    batch_size: int = 256,
    train_ratio: float = 0.9,
    is_skip_hash_check: bool = False,
    num_workers: int = 8,
    score_args: Optional[dict] = None,
) -> tuple[DataLoader, DataLoader, int, float]:
    if score_args is None:
        score_args = DEFAULT_SCORE_ARGS
    train_dataset, val_dataset, total_train_size, is_completes, _, _ = (
        load_solitire_dataset(
            directories,
            score_args=score_args,
            train_ratio=train_ratio,
            is_skip_hash_check=is_skip_hash_check,
        )
    )
    train_sampler = BalancedBatchSampler(
        win_idx=[i for i, is_complete in enumerate(is_completes) if is_complete],
        lose_idx=[i for i, is_complete in enumerate(is_completes) if not is_complete],
        batch_size=batch_size,
        ratio=0.5,
    )
    train_dataloader = DataLoader(
        train_dataset,
        batch_sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=False,
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=False,
    )
    return (
        train_dataloader,
        val_dataloader,
        total_train_size,
        train_dataset.complete_rate,
    )


class DirStratifiedBatchSampler(Sampler):
    def __init__(
        self,
        dir_to_idx,
        dir_probs,
        batch_size,
        win_ratio=0.5,
        shuffle_dirs=True,
        total_train_size=None,
        steps_per_epoch=None,
    ):
        self.dir_ids = list(dir_to_idx.keys())
        self.dir_to_idx = dir_to_idx
        self.dir_probs = [dir_probs[d] for d in self.dir_ids]
        self.B = batch_size
        self.k_win = int(round(batch_size * win_ratio))
        self.k_lose = batch_size - self.k_win
        self.shuffle_dirs = shuffle_dirs

        self.steps_per_epoch = steps_per_epoch
        if self.steps_per_epoch is None:
            if total_train_size is not None:
                import math

                self.steps_per_epoch = math.ceil(total_train_size / self.B)
            else:
                # フォールバック：後述Bに近い計算
                wins_total = sum(len(dir_to_idx[d]["win"]) for d in dir_to_idx)
                loses_total = sum(len(dir_to_idx[d]["lose"]) for d in dir_to_idx)
                k_win = max(1, int(round(self.B * win_ratio)))
                k_lose = self.B - k_win
                self.steps_per_epoch = max(
                    1,
                    min(math.ceil(wins_total / k_win), math.ceil(loses_total / k_lose)),
                )

    def __iter__(self):
        # ---- 事前に全体プールを作っておく（毎バッチ再計算しない）----
        all_wins = [i for d in self.dir_ids for i in self.dir_to_idx[d]["win"]]
        all_loses = [i for d in self.dir_ids for i in self.dir_to_idx[d]["lose"]]

        # 空プールのフェイルセーフ：極端なケースに備え、一度だけチェック
        if len(all_wins) == 0 or len(all_loses) == 0:
            raise RuntimeError("DirStratifiedBatchSampler: no wins or loses available.")

        # 抽選ヘルパ：空なら全体プールから補充、足りなければリプレースありで補完
        def draw(pool, k, global_pool):
            if k <= 0:
                return []
            if len(pool) >= k:
                return random.sample(pool, k)
            if len(pool) > 0:
                return pool + random.choices(pool, k=k - len(pool))
            # ディレクトリ内が空 → 全体から引く（必ず長さ>0）
            return random.sample(global_pool, k)

        for _ in range(self.steps_per_epoch):
            # ① ディレクトリ割当（多項的）
            alloc_dirs = random.choices(self.dir_ids, weights=self.dir_probs, k=self.B)
            if self.shuffle_dirs:
                random.shuffle(alloc_dirs)

            from collections import Counter

            cdir = Counter(alloc_dirs)

            batch = []
            # --- まずは比例配分（丸めで合わないので後で端数調整する）---
            tmp_alloc = []
            for d, c in cdir.items():
                w = int(round(c * self.k_win / self.B))
                w = max(0, min(w, c))  # 0<=w<=c の範囲に丸め
                l = c - w
                tmp_alloc.append((d, w, l))

            # ---- 端数調整：合計が目標に一致するよう微調整 ----
            sum_w = sum(w for _, w, _ in tmp_alloc)
            sum_l = sum(l for _, _, l in tmp_alloc)

            # 勝ち側の不足/過剰を調整
            # 不足なら w を+1、過剰なら -1（cの範囲内）にして近いディレクトリから補正
            def adjust_side(target, cur, get_c, getter, setter):
                delta = target - cur
                if delta == 0:
                    return
                # 並びは適当でOK。大きい c を優先すると収束しやすい
                # getter/setter は tmp_alloc の w,l を触る小関数（下で定義）
                items = list(range(len(tmp_alloc)))
                # 小さな乱れを入れて偏り防止
                random.shuffle(items)
                # c の大きい順にするとより安定
                items.sort(key=lambda i: get_c(i), reverse=True)
                step = 1 if delta > 0 else -1
                remain = abs(delta)
                for i in items:
                    if remain == 0:
                        break
                    d, w, l = tmp_alloc[i]
                    c = get_c(i)
                    # 片側を±1 したときに 0<=w<=c, 0<=l<=c を満たすか確認
                    nw, nl = getter(i) + step, c - (getter(i) + step)
                    if 0 <= nw <= c and 0 <= nl <= c:
                        setter(i, nw, nl)
                        remain -= 1

            # 小関数で w/l を読み書き
            def get_c(i):
                return cdir[tmp_alloc[i][0]]

            def get_w(i):
                return tmp_alloc[i][1]

            def set_wl(i, w, l):
                d, _, _ = tmp_alloc[i]
                tmp_alloc[i] = (d, w, l)

            # l は自動で c-w なので、勝ち側を合わせれば負け側も揃うはず
            # 念のため二重チェック
            sum_w = sum(w for _, w, _ in tmp_alloc)
            sum_l = sum(l for _, _, l in tmp_alloc)
            if sum_w != self.k_win or sum_l != self.k_lose:
                # 不一致が残るのは非常に稀だが、最後は全体プールで補正
                pass

            # ③ 実際に引く（空バケット/不足は draw() が面倒を見ます）
            for d, w, l in tmp_alloc:
                wins = self.dir_to_idx[d]["win"]
                loses = self.dir_to_idx[d]["lose"]
                batch += draw(wins, w, all_wins)
                batch += draw(loses, l, all_loses)

            # ④ まだ端数が残っていたら全体プールで埋める（理論上ここは0のはず）
            cur = len(batch)
            if cur < self.B:
                rest = self.B - cur
                # 念のため半々で埋める
                add_w = rest // 2
                add_l = rest - add_w
                batch += random.choices(all_wins, k=add_w)
                batch += random.choices(all_loses, k=add_l)

            random.shuffle(batch)
            yield batch

    def __len__(self):
        return self.steps_per_epoch


def softmax_temp(xs: Sequence[float], tau: float) -> List[float]:
    """温度付きソフトマックス（数値安定化）。tau は小さいほど差を強調。"""
    if tau <= 0:
        raise ValueError("tau must be > 0")
    m = max(x / tau for x in xs)
    exps = [math.exp(x / tau - m) for x in xs]
    Z = sum(exps) or 1.0
    return [e / Z for e in exps]


def normalize(ps: Sequence[float]) -> List[float]:
    s = float(sum(ps))
    if s <= 0:
        # すべてゼロ等の異常値の場合は一様にする
        return [1.0 / len(ps)] * len(ps)
    return [p / s for p in ps]


def clip_mix(
    ps: Sequence[float], floor: float = 0.0, cap: float | None = None
) -> List[float]:
    """各成分に下限/上限を適用してから再正規化。"""
    q = [max(p, floor) for p in ps]
    if cap is not None:
        q = [min(p, cap) for p in q]
    return normalize(q)


def build_dir_probs(
    n_list: Sequence[int],  # ディレクトリ d の学習用エピソード数 (train)
    s_list: Sequence[float],  # ディレクトリ d の成功率 (0.0–1.0)
    alpha: float = 0.3,  # サイズ寄りの重み（0=品質のみ, 1=サイズのみ）
    tau: float = 0.04,  # 品質の温度（小さいほど差を強調）
    floor: float = 0.001,  # 各ディレクトリの最小比（例: 0.1%）
    cap: float | None = 0.30,  # 各ディレクトリの最大比（例: 30%）。制限しないなら None
) -> List[float]:
    if len(n_list) != len(s_list):
        raise ValueError("n_list and s_list must have the same length")
    if any(n < 0 for n in n_list):
        raise ValueError("n_list must be non-negative")
    if any(not (0.0 <= s <= 1.0) for s in s_list):
        raise ValueError("s_list must be in [0,1]")

    # サイズ成分（件数比例）
    size_part = normalize(n_list)

    # 品質成分（成功率の温度付きソフトマックス）
    qual_part = softmax_temp(s_list, tau=tau)

    # ミックス
    mixed = [alpha * a + (1 - alpha) * b for a, b in zip(size_part, qual_part)]

    # 下限/上限を適用して再正規化
    probs = clip_mix(mixed, floor=floor, cap=cap)

    return probs


def build_dir_probs_with_quota(
    n_list,
    s_list,
    is_human,
    alpha=0.3,
    tau=0.04,
    floor=0.01,
    cap=0.30,
    quota_human=0.10,
):
    # 人間/生成に分ける
    idx_h = [i for i, b in enumerate(is_human) if b]
    idx_g = [i for i, b in enumerate(is_human) if not b]

    probs = [0.0] * len(n_list)

    # 生成側のミックス（サイズ+品質）で 1.0 を作る
    size_g = normalize([n_list[i] for i in idx_g])
    qual_g = softmax_temp([s_list[i] for i in idx_g], tau=tau)
    mix_g = [alpha * a + (1 - alpha) * b for a, b in zip(size_g, qual_g)]
    # floor/cap → 再正規化
    mix_g = normalize([min(max(p, floor), cap) for p in mix_g])

    # 割当
    for j, i in enumerate(idx_g):
        probs[i] = mix_g[j] * (1.0 - quota_human)
    # 人間側は均等 or サイズ比例（どちらでもOK。サイズ比例にする例）
    if idx_h:
        size_h = normalize([n_list[i] for i in idx_h])
        for j, i in enumerate(idx_h):
            probs[i] = size_h[j] * quota_human

    # 最後に丸め誤差の再正規化
    s = sum(probs)
    if s > 0:
        probs = [p / s for p in probs]
    return probs


def build_dir_probs_winsor(
    n_list, s_list, alpha=0.3, tau=0.04, floor=0.01, cap=0.30, s_clip=0.25
):
    if len(n_list) != len(s_list):
        raise ValueError("n_list and s_list must have the same length")
    if any(n < 0 for n in n_list):
        raise ValueError("n_list must be non-negative")
    if any(not (0.0 <= s <= 1.0) for s in s_list):
        raise ValueError("s_list must be in [0,1]")
    size_part = normalize(n_list)
    s_w = [min(s, s_clip) for s in s_list]  # ★ 高勝率を飽和
    qual_part = softmax_temp(s_w, tau=tau)
    mixed = [alpha * a + (1 - alpha) * b for a, b in zip(size_part, qual_part)]
    probs = normalize([min(max(p, floor), cap) for p in mixed])  # clip→正規化
    return probs


def load_solitire_dir_stratified_dataloader(
    directories: list[str],
    batch_size: int = 256,
    train_ratio: float = 0.9,
    is_skip_hash_check: bool = False,
    num_workers: int = 8,
    score_args: Optional[dict] = None,
) -> tuple[DataLoader, DataLoader, int, float]:
    if score_args is None:
        score_args = DEFAULT_SCORE_ARGS
    (
        train_dataset,
        val_dataset,
        total_train_size,
        is_completes,
        dir_lengths,
        complete_rates,
    ) = load_solitire_dataset(
        directories,
        score_args=score_args,
        train_ratio=train_ratio,
        is_skip_hash_check=is_skip_hash_check,
    )
    steps = math.ceil(sum(dir_lengths) / batch_size)
    # 例：dir_lengths = [len(ds0_train), len(ds1_train), ...]
    #     is_completes = [True/False,...] (train concatenated order)
    is_humans = ["human" in d for d in directories]

    dir_to_idx = {d: {"win": [], "lose": []} for d in range(len(dir_lengths))}
    offset = 0
    for d, L in enumerate(dir_lengths):
        for i in range(L):
            gidx = offset + i
            if is_completes[gidx]:
                dir_to_idx[d]["win"].append(gidx)
            else:
                dir_to_idx[d]["lose"].append(gidx)
        offset += L  # ← ここに入れる

    dir_probs = build_dir_probs_with_quota(
        n_list=dir_lengths,
        s_list=complete_rates,
        is_human=is_humans,
    )
    for directory, length, complete_rate, dir_prob in zip(
        directories, dir_lengths, complete_rates, dir_probs
    ):
        print(
            f"Directory: {directory}\n Length: {length}, Complete Rate: {complete_rate:.2f}, Probability: {dir_prob:.4f}"
        )
    train_sampler = DirStratifiedBatchSampler(
        dir_to_idx=dir_to_idx,
        dir_probs=dir_probs,
        batch_size=batch_size,
        win_ratio=0.5,
        steps_per_epoch=steps,
    )
    train_dataloader = DataLoader(
        train_dataset,
        batch_sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=False,
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=False,
    )
    return (
        train_dataloader,
        val_dataloader,
        total_train_size,
        train_dataset.complete_rate,
    )


def compute_target_stats_from_dataloader(
    dataloader,
    *,
    # is_complete でフィルタしたい場合に使う:
    #   None … フィルタしない（デフォルト）
    #   True … is_complete==True のサンプルだけ使う
    #   False… is_complete==False のサンプルだけ使う
    filter_by_is_complete: Optional[bool] = None,
    eps: float = 1e-12,
) -> Tuple[float, float]:
    """
    DataLoader からターゲット（score）の μ, σ を計算する。
    バッチは (tokens, score, is_complete) 形式を想定。
    """

    def _to_numpy(x):
        if torch.is_tensor(x):
            return x.detach().cpu().float().numpy()
        return np.asarray(x, dtype=np.float32)

    sumw = 0.0
    sumwy = 0.0
    sumwy2 = 0.0

    for batch in tqdm(dataloader):
        # (tokens, score, is_complete) を想定
        if not (isinstance(batch, (tuple, list)) and len(batch) >= 3):
            raise ValueError("Batch must be (tokens, score, is_complete).")

        scores = batch[1]

        y = _to_numpy(scores).reshape(-1)
        if y.size == 0:
            continue

        # 等重みで集計
        sumw += float(y.size)
        sumwy += float(y.sum())
        sumwy2 += float((y**2).sum())

    if sumw <= 0:
        # 何も集計できなかった場合のフォールバック
        return 0.0, 1.0

    mu = sumwy / sumw
    var = max(eps, (sumwy2 / sumw) - mu * mu)
    sigma = math.sqrt(var)
    return float(mu), float(sigma)
