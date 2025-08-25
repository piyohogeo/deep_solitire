import math
import random
from dataclasses import dataclass, field
from typing import (
    Any,
    Awaitable,
    Callable,
    Dict,
    List,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import tqdm
from solitier_game import SolitireCardPositionFrom, SolitireCardPositionTo, SolitireGame
from solitier_game_lw import SolitireLightWeightGame
from solitier_infer import SolitireValueExecuter

SolitireSearchAction = Tuple[SolitireCardPositionFrom, SolitireCardPositionTo]


class SolitireSearchState:
    def __init__(self, game: SolitireLightWeightGame):
        self.game = game

    def is_terminal(self) -> bool:
        return self.game.is_all_open() or len(self.legal_actions()) == 0

    def get_terminal_value(self) -> float:
        if self.game.is_all_open():
            return 1.0
        elif len(self.legal_actions()) == 0:
            return -1.0
        else:
            raise ValueError("非終端状態に対して get_terminal_value() は呼べません。")

    def legal_actions(self) -> List[SolitireSearchAction]:
        return self.game.enumerate_valid_moves_excluding_same_state(
            is_compatibility=False
        )

    def next(self, action: SolitireSearchAction) -> List["SolitireSearchState"]:
        next_games = self.game.move_uncertain_states(*action)
        return [SolitireSearchState(g) for g in next_games]

    def hash(self) -> int:
        return self.game.states.first.hash()

    def __hash__(self):
        return self.hash()

    def __eq__(self, other):
        if not isinstance(other, SolitireSearchState):
            return False
        return self.game.states.first.is_same_state(other.game.states.first)


def _enumerate_outcomes(
    state: SolitireSearchState, action: SolitireSearchAction
) -> List[Tuple[float, SolitireSearchState]]:
    nxt = state.next(action)
    if not nxt:
        return []
    p = 1.0 / len(nxt)
    return [(p, s) for s in nxt]


# ========= nodes =========
@dataclass
class ChanceNode:
    parent: "DecisionNode"
    action_from_parent: SolitireSearchAction
    N: int = 0
    W: float = 0.0
    outcome_children: List["DecisionNode"] = field(default_factory=list)
    unexpanded_outcomes: List[Tuple[float, SolitireSearchState]] = field(
        default_factory=list
    )
    pending: int = 0  # バッチ収集中の仮訪問数


@dataclass
class DecisionNode:
    state: SolitireSearchState
    parent: Optional[ChanceNode] = None
    N: int = 0
    W: float = 0.0
    children: Dict[Any, ChanceNode] = field(
        default_factory=dict
    )  # action -> ChanceNode
    unexpanded_actions: List[SolitireSearchAction] = field(default_factory=list)
    value_estimate: Optional[float] = None
    pending: int = 0

    def q(self) -> float:
        return self.W / self.N if self.N > 0 else 0.0

    def uct_score(self, child: ChanceNode, c_ucb: float = 1.4) -> float:
        # pending を N に加味（バッチ収集中に同じ経路へ偏らないように）
        eff_parent_N = self.N + self.pending
        eff_child_N = child.N + child.pending
        bonus = c_ucb * math.sqrt(math.log(eff_parent_N + 1.0) / (eff_child_N + 1.0))
        q = (child.W / child.N) if child.N > 0 else 0.0
        return q + bonus if child.N > 0 else float("inf")


Node = Union[DecisionNode, ChanceNode]


# ========= batched MCTS =========
async def mcts_with_value_and_chance_batched(
    root_state: SolitireSearchState,
    V_batch: Callable[
        [List[SolitireSearchState]], Awaitable[List[float]]
    ],  # batched: states -> values ([-1,1])
    iterations: int = 2000,
    batch_size: int = 64,
    c_ucb: float = 1.4,
    is_verbose: bool = False,
    is_tqdm: bool = False,
) -> SolitireSearchAction:
    """
    ソリティア向けMCTS（チャンスノード + 価値ネットの一括推論）。
    返り値: 根で採用する action（訪問回数最大）
    """
    root = DecisionNode(state=root_state)
    root.unexpanded_actions = list(root_state.legal_actions())

    value_cache: Dict[int, float] = {}

    # --- 1回分の選択+展開（葉を1つ確保し、通過ノードに pending を積む） ---
    def _select_and_expand_one() -> Tuple[DecisionNode, List[Node]]:
        node: Node = root
        path: List[Node] = [node]  # Decision/Chance 混在の経路

        # Selection
        while True:
            if isinstance(node, DecisionNode):
                if node.state.is_terminal():
                    break
                if node.unexpanded_actions:
                    # expand 1 action -> make a chance node
                    a = node.unexpanded_actions.pop()
                    chance = ChanceNode(parent=node, action_from_parent=a)
                    chance.unexpanded_outcomes = _enumerate_outcomes(node.state, a)
                    node.children[a] = chance
                    node = chance
                    path.append(node)
                    # 続けてチャンス側の展開へ（下で処理）
                elif node.children:
                    node = max(
                        node.children.values(), key=lambda ch: node.uct_score(ch, c_ucb)
                    )
                    path.append(node)
                    continue
                else:
                    break  # 合法手なし

            # ChanceNode の場合
            if isinstance(node, ChanceNode):
                if node.unexpanded_outcomes:
                    # node is ChanceNode
                    lst = node.unexpanded_outcomes  # List[(p, state)]  ※全部 p が同じ
                    j = random.randrange(len(lst))  # 一様にインデックスを引く
                    p, s_next = lst[j]
                    # swap-pop で O(1) で削除（順序は保持しない）
                    lst[j] = lst[-1]
                    lst.pop()

                    child = DecisionNode(state=s_next, parent=node)
                    child.unexpanded_actions = list(s_next.legal_actions())
                    node.outcome_children.append(child)
                    node = child
                    path.append(node)  # ここが今回の「新しい葉」
                    break
                elif node.outcome_children:
                    j = random.randrange(len(node.outcome_children))
                    node = node.outcome_children[j]  # 一様に1つ
                    path.append(node)
                    continue
                else:
                    break  # outcomes なし

        # 経路に pending を積む（バッチ収集の間だけ有効）
        for n in path:
            n.pending += 1
        return node, path

    sims_done = 0
    if is_tqdm:
        pbar = tqdm.tqdm(total=iterations)
    while sims_done < iterations:
        # ---- 葉をまとめて収集 ----
        leaves: List[DecisionNode] = []
        paths: Dict[int, List[Node]] = {}  # id(leaf) -> path
        uniq_states: Dict[int, SolitireSearchState] = (
            {}
        )  # state_key -> state（キャッシュ未命中のみ）

        b = min(batch_size, iterations - sims_done)
        for _ in range(b):
            leaf, path = _select_and_expand_one()
            if is_verbose:
                for node in path:
                    if isinstance(node, ChanceNode):
                        move = node.action_from_parent
                        print(f"{move}", end="->" if node != path[-1] else "")
                print("")
            paths[id(leaf)] = path
            leaves.append(leaf)

            k = leaf.state.hash()
            if leaf.state.is_terminal():
                value_cache[k] = leaf.state.get_terminal_value()
            else:
                if k not in value_cache:
                    uniq_states[k] = leaf.state

        # ---- 未評価の状態だけ一括推論 ----
        if uniq_states:
            states = list(uniq_states.values())
            vals = await V_batch(states)  # <- バッチ推論を1回
            if len(vals) != len(states):
                raise ValueError(
                    "V_batch は states と同じ長さのリストを返してください。"
                )
            for st, v in zip(states, vals):
                value_cache[st.hash()] = float(v)

        # ---- それぞれの葉に値を割り当てて逆伝播 ----
        for leaf in leaves:
            v = value_cache[leaf.state.hash()]

            # pending を戻し、実値でバックアップ
            path = paths[id(leaf)]
            for n in path:
                n.pending -= 1

            # back propagate leaf to root
            node: DecisionNode = leaf
            while node is not None:
                node.N += 1
                node.W += v
                node = node.parent

        sims_done += b
        if is_tqdm:
            pbar.update(b)

    # 根で最終手を決定：訪問回数最大（子=チャンスノード）
    if not root.children:
        return None
    best_action, _ = max(root.children.items(), key=lambda kv: kv[1].N)
    return best_action


async def estimate_move_for_game_by_mcts(
    game: SolitireGame,
    executer: SolitireValueExecuter,
    iterations: int = 2000,
    batch_size: int = 64,
    c_ucb: float = 1.4,
    is_verbose: bool = False,
    is_tqdm: bool = False,
) -> Optional[SolitireSearchAction]:
    """
    ソリティアゲームに対してMCTSで次の手を推定する。
    返り値: 採用する action（訪問回数最大）
    """
    lw_game = SolitireLightWeightGame.from_solitire_game(game)
    root_state = SolitireSearchState(lw_game)

    async def v_batch(ss: List[SolitireSearchState]) -> List[float]:
        states = [s.game.states.first for s in ss]
        vals = await executer.execute(states)
        normalized_vals = [val / 5.2 * 2.0 - 1.0 for val in vals]  # [-1,1]に正規化
        clipped_vals = [max(-1.0, min(1.0, v)) for v in normalized_vals]
        return clipped_vals

    return await mcts_with_value_and_chance_batched(
        root_state,
        v_batch,
        iterations,
        batch_size,
        c_ucb,
        is_verbose=is_verbose,
        is_tqdm=is_tqdm,
    )
