import copy
from array import array
from typing import List, Tuple

from bitset import BitSet
from solitier_game import (
    Card,
    FromFoundation,
    FromPile,
    SolitireCardPositionFrom,
    SolitireCardPositionTo,
    SolitireGame,
    SolitireState,
    Suit,
    ToFoundation,
    ToPile,
)
from solitier_token import (
    Token,
    TokenClose,
    TokenClosedPile,
    TokenClosedStock,
    TokenCLS,
    TokenEOS,
    TokenFoundation,
    TokenOpen,
    TokenOpenedPile,
    TokenOpenedStock,
    TokenWaste,
)


def invert_color_suits(suit_index: int) -> array:
    match suit_index:
        case 0 | 3:  # Spade or Club
            return array("I", [1, 2])  # Heart or Diamond
        case 1 | 2:  # Heart or Diamond
            return array("I", [0, 3])  # Spade or Club
        case _:
            raise ValueError(f"Invalid suit index: {suit_index}")


def card_to_int(card: Card) -> int:
    return card.suit.value * 13 + (card.number - 1)


def suit_number_to_int(suit_index: int, number: int) -> int:
    return suit_index * 13 + (number - 1)


def int_to_card(index: int) -> Card:
    suit = Suit(index // 13)
    number = index % 13 + 1
    return Card(suit, number)


def int_to_card_suit(index: int) -> int:
    suit_index = index // 13
    return suit_index


def int_to_card_number(index: int) -> int:
    number = index % 13 + 1
    return number


class SolitireLightWeightState:
    def __init__(self, state: SolitireState):
        self._token_indices = None
        self._hash_cache = None
        self.waste_card_indices = array(
            "I", [card_to_int(card_state.card) for card_state in state.waste_stock]
        )
        self.foundation_counts = array("I", [0] * 4)
        for suit in state.foundation:
            self.foundation_counts[suit.value] = len(state.foundation[suit])
        self.closed_pile_counts = array(
            "I",
            [
                len([card_state.card for card_state in piles if not card_state.is_open])
                for piles in state.piless
            ],
        )
        self.opened_pile_card_indicess = []
        for piles in state.piless:
            self.opened_pile_card_indicess.append(
                array(
                    "i",
                    [
                        card_to_int(card_state.card)
                        for card_state in piles
                        if card_state.is_open
                    ],
                )
            )
        closed_cards = []
        for piles in state.piless:
            closed_cards.extend(
                [card_state.card for card_state in piles if not card_state.is_open]
            )
        if state.stock_cycle_count > 0:
            self.opened_stock_card_indices = array(
                "I", [card_to_int(card_state.card) for card_state in state.stock]
            )
            self.closed_stock_count = 0
        else:
            self.opened_stock_card_indices = array("I")
            self.closed_stock_count = len(state.stock)
            closed_cards.extend([card_state.card for card_state in state.stock])
        self.closed_card_indices = BitSet()
        for closed_card in closed_cards:
            self.closed_card_indices = self.closed_card_indices.add(
                card_to_int(closed_card)
            )
        self.stock_cycle_count = state.stock_cycle_count

        # Card -> FromPosition のマッピング
        self._card_from_positions = [None] * 52
        if len(self.waste_card_indices) > 0:
            self._set_card_from_position(self.waste_card_indices[-1], "FromWaste")
        for suit_index, foundation_count in enumerate(self.foundation_counts):
            if foundation_count > 0:
                suit = Suit(suit_index)
                self._set_card_from_position(
                    suit_number_to_int(suit_index, foundation_count),
                    FromFoundation(suit),
                )
        for col, (closed_pile_count, open_pile_card_indices) in enumerate(
            zip(self.closed_pile_counts, self.opened_pile_card_indicess)
        ):
            for row, card_index in enumerate(open_pile_card_indices):
                self._set_card_from_position(
                    card_index, FromPile(col, row + closed_pile_count)
                )

    def _set_card_from_position(
        self, card_index: int, from_position: SolitireCardPositionFrom
    ):
        self._card_from_positions[card_index] = from_position

    def _get_card_from_position(self, card_index: int) -> SolitireCardPositionFrom:
        return self._card_from_positions[card_index]

    def enumerate_valid_moves(self) -> List[tuple[Card, SolitireCardPositionTo]]:
        valid_moves = []
        if (
            len(self.opened_stock_card_indices) + self.closed_stock_count > 0
            or len(self.waste_card_indices) > 0
        ):
            valid_moves.append(("FromStock", "ToStock"))
        # To Foundation
        for suit_index, foundation_count in enumerate(self.foundation_counts):
            if foundation_count < 13:
                card_index = suit_number_to_int(suit_index, foundation_count + 1)
                from_position = self._get_card_from_position(card_index)
                if from_position is not None:
                    suit = Suit(suit_index)
                    match from_position:
                        case FromPile(col, row):
                            if (
                                row
                                == self.closed_pile_counts[col]
                                + len(self.opened_pile_card_indicess[col])
                                - 1
                            ):
                                valid_moves.append((from_position, ToFoundation(suit)))
                        case _:
                            valid_moves.append((from_position, ToFoundation(suit)))
        # To Pile
        for col, (closed_pile_count, open_pile_card_indices) in enumerate(
            zip(self.closed_pile_counts, self.opened_pile_card_indicess)
        ):
            if len(open_pile_card_indices) > 0:
                to_card_index = open_pile_card_indices[-1]
                to_card_suit_index = int_to_card_suit(to_card_index)
                to_card_number = int_to_card_number(to_card_index)
                if to_card_number > 1:
                    possible_from_card_suit_indices = invert_color_suits(
                        to_card_suit_index
                    )
                    for (
                        possible_from_card_suit_index
                    ) in possible_from_card_suit_indices:
                        from_position = self._get_card_from_position(
                            suit_number_to_int(
                                possible_from_card_suit_index, to_card_number - 1
                            ),
                        )
                        if from_position is not None:
                            valid_moves.append((from_position, ToPile(col)))
            else:
                for possible_from_card_suit_index in range(4):
                    from_position = self._get_card_from_position(
                        suit_number_to_int(possible_from_card_suit_index, 13)
                    )
                    if from_position is not None:
                        valid_moves.append((from_position, ToPile(col)))
        return valid_moves

    def _move_stock_to_stock(self) -> List["SolitireLightWeightState"]:
        if len(self.opened_stock_card_indices) > 0:
            new_state = self.copy()
            if len(new_state.waste_card_indices) > 0:
                new_state._set_card_from_position(
                    new_state.waste_card_indices[-1], None
                )
            new_state.waste_card_indices.append(
                new_state.opened_stock_card_indices.pop(0)
            )
            new_state._set_card_from_position(
                new_state.waste_card_indices[-1], "FromWaste"
            )
            return [new_state]
        elif self.closed_stock_count > 0:
            new_states = []
            for closed_card_index in self.closed_card_indices:
                new_state = self.copy()
                if len(new_state.waste_card_indices) > 0:
                    new_state._set_card_from_position(
                        new_state.waste_card_indices[-1], None
                    )
                new_state.waste_card_indices.append(closed_card_index)
                new_state.closed_stock_count -= 1
                new_state.closed_card_indices = new_state.closed_card_indices.discard(
                    closed_card_index
                )
                new_state._set_card_from_position(
                    new_state.waste_card_indices[-1], "FromWaste"
                )
                new_states.append(new_state)
            return new_states
        elif len(self.waste_card_indices) > 0:
            new_state = self.copy()
            new_state._set_card_from_position(new_state.waste_card_indices[-1], None)
            new_state.opened_stock_card_indices = array(
                "I", [card_index for card_index in new_state.waste_card_indices]
            )
            new_state.waste_card_indices = array("I")
            new_state.stock_cycle_count += 1
            return new_state._move_stock_to_stock()
        else:
            return []

    def _move_pile_to_pile(
        self, from_pile: Tuple[int, int], to_col: int
    ) -> List["SolitireLightWeightState"]:
        from_col, from_row = from_pile
        from_closed_pile_count = self.closed_pile_counts[from_col]
        from_opened_row = from_row - from_closed_pile_count
        from_opended_pile_move_card_indices = self.opened_pile_card_indicess[from_col][
            from_opened_row:
        ]

        base_new_state = self.copy()
        for card_index in from_opended_pile_move_card_indices:
            base_new_state._set_card_from_position(card_index, None)
        del base_new_state.opened_pile_card_indicess[from_col][from_opened_row:]
        to_col_offset = self.closed_pile_counts[to_col] + len(
            base_new_state.opened_pile_card_indicess[to_col]
        )
        base_new_state.opened_pile_card_indicess[to_col].extend(
            from_opended_pile_move_card_indices
        )
        for i, card_index in enumerate(from_opended_pile_move_card_indices):
            base_new_state._set_card_from_position(
                card_index, FromPile(to_col, i + to_col_offset)
            )

        if from_opened_row == 0 and from_closed_pile_count > 0:
            new_states = []
            for closed_card_index in self.closed_card_indices:
                new_state = base_new_state.copy()
                new_state.closed_pile_counts[from_col] -= 1
                new_state.closed_card_indices = new_state.closed_card_indices.discard(
                    closed_card_index
                )
                new_state.opened_pile_card_indicess[from_col].insert(
                    0, closed_card_index
                )
                new_state._set_card_from_position(
                    closed_card_index,
                    FromPile(from_col, new_state.closed_pile_counts[from_col]),
                )
                new_states.append(new_state)
            return new_states
        else:
            return [base_new_state]

    def _move_waste_to_pile(self, to_col: int) -> List["SolitireLightWeightState"]:
        new_state = self.copy()
        card_index = new_state.waste_card_indices.pop(-1)
        new_state._set_card_from_position(card_index, None)
        if len(new_state.waste_card_indices) > 0:
            new_state._set_card_from_position(
                new_state.waste_card_indices[-1], "FromWaste"
            )
        new_state.opened_pile_card_indicess[to_col].append(card_index)
        new_state._set_card_from_position(
            card_index,
            FromPile(
                to_col,
                new_state.closed_pile_counts[to_col]
                + len(new_state.opened_pile_card_indicess[to_col])
                - 1,
            ),
        )
        return [new_state]

    def _move_pile_to_foundation(
        self, from_pile: Tuple[int, int], to_suit: Suit
    ) -> List["SolitireLightWeightState"]:
        from_col, from_row = from_pile
        to_suit_index = to_suit.value

        base_new_state = self.copy()
        card_index = base_new_state.opened_pile_card_indicess[from_col].pop(-1)
        base_new_state._set_card_from_position(card_index, FromFoundation(to_suit))
        if base_new_state.foundation_counts[to_suit_index] != 0:
            base_new_state._set_card_from_position(
                suit_number_to_int(
                    to_suit_index, base_new_state.foundation_counts[to_suit_index]
                ),
                None,
            )
        base_new_state.foundation_counts[to_suit_index] += 1

        if (
            len(base_new_state.opened_pile_card_indicess[from_col]) == 0
            and base_new_state.closed_pile_counts[from_col] > 0
        ):
            new_states = []
            for closed_card_index in self.closed_card_indices:
                new_state = base_new_state.copy()
                new_state.closed_pile_counts[from_col] -= 1
                new_state.closed_card_indices = new_state.closed_card_indices.discard(
                    closed_card_index
                )
                new_state.opened_pile_card_indicess[from_col].insert(
                    0, closed_card_index
                )
                new_state._set_card_from_position(
                    closed_card_index,
                    FromPile(from_col, new_state.closed_pile_counts[from_col]),
                )
                new_states.append(new_state)
            return new_states
        else:
            return [base_new_state]

    def _move_foundation_to_pile(
        self, from_suit: Suit, to_col: int
    ) -> List["SolitireLightWeightState"]:
        from_suit_index = from_suit.value
        new_state = self.copy()
        card_index = suit_number_to_int(
            from_suit_index, new_state.foundation_counts[from_suit_index]
        )
        new_state.foundation_counts[from_suit_index] -= 1
        if new_state.foundation_counts[from_suit_index] != 0:
            new_state._set_card_from_position(
                suit_number_to_int(
                    from_suit_index, new_state.foundation_counts[from_suit_index]
                ),
                FromFoundation(from_suit),
            )
        new_state.opened_pile_card_indicess[to_col].append(card_index)
        new_state._set_card_from_position(
            card_index,
            FromPile(
                to_col,
                new_state.closed_pile_counts[to_col]
                + len(new_state.opened_pile_card_indicess[to_col])
                - 1,
            ),
        )
        return [new_state]

    def _move_waste_to_foundation(
        self, to_suit: Suit
    ) -> List["SolitireLightWeightState"]:
        to_suit_index = to_suit.value
        new_state = self.copy()
        card_index = new_state.waste_card_indices.pop(-1)
        new_state._set_card_from_position(card_index, FromFoundation(to_suit))
        if new_state.foundation_counts[to_suit_index] != 0:
            new_state._set_card_from_position(
                suit_number_to_int(
                    to_suit_index, new_state.foundation_counts[to_suit_index]
                ),
                None,
            )
        if len(new_state.waste_card_indices) > 0:
            new_state._set_card_from_position(
                new_state.waste_card_indices[-1], "FromWaste"
            )
        new_state.foundation_counts[to_suit_index] += 1
        return [new_state]

    def move_uncertain_states(
        self,
        from_position: SolitireCardPositionFrom,
        to_position: SolitireCardPositionTo,
    ) -> List["SolitireLightWeightState"]:
        match from_position, to_position:
            case "FromStock", "ToStock":
                return self._move_stock_to_stock()
            case "FromWaste", ToPile(to_col):
                return self._move_waste_to_pile(to_col)
            case "FromWaste", ToFoundation(to_suit):
                return self._move_waste_to_foundation(to_suit)
            case FromPile(from_col, from_row), ToPile(to_col):
                return self._move_pile_to_pile((from_col, from_row), to_col)
            case FromPile(from_col, from_row), ToFoundation(to_suit):
                return self._move_pile_to_foundation((from_col, from_row), to_suit)
            case FromFoundation(from_suit), ToPile(to_col):
                return self._move_foundation_to_pile(from_suit, to_col)
            case _, _:
                raise ValueError(f"Invalid move from {from_position} to {to_position}")

    def is_same_state(
        self, other: "SolitireLightWeightState", is_compatibility: bool = True
    ) -> bool:
        if self._token_indices is None:
            self.to_token_indices()
        if other._token_indices is None:
            other.to_token_indices()
        is_same_token_indices = (
            self._token_indices == other._token_indices
            if self._hash_cache == other._hash_cache
            else False
        )
        # Stock comparison open to closed fallback for compatibility
        if (
            is_compatibility
            and self.closed_stock_count == len(other.opened_stock_card_indices)
            and other.closed_stock_count == len(self.opened_stock_card_indices)
        ):
            is_same = self.is_same_state_compare(other)
        else:
            is_same = is_same_token_indices
        return is_same

    def is_same_state_compare(self, other: "SolitireLightWeightState") -> bool:
        if self.waste_card_indices != other.waste_card_indices:
            return False
        if self.foundation_counts != other.foundation_counts:
            return False
        if self.closed_pile_counts != other.closed_pile_counts:
            return False
        if self.opened_pile_card_indicess != other.opened_pile_card_indicess:
            return False
        if (
            self.closed_stock_count != other.closed_stock_count
            or self.opened_stock_card_indices != other.opened_stock_card_indices
        ):
            if not (
                self.closed_stock_count == len(other.opened_stock_card_indices)
                and other.closed_stock_count == len(self.opened_stock_card_indices)
            ):
                return False
        return True

    def is_all_open(self) -> bool:
        return len(self.closed_card_indices) == 0

    def open_count(self) -> int:
        foundation_count = sum(self.foundation_counts)
        opened_pile_count = sum(
            len(card_indices) for card_indices in self.opened_pile_card_indicess
        )
        return foundation_count + opened_pile_count

    def verify(self) -> bool:
        card_from_positions = [None] * 52

        def _set_card_from_position(
            card_index: int, from_position: SolitireCardPositionFrom
        ):
            card_from_positions[card_index] = from_position

        def _get_card_from_position(card_index: int) -> SolitireCardPositionFrom:
            return card_from_positions[card_index]

        if len(self.waste_card_indices) > 0:
            _set_card_from_position(self.waste_card_indices[-1], "FromWaste")
        for suit_index, foundation_count in enumerate(self.foundation_counts):
            if foundation_count > 0:
                suit = Suit(suit_index)
                _set_card_from_position(
                    suit_number_to_int(suit_index, foundation_count),
                    FromFoundation(suit),
                )
        for col, (closed_pile_count, open_pile_card_indices) in enumerate(
            zip(self.closed_pile_counts, self.opened_pile_card_indicess)
        ):
            for row, card_index in enumerate(open_pile_card_indices):
                _set_card_from_position(
                    card_index, FromPile(col, row + closed_pile_count)
                )
        for suit in Suit:
            for number in range(1, 14):
                card = Card(suit, number)
                verify_pos = _get_card_from_position(card_to_int(card))
                cls_pos = self._get_card_from_position(card_to_int(card))
                if verify_pos != cls_pos:
                    print(f"Card {card} position mismatch: {verify_pos} != {cls_pos}")
        for suit in Suit:
            for number in range(1, 14):
                card = Card(suit, number)
                verify_pos = _get_card_from_position(card_to_int(card))
                cls_pos = self._get_card_from_position(card_to_int(card))
                if verify_pos != cls_pos:
                    return False
        if len(self.closed_card_indices) != self.closed_stock_count + sum(
            self.closed_pile_counts
        ):
            print(
                f"Closed cards count mismatch: {len(self.closed_card_indices)} !=",
                f"{self.closed_stock_count + sum(self.closed_pile_counts)}",
            )
            return False
        cards = []
        cards.extend([int_to_card(index) for index in self.closed_card_indices])
        cards.extend([int_to_card(index) for index in self.opened_stock_card_indices])
        cards.extend([int_to_card(index) for index in self.waste_card_indices])
        for suit_index, foundation_count in enumerate(self.foundation_counts):
            suit = Suit(suit_index)
            for number in range(1, foundation_count + 1):
                cards.append(Card(suit, number))
        for opened_pile_card_indices in self.opened_pile_card_indicess:
            cards.extend([int_to_card(index) for index in opened_pile_card_indices])
        cards.sort(key=lambda c: (c.suit.value, c.number))
        ref_cards = [Card(suit, number) for suit in Suit for number in range(1, 14)]
        ref_cards.sort(key=lambda c: (c.suit.value, c.number))
        if cards != ref_cards:
            print(f"Cards mismatch: {cards} != {ref_cards}")
            return False
        if not isinstance(self.opened_stock_card_indices, array):
            print("opened_stock_card_indices is not array")
            return False
        if not isinstance(self.waste_card_indices, array):
            print("waste_card_indices is not array")
            return False

        return True

    def to_tokens(self) -> List[Token]:
        tokens = []
        tokens.append(TokenCLS())
        for suit_index, foundation_count in enumerate(self.foundation_counts):
            tokens.append(TokenFoundation(Suit(suit_index)))
            for number in range(1, foundation_count + 1):
                tokens.append(TokenOpen(Card(Suit(suit_index), number)))
        if self.stock_cycle_count > 0:
            tokens.append(TokenOpenedStock())
            for card_index in self.opened_stock_card_indices:
                tokens.append(TokenOpen(int_to_card(card_index)))
        else:
            tokens.append(TokenClosedStock())
            for _ in range(self.closed_stock_count):
                tokens.append(TokenClose())
        tokens.append(TokenWaste())
        for card_index in self.waste_card_indices:
            tokens.append(TokenOpen(int_to_card(card_index)))
        for closed_pile_count, opened_pile_card_indices in zip(
            self.closed_pile_counts, self.opened_pile_card_indicess
        ):
            tokens.append(TokenClosedPile())
            for _ in range(closed_pile_count):
                tokens.append(TokenClose())
            tokens.append(TokenOpenedPile())
            for card_index in opened_pile_card_indices:
                tokens.append(TokenOpen(int_to_card(card_index)))
        tokens.append(TokenEOS())
        return tokens

    def to_token_indices(self) -> array:
        tokens = array("I")
        tokens.append(52 + 4 + 1)  # TokenCLS
        for suit_index, foundation_count in enumerate(self.foundation_counts):
            tokens.append(52 + 1 + suit_index)  # TokenFoundation
            for number in range(1, foundation_count + 1):
                tokens.append(suit_index * 13 + number - 1)
        if self.stock_cycle_count > 0:
            tokens.append(52 + 4 + 3)  # TokenOpenedStock
            for card_index in self.opened_stock_card_indices:
                tokens.append(card_index)
        else:
            tokens.append(52 + 4 + 4)  # TokenClosedStock
            for _ in range(self.closed_stock_count):
                tokens.append(52)
        tokens.append(52 + 4 + 5)  # TokenWaste
        for card_index in self.waste_card_indices:
            tokens.append(card_index)
        for closed_pile_count, opened_pile_card_indices in zip(
            self.closed_pile_counts, self.opened_pile_card_indicess
        ):
            tokens.append(52 + 4 + 7)  # TokenClosedPile
            for _ in range(closed_pile_count):
                tokens.append(52)
            tokens.append(52 + 4 + 6)  # TokenOpenedPile
            for card_index in opened_pile_card_indices:
                tokens.append(card_index)
        tokens.append(52 + 4 + 2)  # TokenEOS
        self._token_indices = tokens
        self._hash_cache = hash(tuple(self._token_indices))
        return tokens

    def hash(self):
        if self._token_indices is None:
            self.to_token_indices()
        return self._hash_cache

    def copy(self) -> "SolitireLightWeightState":
        new_state = copy.copy(self)
        new_state.waste_card_indices = self.waste_card_indices[:]
        new_state.foundation_counts = self.foundation_counts[:]
        new_state.closed_pile_counts = self.closed_pile_counts[:]
        new_state.opened_pile_card_indicess = [
            card_indices[:] for card_indices in self.opened_pile_card_indicess
        ]
        new_state.opened_stock_card_indices = self.opened_stock_card_indices[:]
        new_state._card_from_positions = self._card_from_positions.copy()
        new_state._token_indices = None
        new_state._hash_cache = None
        return new_state


class SolitireLightWeightGame:
    def __init__(self, initial_states: List[SolitireLightWeightState]):
        self.states = initial_states
        self.valid_moves_cache = None
        self.moved_states_cache = {}
        self.is_all_open_cache = None
        self.open_count_cache = None

    def is_same_as_any_state(
        self, state: SolitireLightWeightState, is_compatibility: bool = True
    ) -> bool:
        return any(
            s.is_same_state(state, is_compatibility=is_compatibility)
            for s in self.states
        )

    def enumerate_valid_moves_excluding_same_state(
        self, is_compatibility: bool = True
    ) -> List[Tuple[SolitireCardPositionFrom, SolitireCardPositionTo]]:
        if self.valid_moves_cache is not None:
            return self.valid_moves_cache
        valid_moves = self.states[-1].enumerate_valid_moves()
        valid_moves_excluding_same_state = []
        for from_position, to_position in valid_moves:
            new_states = self.states[-1].move_uncertain_states(
                from_position, to_position
            )
            assert len(new_states) > 0
            self.moved_states_cache[(from_position, to_position)] = new_states
            # open move is always new state
            if len(new_states) > 1:
                valid_moves_excluding_same_state.append((from_position, to_position))
            # elif not new_states[0].hash() in state_set:
            elif not any(
                new_states[0].is_same_state(s, is_compatibility=is_compatibility)
                for s in self.states
            ):
                valid_moves_excluding_same_state.append((from_position, to_position))
        self.valid_moves_cache = valid_moves_excluding_same_state
        return valid_moves_excluding_same_state

    # unchecked
    def move_uncertain_states(
        self,
        from_position: SolitireCardPositionFrom,
        to_position: SolitireCardPositionTo,
    ) -> List["SolitireLightWeightGame"]:
        if (from_position, to_position) in self.moved_states_cache:
            new_states = self.moved_states_cache[(from_position, to_position)]
        else:
            new_states = self.states[-1].move_uncertain_states(
                from_position, to_position
            )
            self.moved_states_cache[(from_position, to_position)] = new_states
        new_games = []
        for new_state in new_states:
            if new_state is not None and not self.is_same_as_any_state(new_state):
                new_game = self.copy()
                new_game.states.append(new_state)
                new_games.append(new_game)
        return new_games

    def get_last_state(self) -> SolitireLightWeightState:
        return self.states[-1]

    def is_all_open(self) -> bool:
        if self.is_all_open_cache is None:
            self.is_all_open_cache = self.states[-1].is_all_open()
        return self.is_all_open_cache

    def open_count(self) -> int:
        if self.open_count_cache is None:
            self.open_count_cache = self.states[-1].open_count()
        return self.open_count_cache

    def verify(self) -> bool:
        return all(state.verify() for state in self.states)

    @classmethod
    def from_solitire_game(cls, game: SolitireGame) -> "SolitireLightWeightGame":
        return cls([SolitireLightWeightState(state) for state in game.states])

    def copy(self) -> "SolitireLightWeightGame":
        new_game = copy.copy(self)
        new_game.states = self.states.copy()
        new_game.moved_states_cache = {}
        new_game.is_all_open_cache = None
        new_game.open_count_cache = None
        new_game.valid_moves_cache = None
        return new_game
