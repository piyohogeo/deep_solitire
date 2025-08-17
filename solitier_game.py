import copy
import pickle
import random
from collections import deque
from dataclasses import dataclass, replace
from enum import Enum
from typing import Literal, Optional, Tuple, Union


class Suit(Enum):
    C = 0
    D = 1
    H = 2
    S = 3

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.name


@dataclass(frozen=True)
class Card:
    suit: Suit
    number: int


@dataclass(frozen=True)
class CardState:
    card: Card
    is_open: bool = False

    def __repr__(self):
        return f"CardState(card={self.card}, is_open={self.is_open})"


def all_cards() -> list[Card]:
    return [Card(suit=suit, number=number) for suit in Suit for number in range(1, 14)]


def is_suit_black(Suit: Suit) -> bool:
    if Suit == Suit.S or Suit == Suit.C:
        return True
    return False


def is_valid_pile_move(from_card: Card, to_card: Union[Card, None]) -> bool:
    if to_card is None:
        return from_card.number == 13
    if is_suit_black(from_card.suit) != is_suit_black(to_card.suit):
        return from_card.number == to_card.number - 1
    return False


def is_valid_foundation_move(
    from_card: Card, to_suit: Suit, to_card: Union[Card, None]
) -> bool:
    if from_card.suit != to_suit:
        return False
    if to_card is None:
        return from_card.number == 1
    return from_card.suit == to_card.suit and from_card.number == to_card.number + 1


PILES_COUNT = 7

FromStock = Literal["FromStock"]
FromWaste = Literal["FromWaste"]


@dataclass(frozen=True)
class FromFoundation:
    suit: Suit


@dataclass(frozen=True)
class FromPile:
    col: int
    row: int


SolitireCardPositionFrom = Union[FromStock, FromWaste, FromFoundation, FromPile]

ToStock = Literal["ToStock"]


@dataclass(frozen=True)
class ToFoundation:
    suit: Suit


@dataclass(frozen=True)
class ToPile:
    col: int


SolitireCardPositionTo = Union[ToStock, ToFoundation, ToPile]


class SolitireState:
    def __init__(self):
        cards = all_cards()
        random.shuffle(cards)
        self.piless: list[list[CardState]] = [[] for _ in range(7)]
        for i in range(PILES_COUNT):
            for j in range(i + 1):
                card = cards.pop()
                self.piless[i].append(CardState(card, is_open=j == i))
        self.stock: list[CardState] = [CardState(card, is_open=False) for card in cards]
        self.waste_stock: list[CardState] = []
        self.foundation: dict[Suit, list[CardState]] = {suit: [] for suit in Suit}
        self.stock_cycle_count = 0

    def _is_valid_open_stock(self):
        if len(self.stock) > 0:
            return True
        elif len(self.waste_stock) > 0:
            return True
        return False

    def _open_stock_(self) -> bool:
        if len(self.stock) > 0:
            card_state = self.stock.pop(0)
            card_state = replace(card_state, is_open=True)
            self.waste_stock.append(card_state)
            return self.stock_cycle_count == 0
        elif len(self.waste_stock) > 0:
            self.stock = self.waste_stock
            self.waste_stock = []
            self.stock = [
                replace(card_state, is_open=False) for card_state in self.stock
            ]
            self.stock_cycle_count += 1
            return self._open_stock_()
        else:
            return False

    def _is_valid_move_pile_to_pile(
        self, from_pile: Tuple[int, int], to_pile: int
    ) -> bool:
        from_col, from_row = from_pile
        if from_col == to_pile:
            return False
        if 0 <= from_col < len(self.piless) and 0 <= from_row < len(
            self.piless[from_col]
        ):
            from_card_states = self.piless[from_col][from_row:]
            if not from_card_states[0].is_open:
                return False
            to_card_state = self.piless[to_pile][-1] if self.piless[to_pile] else None
            return is_valid_pile_move(
                from_card_states[0].card, to_card_state.card if to_card_state else None
            )
        return False

    def _move_pile_to_pile_(self, from_pile: Tuple[int, int], to_pile: int) -> bool:
        from_col, from_row = from_pile
        from_card_states = self.piless[from_col][from_row:]
        self.piless[to_pile].extend(from_card_states)
        self.piless[from_col] = self.piless[from_col][:from_row]
        if len(self.piless[from_col]) > 0 and not self.piless[from_col][-1].is_open:
            self.piless[from_col][-1] = replace(self.piless[from_col][-1], is_open=True)
            return True
        return False

    def _is_valid_move_waste_to_pile(self, to_pile: int) -> bool:
        if len(self.waste_stock) > 0:
            card_state = self.waste_stock[-1]
            if not card_state.is_open:
                return False
            to_card_state = self.piless[to_pile][-1] if self.piless[to_pile] else None
            return is_valid_pile_move(
                card_state.card, to_card_state.card if to_card_state else None
            )
        return False

    def _move_waste_to_pile_(self, to_pile: int) -> bool:
        card_state = self.waste_stock.pop(-1)
        self.piless[to_pile].append(card_state)
        return False

    def _is_valid_move_pile_to_foundation(
        self, from_pile: Tuple[int, int], to_suit: Suit
    ) -> bool:
        from_col, from_row = from_pile
        if 0 <= from_col < len(self.piless) and self.piless[from_col]:
            if len(self.piless[from_col]) == 0:
                return False
            if from_row != len(self.piless[from_col]) - 1:
                return False
            card_state = self.piless[from_col][from_row]
            card = card_state.card
            return is_valid_foundation_move(
                card,
                to_suit,
                (
                    self.foundation[to_suit][-1].card
                    if self.foundation[to_suit]
                    else None
                ),
            )
        return False

    def _move_pile_to_foundation_(self, from_pile: int, to_suit: Suit) -> bool:
        card_state = self.piless[from_pile].pop(-1)
        self.foundation[to_suit].append(card_state)
        if len(self.piless[from_pile]) > 0 and not self.piless[from_pile][-1].is_open:
            self.piless[from_pile][-1] = replace(
                self.piless[from_pile][-1], is_open=True
            )
            return True
        return False

    def _is_valid_move_foundation_to_pile(self, suit: Suit, to_pile: int) -> bool:
        if self.foundation[suit]:
            card_state = self.foundation[suit][-1]
            to_card_state = self.piless[to_pile][-1] if self.piless[to_pile] else None
            return is_valid_pile_move(
                card_state.card, to_card_state.card if to_card_state else None
            )
        return False

    def _move_foundation_to_pile_(self, suit: Suit, to_pile: int) -> bool:
        card_state = self.foundation[suit].pop(-1)
        self.piless[to_pile].append(card_state)
        return False

    def _is_valid_move_waste_to_foundation(self, to_suit: Suit) -> bool:
        if len(self.waste_stock) > 0:
            card_state = self.waste_stock[-1]
            return is_valid_foundation_move(
                card_state.card,
                to_suit,
                (
                    self.foundation[to_suit][-1].card
                    if self.foundation[to_suit]
                    else None
                ),
            )
        return False

    def _move_waste_to_foundation_(self, to_suit: Suit) -> bool:
        card_state = self.waste_stock.pop(-1)
        self.foundation[to_suit].append(card_state)
        return False

    def _extract_all_closed_cards(self) -> list[Card]:
        closed_cards = []
        for piles in self.piless:
            for card_state in piles:
                if not card_state.is_open:
                    closed_cards.append(card_state.card)
        if self.stock_cycle_count == 0:
            stocked_cards = [card_state.card for card_state in self.stock]
        else:
            stocked_cards = []
        closed_cards.extend(stocked_cards)
        return closed_cards

    def _replace_closed_cards_by_map_(self, card_map: dict[Card, Card]):
        """
        Replace closed cards in the state with the corresponding cards from the map.
        """
        for piles in self.piless:
            for i, card_state in enumerate(piles):
                if not card_state.is_open:
                    piles[i] = CardState(card_map[card_state.card], is_open=False)
        if self.stock_cycle_count == 0:
            for i, card_state in enumerate(self.stock):
                if not card_state.is_open:
                    self.stock[i] = CardState(card_map[card_state.card], is_open=False)

    def build_uncertain_closed_states(self) -> list["SolitireState"]:
        """
        Build a list of uncertain states by replacing closed cards with all possible cards.
        """
        closed_cards = self._extract_all_closed_cards()
        if len(closed_cards) <= 1:
            return [copy.deepcopy(self)]
        target_cards = deque(closed_cards)
        uncertain_states = []
        for i in range(len(closed_cards)):
            card_map = {
                closed_card: target_card
                for closed_card, target_card in zip(closed_cards, target_cards)
            }
            new_state = copy.deepcopy(self)
            new_state._replace_closed_cards_by_map_(card_map)
            uncertain_states.append(new_state)
            target_cards.rotate(-1)
        return uncertain_states

    def resolve_card_from(self, card: Card) -> Optional[SolitireCardPositionFrom]:
        for col, card_states in enumerate(self.piless):
            for row, card_state in enumerate(card_states):
                if card_state.is_open and card_state.card == card:
                    return FromPile(col, row)
        if len(self.stock) > 0 and card == self.stock[0].card:
            return "FromStock"
        if len(self.waste_stock) > 0 and card == self.waste_stock[-1].card:
            return "FromWaste"
        for suit, card_states in self.foundation.items():
            if len(card_states) > 0 and card_states[-1].card == card:
                return FromFoundation(suit)
        return None

    def resolve_card_to(self, card: Card) -> Optional[SolitireCardPositionTo]:
        for suit, card_states in self.foundation.items():
            if len(card_states) > 0 and card_states[-1].card == card:
                return ToFoundation(suit)
        for col, card_states in enumerate(self.piless):
            if len(card_states) > 0 and card_states[-1].card == card:
                return ToPile(col)
        if len(self.stock) > 0 and card == self.stock[0].card:
            return "ToStock"
        return None

    def get_card_state_at_from_position(
        self, position: SolitireCardPositionFrom
    ) -> Optional[CardState]:
        match position:
            case "FromStock":
                return self.stock[0] if self.stock else None
            case "FromWaste":
                return self.waste_stock[-1] if self.waste_stock else None
            case FromFoundation(suit=suit):
                return self.foundation[suit][-1] if self.foundation[suit] else None
            case FromPile(col=col, row=row):
                return (
                    self.piless[col][row]
                    if col < len(self.piless) and len(self.piless[col]) > row
                    else None
                )
        return None

    def get_card_state_at_to_position(
        self, position: SolitireCardPositionTo
    ) -> Optional[CardState]:
        match position:
            case "ToStock":
                return self.stock[0] if self.stock else None
            case ToFoundation(suit=suit):
                return self.foundation[suit][-1] if self.foundation[suit] else None
            case ToPile(col=col):
                return (
                    self.piless[col][-1]
                    if col < len(self.piless) and self.piless[col]
                    else None
                )
        return None

    def is_valid_move(
        self,
        from_position: SolitireCardPositionFrom,
        to_position: SolitireCardPositionTo,
    ) -> bool:
        match from_position:
            case "FromStock":
                match to_position:
                    case "ToStock":
                        return self._is_valid_open_stock()
                    case _:
                        return False
            case "FromWaste":
                match to_position:
                    case ToPile(col=col):
                        return self._is_valid_move_waste_to_pile(col)
                    case ToFoundation(suit=suit):
                        return self._is_valid_move_waste_to_foundation(suit)
                    case _:
                        return False
            case FromFoundation(suit=from_suit):
                match to_position:
                    case ToPile(col=col):
                        return self._is_valid_move_foundation_to_pile(from_suit, col)
                    case _:
                        return False
            case FromPile(col=from_col, row=from_row):
                match to_position:
                    case ToFoundation(suit=to_suit):
                        return self._is_valid_move_pile_to_foundation(
                            (from_col, from_row), to_suit
                        )
                    case ToPile(col=to_col):
                        return self._is_valid_move_pile_to_pile(
                            (from_col, from_row), to_col
                        )
                    case _:
                        return False
        return False

    def checked_move(
        self,
        from_position: SolitireCardPositionFrom,
        to_position: SolitireCardPositionTo,
    ) -> Optional["SolitireState"]:
        if not self.is_valid_move(from_position, to_position):
            return None
        return self.move(from_position, to_position)

    def move(
        self,
        from_position: SolitireCardPositionFrom,
        to_position: SolitireCardPositionTo,
    ) -> Optional["SolitireState"]:
        new_state = copy.deepcopy(self)
        new_state._move_(from_position, to_position)
        return new_state

    def move_uncertain_states(
        self,
        from_position: SolitireCardPositionFrom,
        to_position: SolitireCardPositionTo,
    ) -> list["SolitireState"]:
        uncertain_state = copy.deepcopy(self)
        if not uncertain_state._move_(from_position, to_position):
            return [uncertain_state]
        uncertain_states = self.build_uncertain_closed_states()

        for uncertain_state in uncertain_states:
            uncertain_state._move_(from_position, to_position)
        # remove states that are same opened card state
        return uncertain_states

    def _move_(
        self,
        from_position: SolitireCardPositionFrom,
        to_position: SolitireCardPositionTo,
    ) -> Optional[bool]:
        match from_position:
            case "FromStock":
                if to_position == "ToStock":
                    return self._open_stock_()
            case "FromWaste":
                match to_position:
                    case ToPile(col=to_col):
                        return self._move_waste_to_pile_(to_col)
                    case ToFoundation(suit=to_suit):
                        return self._move_waste_to_foundation_(to_suit)
            case FromFoundation(suit=from_suit):
                match to_position:
                    case ToPile(col=to_col):
                        return self._move_foundation_to_pile_(from_suit, to_col)
            case FromPile(col=from_col, row=from_row):
                match to_position:
                    case ToFoundation(suit=to_suit):
                        return self._move_pile_to_foundation_(from_col, to_suit)
                    case ToPile(col=to_col):
                        return self._move_pile_to_pile_((from_col, from_row), to_col)
        return None

    def enumerate_valid_moves_from(
        self, from_position: SolitireCardPositionFrom
    ) -> list[SolitireCardPositionTo]:
        valid_moves = []
        match from_position:
            case "FromStock":
                if self._is_valid_open_stock():
                    valid_moves.append("ToStock")
            case "FromWaste":
                for suit in Suit:
                    if self._is_valid_move_waste_to_foundation(suit):
                        valid_moves.append(ToFoundation(suit))
                for col in range(len(self.piless)):
                    if self._is_valid_move_waste_to_pile(col):
                        valid_moves.append(ToPile(col))
            case FromFoundation(suit=from_suit):
                for col in range(len(self.piless)):
                    if self._is_valid_move_foundation_to_pile(from_suit, col):
                        valid_moves.append(ToPile(col))
            case FromPile(col=from_col, row=from_row):
                for suit in Suit:
                    if self._is_valid_move_pile_to_foundation(
                        (from_col, from_row), suit
                    ):
                        valid_moves.append(ToFoundation(suit))
                for to_col in range(len(self.piless)):
                    if self._is_valid_move_pile_to_pile((from_col, from_row), to_col):
                        valid_moves.append(ToPile(to_col))
        return valid_moves

    def _enumerate_possible_from_positions(self) -> list[SolitireCardPositionFrom]:
        from_positions = []
        from_positions.append("FromStock")
        if len(self.waste_stock) > 0:
            from_positions.append("FromWaste")
        for suit in Suit:
            if self.foundation[suit]:
                from_positions.append(FromFoundation(suit))
        for col, card_states in enumerate(self.piless):
            for row, card_state in enumerate(card_states):
                if card_state.is_open:
                    from_positions.append(FromPile(col, row))
        return from_positions

    def enumerate_valid_moves(
        self,
    ) -> list[Tuple[SolitireCardPositionFrom, SolitireCardPositionTo]]:
        valid_moves = []
        for from_position in self._enumerate_possible_from_positions():
            for to_position in self.enumerate_valid_moves_from(from_position):
                valid_moves.append((from_position, to_position))
        return valid_moves

    def enumerate_valid_from_positions(
        self,
    ) -> list[SolitireCardPositionFrom]:
        valid_moves = self.enumerate_valid_moves()
        return list(set(from_position for from_position, _ in valid_moves))

    def open_count(self) -> int:
        count = 0
        for piles in self.piless:
            for card_state in piles:
                if card_state.is_open:
                    count += 1
        for suit in Suit:
            count += len(self.foundation[suit])
        return count

    def is_all_open(self) -> bool:
        return self.open_count() == len(all_cards())

    def is_same_state(self, other: "SolitireState") -> bool:
        return (
            self.piless == other.piless
            and self.stock == other.stock
            and self.waste_stock == other.waste_stock
            and self.foundation == other.foundation
        )

    def _is_same_opened_card_state(self, other: "SolitireState") -> bool:
        if len(self.piless) != len(other.piless):
            return False
        for piles, other_pile in zip(self.piless, other.piless):
            if len(piles) != len(other_pile):
                return False
            for card_state, other_card_state in zip(piles, other_pile):
                if card_state.is_open != other_card_state.is_open:
                    return False
                if card_state.is_open and card_state.card != other_card_state.card:
                    return False
        if len(self.waste_stock) != len(other.waste_stock):
            return False
        for card_state, other_card_state in zip(self.waste_stock, other.waste_stock):
            if card_state.is_open != other_card_state.is_open:
                return False
            if card_state.is_open and card_state.card != other_card_state.card:
                return False
        # check stock if stock_cycle_count > 0
        if self.stock_cycle_count > 0:
            if len(self.stock) != len(other.stock):
                return False
            for card_state, other_card_state in zip(self.stock, other.stock):
                if card_state.card != other_card_state.card:
                    return False

        return True


class SolitireGame:
    def __init__(self):
        self.state = SolitireState()
        self.states = [self.state]
        self.moves = []

    def is_success(self) -> bool:
        return self.state.is_all_open()

    def checked_move(
        self,
        from_position: SolitireCardPositionFrom,
        to_position: SolitireCardPositionTo,
    ) -> bool:
        if not self.state.is_valid_move(from_position, to_position):
            return False
        new_state = self.state.move(from_position, to_position)
        if new_state is None:
            return False
        self.states.append(new_state)
        self.state = new_state
        self.moves.append((from_position, to_position))
        return True

    def checked_move_excluding_same_state(
        self,
        from_position: SolitireCardPositionFrom,
        to_position: SolitireCardPositionTo,
    ) -> bool:
        if not self.state.is_valid_move(from_position, to_position):
            return False
        new_state = self.state.move(from_position, to_position)
        if new_state is None or self.is_same_as_any_states(new_state):
            return False
        self.states.append(new_state)
        self.state = new_state
        self.moves.append((from_position, to_position))
        return True

    def is_same_as_any_states(self, state: SolitireState) -> bool:
        return any(s.is_same_state(state) for s in self.states)

    def enumerate_valid_moves_from_excluding_same_state(
        self, from_position: SolitireCardPositionFrom
    ) -> list[SolitireCardPositionTo]:
        valid_moves = self.state.enumerate_valid_moves_from(from_position)
        valid_moves_excluding_same_state = []
        for to_position in valid_moves:
            new_state = self.state.move(from_position, to_position)
            if new_state is not None and not self.is_same_as_any_states(new_state):
                valid_moves_excluding_same_state.append(to_position)
        return valid_moves_excluding_same_state

    def enumerate_valid_moves_excluding_same_state(
        self,
    ) -> list[Tuple[SolitireCardPositionFrom, SolitireCardPositionTo]]:
        valid_moves = self.state.enumerate_valid_moves()
        valid_moves_excluding_same_state = []
        for from_position, to_position in valid_moves:
            new_state = self.state.move(from_position, to_position)
            if new_state is not None and not self.is_same_as_any_states(new_state):
                valid_moves_excluding_same_state.append((from_position, to_position))
        return valid_moves_excluding_same_state

    def enumerate_valid_from_positions_excluding_same_state(
        self,
    ) -> list[SolitireCardPositionFrom]:
        valid_positions = self.enumerate_valid_moves_excluding_same_state()
        return list(set(from_position for from_position, _ in valid_positions))

    def _deduplicate_first_same_states(self):
        for i, state in enumerate(self.states):
            for j, followed_state in enumerate(self.states[i + 1 :], start=i + 1):
                if state.is_same_state(followed_state):
                    cycle_offset = (
                        self.states[j].stock_cycle_count
                        - self.states[i].stock_cycle_count
                    )
                    for state in self.states[j:]:
                        state.stock_cycle_count -= cycle_offset
                    self.states = self.states[:i] + self.states[j:]
                    return True
        return False

    def deduplicate_same_states(self):
        while self._deduplicate_first_same_states():
            pass

    def undo(self) -> bool:
        if len(self.states) <= 1:
            return False
        self.states.pop()
        self.state = self.states[-1]
        self.moves.pop()
        return True

    def save_to_file(self, filename: str):
        with open(filename, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load_from_file(cls, filename: str) -> "SolitireGame":
        with open(filename, "rb") as f:
            game = pickle.load(f)
            if not isinstance(game, SolitireGame):
                raise ValueError(f"Expected SolitireGame, got {type(game)}")
            return game
