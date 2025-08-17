from dataclasses import dataclass
from typing import Optional, Tuple, Union

from solitier_game import PILES_COUNT, Card, CardState, SolitireState, Suit


@dataclass(frozen=True)
class TokenClose:
    pass


@dataclass(frozen=True)
class TokenCLS:
    pass


@dataclass(frozen=True)
class TokenEOS:
    pass


@dataclass(frozen=True)
class TokenClosedPile:
    pass


@dataclass(frozen=True)
class TokenOpenedPile:
    pass


@dataclass(frozen=True)
class TokenClosedStock:
    pass


@dataclass(frozen=True)
class TokenOpenedStock:
    pass


@dataclass(frozen=True)
class TokenWaste:
    pass


@dataclass(frozen=True)
class TokenOpen:
    """
    Represents an unknown token open event.
    """

    card: Card


@dataclass(frozen=True)
class TokenFoundation:
    """
    Represents a foundation token event with specific suit.
    """

    suit: Suit


Token = Union[
    TokenClose,
    TokenOpen,
    TokenFoundation,
    TokenCLS,
    TokenEOS,
    TokenOpenedPile,
    TokenClosedPile,
    TokenOpenedStock,
    TokenClosedStock,
    TokenWaste,
]

TOKENS_SEQ_LEN = 52 + 4 + 4 + PILES_COUNT * 2
TOKEN_INDEX_LEN = 52 + 4 + 8
SPECIAL_TOKEN_BEGIN = 52 + 1


def token_to_index(token: Token) -> int:
    match token:
        case TokenOpen(card):
            return card.number - 1 + 13 * card.suit.value
        case TokenClose():
            return 52
        # special tokens
        case TokenFoundation(suit):
            return suit.value + 52 + 1
        case TokenCLS():
            return 52 + 4 + 1
        case TokenEOS():
            return 52 + 4 + 2
        case TokenOpenedStock():
            return 52 + 4 + 3
        case TokenClosedStock():
            return 52 + 4 + 4
        case TokenWaste():
            return 52 + 4 + 5
        case TokenOpenedPile():
            return 52 + 4 + 6
        case TokenClosedPile():
            return 52 + 4 + 7
        case _:
            raise ValueError(f"Unknown token type: {token}")


def index_to_token(index: int) -> Token:
    if index < 52:
        suit = Suit(index // 13)
        number = index % 13 + 1
        return TokenOpen(Card(suit, number))
    elif index == 52:
        return TokenClose()
    elif index < 52 + 1 + 4:
        return TokenFoundation(Suit(index - 52 - 1))
    elif index == 52 + 4 + 1:
        return TokenCLS()
    elif index == 52 + 4 + 2:
        return TokenEOS()
    elif index == 52 + 4 + 3:
        return TokenOpenedStock()
    elif index == 52 + 4 + 4:
        return TokenClosedStock()
    elif index == 52 + 4 + 5:
        return TokenWaste()
    elif index == 52 + 4 + 6:
        return TokenOpenedPile()
    elif index == 52 + 4 + 7:
        return TokenClosedPile()
    else:
        raise ValueError(f"Unknown token index: {index}")


def is_special_token_index(index: int) -> bool:
    return index >= SPECIAL_TOKEN_BEGIN


def state_to_tokens(state: SolitireState) -> list[Token]:
    card_state_tokens = state_flatten(state)
    return [token for _, token in card_state_tokens]


def state_to_token_indices(state: SolitireState) -> list[int]:
    card_state_tokens = state_flatten(state)
    return [token_to_index(token) for _, token in card_state_tokens]


def state_flatten(state: SolitireState) -> list[Tuple[Optional[CardState], Token]]:
    """
    Flatten the state into a list of tokens, including open and closed cards.
    """
    tokens = []
    tokens.append((None, TokenCLS()))
    for suit in Suit:
        tokens.append((None, TokenFoundation(suit)))
        for card_state in state.foundation[suit]:
            tokens.append((card_state, TokenOpen(card_state.card)))
    if state.stock_cycle_count > 0:
        tokens.append((None, TokenOpenedStock()))
        for card_state in state.stock:
            tokens.append((card_state, TokenOpen(card_state.card)))
    else:
        tokens.append((None, TokenClosedStock()))
        for card_state in state.stock:
            tokens.append((card_state, TokenClose()))
    tokens.append((None, TokenWaste()))
    for card_state in state.waste_stock:
        tokens.append((card_state, TokenOpen(card_state.card)))
    for piles in state.piless:
        tokens.append((None, TokenClosedPile()))
        for card_state in piles:
            if not card_state.is_open:
                tokens.append((card_state, TokenClose()))
        tokens.append((None, TokenOpenedPile()))
        for card_state in piles:
            if card_state.is_open:
                tokens.append((card_state, TokenOpen(card_state.card)))
    tokens.append((None, TokenEOS()))

    return tokens


def flattened_card_state_tokens_and_tokens_to_map_card_states(
    card_state_tokens: list[Tuple[Optional[CardState], Token]],
    target_tokens: list[Token],
) -> dict[Token, Optional[CardState]]:
    """
    Map tokens to their corresponding card states in the flattened state.
    """
    map = {}
    for (card_state, token), target_token in zip(card_state_tokens, target_tokens):
        match card_state:
            case None:
                pass
            case CardState(card, _):
                match target_token:
                    case TokenOpen(card):
                        map[card_state.card] = CardState(card, is_open=True)
                    case TokenClose():
                        map[card_state.card] = CardState(card_state.card, is_open=False)
                    case _:
                        pass
            case _:
                pass
    return map
