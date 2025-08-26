import time
from dataclasses import replace
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pygame
from solitier_game import (
    Card,
    CardState,
    FromFoundation,
    FromPile,
    SolitireCardPositionFrom,
    SolitireCardPositionTo,
    SolitireGame,
    SolitireState,
    Suit,
    ToFoundation,
    ToPile,
    all_cards,
)
from solitier_game_lw import SolitireLightWeightGame, SolitireLightWeightState
from solitier_token import state_to_token_indices

suit_filenames = ["clubs", "diamonds", "hearts", "spades"]  # カードのスート名
number_filenames = [
    "ace",
    "2",
    "3",
    "4",
    "5",
    "6",
    "7",
    "8",
    "9",
    "10",
    "jack",
    "queen",
    "king",
]

background_color = (0, 130, 0)  # 緑色の背景


class CardSprite(pygame.sprite.Sprite):
    def __init__(
        self, back_image: pygame.Surface, image: pygame.Surface, card_state: CardState
    ):
        super().__init__()
        self.back_image = back_image
        self.open_image = image
        self.image = image
        self.rect = self.image.get_rect()
        self.card_state = card_state
        self.selected = False

    def draw(self, screen):
        if self.card_state.is_open:
            self.image = self.open_image
        else:
            self.image = self.back_image
        screen.blit(self.image, self.rect)
        if self.selected:
            # 枠を描画（太線・色を変えてもOK）
            pygame.draw.rect(
                screen, (255, 0, 0), self.rect, width=2  # 赤い枠  # 線の太さ
            )

    def set_position(self, x: int, y: int):
        self.rect = self.image.get_rect(topleft=(x, y))


class PlaceholderSprite(pygame.sprite.Sprite):
    def __init__(
        self,
        back_image: pygame.Surface,
        is_piles: bool = False,
        is_foundation: bool = False,
        is_stock: bool = False,
        pile_index: Optional[int] = None,
        foundation_suit: Optional[Suit] = None,
    ):
        super().__init__()
        self.back_image = back_image
        self.rect = self.back_image.get_rect()
        self.card_state = None
        self.selected = False
        self.is_piles = is_piles
        self.is_foundation = is_foundation
        self.is_stock = is_stock
        self.pile_index = pile_index
        self.foundation_suit = foundation_suit

    def draw(self, screen):
        if self.selected:
            # 枠を描画（太線・色を変えてもOK）
            pygame.draw.rect(
                screen, (255, 0, 0), self.rect, width=2  # 赤い枠  # 線の太さ
            )
        else:
            pygame.draw.rect(
                screen, (255, 255, 255), self.rect, width=2  # 赤い枠  # 線の太さ
            )

    def set_position(self, x, y):
        self.rect = self.back_image.get_rect(topleft=(x, y))


class CardSpriteLayeredUpdates(pygame.sprite.LayeredUpdates):
    def __init__(self, scale=0.6):
        back_image = pygame.image.load(
            "card_images/back.png"
        )  # Load your back image here
        original_card_width, original_card_height = back_image.get_size()
        self.card_width = int(original_card_width * scale)
        self.card_height = int(original_card_height * scale)

        back_image = pygame.transform.smoothscale(
            back_image, (self.card_width, self.card_height)
        )  # 新しいサイズ

        card_images = {}
        for row in range(len(suit_filenames)):
            for col in range(len(number_filenames)):
                filename = (
                    f"card_images/{number_filenames[col]}_of_{suit_filenames[row]}.png"
                )
                card_img = pygame.image.load(filename)
                card_img = pygame.transform.smoothscale(
                    card_img, (self.card_width, self.card_height)
                )  # 新しいサイズ
                card_images[(row, col)] = card_img

        self.all_sprites_updates = pygame.sprite.LayeredUpdates()
        self.card_splites = {}
        for card in all_cards():
            sprite = CardSprite(
                back_image,
                card_images[(card.suit.value, card.number - 1)],
                CardState(
                    card=card,
                    is_open=False,
                ),
            )
            self.all_sprites_updates.add(sprite, layer=card.number - 1)
            self.card_splites[card] = sprite

        self.foundation_placeholder_sprites = {}
        for suit in range(4):
            placeholder = PlaceholderSprite(
                back_image, is_foundation=True, foundation_suit=Suit(suit)
            )
            self.all_sprites_updates.add(placeholder, layer=0)
            self.foundation_placeholder_sprites[Suit(suit)] = placeholder

        self.piles_placeholder_sprites = {}
        for i in range(7):
            placeholder = PlaceholderSprite(back_image, is_piles=True, pile_index=i)
            self.all_sprites_updates.add(placeholder, layer=0)
            self.piles_placeholder_sprites[i] = placeholder

        self.stock_placeholder_sprite = PlaceholderSprite(back_image, is_stock=True)
        self.all_sprites_updates.add(self.stock_placeholder_sprite, layer=0)


def test_show_all_cards():
    screen = pygame.display.set_mode((1000, 800))
    pygame.display.set_caption("Solitire Cards")

    clock = pygame.time.Clock()
    running = True

    sprites_updates = CardSpriteLayeredUpdates()

    while running:
        events = pygame.event.get()
        for event in events:
            if event.type == pygame.QUIT:
                running = False

        screen.fill(background_color)
        for sprite in sprites_updates.card_splites.values():
            x = sprite.card_state.card.number * sprites_updates.card_width / 2
            y = sprite.card_state.card.suit.value * sprites_updates.card_height
            sprite.set_position(x, y)
            sprite.card_state = replace(
                sprite.card_state, is_open=True
            )  # 全てのカードを開く
            sprites_updates.all_sprites_updates.change_layer(
                sprite, sprite.card_state.card.number - 1
            )

        for event in events:
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                print(event.pos)
                for sprite in reversed(
                    sprites_updates.all_sprites_updates.sprites()
                ):  # 上の順で当たり判定
                    if sprite.rect.collidepoint(event.pos):
                        sprite.selected = not sprite.selected
                        if isinstance(sprite, CardSprite):
                            print(
                                f"Selected: {sprite.card_state.card} - Open: {sprite.card_state.is_open}"
                            )
                        elif isinstance(sprite, PlaceholderSprite):
                            info = (
                                sprite.foundation_suit
                                if sprite.is_foundation
                                else (sprite.pile_index if sprite.is_piles else "stock")
                            )
                            print(f"Selected Placeholder: " f"{info}")
                        break

        for sprite in sprites_updates.all_sprites_updates:
            sprite.draw(screen)
        pygame.display.flip()  # Update the display

        clock.tick(60)  # Limit to 60 FPS

    pygame.quit()


class SolitiereStateVisualizer:
    def __init__(self):
        self.margin = 10  # マージンを設定
        self.card_shift = 30  # カードの重なりを設定

        self.spritse_updates = CardSpriteLayeredUpdates()
        self.mapped_spritse_updates = None
        self.width = 1200
        self.height = 800
        self.stock_view_count = 7

    def visualize(self, state: SolitireState, top=0, left=0):
        for suit, card_states in state.foundation.items():
            y = (
                top
                + suit.value * (self.spritse_updates.card_height + self.margin)
                + self.margin
            )
            x = left + self.margin
            placeholder_sprite = self.spritse_updates.foundation_placeholder_sprites[
                suit
            ]
            placeholder_sprite.set_position(x, y)
            self.spritse_updates.all_sprites_updates.change_layer(placeholder_sprite, 0)
            for card_state in card_states:
                sprite = self.spritse_updates.card_splites[card_state.card]
                sprite.set_position(x, y)
                sprite.card_state = replace(
                    sprite.card_state, is_open=card_state.is_open
                )
                self.spritse_updates.all_sprites_updates.change_layer(
                    sprite, card_state.card.number - 1
                )

        for col, card_states in enumerate(state.piless):
            x = (
                left
                + (col + 1) * (self.spritse_updates.card_width + self.margin)
                + self.margin
            )
            y = top + self.margin
            placeholder_sprite = self.spritse_updates.piles_placeholder_sprites[col]
            placeholder_sprite.set_position(x, y)
            self.spritse_updates.all_sprites_updates.change_layer(placeholder_sprite, 0)
            # 各列のカードを配置
            for row, card_state in enumerate(card_states):
                y = top + row * self.card_shift + self.margin
                sprite = self.spritse_updates.card_splites[card_state.card]
                sprite.set_position(x, y)
                sprite.card_state = replace(
                    sprite.card_state, is_open=card_state.is_open
                )
                self.spritse_updates.all_sprites_updates.change_layer(sprite, row)

        x = (
            left
            + (len(state.piless) + 1) * (self.spritse_updates.card_width + self.margin)
            + self.margin
        )
        y = (
            top
            + self.spritse_updates.card_height
            + self.margin
            + (self.stock_view_count - 1) * self.card_shift
        )
        self.spritse_updates.stock_placeholder_sprite.set_position(
            x, y + (self.stock_view_count - 1) * self.card_shift
        )
        self.spritse_updates.all_sprites_updates.change_layer(
            self.spritse_updates.stock_placeholder_sprite, 0
        )
        for index, card_state in enumerate(reversed(state.stock)):
            sprite = self.spritse_updates.card_splites[card_state.card]
            display_index = max(0, index - len(state.stock) + self.stock_view_count)
            sprite.set_position(x, y + display_index * self.card_shift)
            if state.stock_cycle_count > 0:
                sprite.card_state = replace(sprite.card_state, is_open=True)
            else:
                sprite.card_state = replace(sprite.card_state, is_open=False)
            self.spritse_updates.all_sprites_updates.change_layer(sprite, index)

        for index, card_state in enumerate(state.waste_stock):
            sprite = self.spritse_updates.card_splites[card_state.card]
            display_index = max(
                0, index - len(state.waste_stock) + self.stock_view_count
            )
            sprite.set_position(x, display_index * self.card_shift)
            sprite.card_state = replace(sprite.card_state, is_open=card_state.is_open)
            self.spritse_updates.all_sprites_updates.change_layer(sprite, index)

    def lookup_sprite_from_position(
        self, state: SolitireState, from_position: SolitireCardPositionFrom
    ):
        card_state = state.get_card_state_at_from_position(from_position)
        match from_position:
            case "FromStock":
                if card_state is not None:
                    return self.spritse_updates.card_splites[card_state.card]
                else:
                    return self.spritse_updates.stock_placeholder_sprite
            case "FromWaste":
                if card_state is not None:
                    return self.spritse_updates.card_splites[card_state.card]
                else:
                    return None
            case FromFoundation(suit):
                if card_state is not None:
                    return self.spritse_updates.card_splites[card_state.card]
                else:
                    return self.spritse_updates.foundation_placeholder_sprites[suit]
            case FromPile(col, _):
                if card_state is not None:
                    return self.spritse_updates.card_splites[card_state.card]
                else:
                    return self.spritse_updates.piles_placeholder_sprites[col]
        return None

    def lookup_sprite_to_position(
        self,
        state: SolitireState,
        to_position: SolitireCardPositionTo,
    ):
        card_state = state.get_card_state_at_to_position(to_position)
        match to_position:
            case "ToStock":
                if card_state is not None:
                    return self.spritse_updates.card_splites[card_state.card]
                else:
                    return self.spritse_updates.stock_placeholder_sprite
            case "ToWaste":
                if card_state is not None:
                    return self.spritse_updates.card_splites[card_state.card]
                else:
                    return None
            case ToFoundation(suit):
                if card_state is not None:
                    return self.spritse_updates.card_splites[card_state.card]
                else:
                    return self.spritse_updates.foundation_placeholder_sprites[suit]
            case ToPile(col):
                if card_state is not None:
                    return self.spritse_updates.card_splites[card_state.card]
                else:
                    return self.spritse_updates.piles_placeholder_sprites[col]
        return None

    def unselect_all(self):
        for sprite in self.spritse_updates.all_sprites_updates:
            sprite.selected = False

    def select_from_position(
        self, state: SolitireState, from_position: SolitireCardPositionFrom
    ):
        sprite = self.lookup_sprite_from_position(state, from_position)
        if sprite is not None:
            sprite.selected = True

    def select_to_position(
        self, state: SolitireState, to_position: SolitireCardPositionTo
    ):
        sprite = self.lookup_sprite_to_position(state, to_position)
        if sprite is not None:
            sprite.selected = True

    def find_sprite_at_pos(self, pos) -> Optional[Union[CardSprite, PlaceholderSprite]]:
        for sprite in reversed(self.spritse_updates.all_sprites_updates.sprites()):
            if sprite.rect.collidepoint(pos):
                return sprite
        return None

    def lookup_from_position_at_sprite(
        self, state: SolitireState, sprite: Union[CardSprite, PlaceholderSprite]
    ) -> Optional[SolitireCardPositionFrom]:
        if isinstance(sprite, CardSprite):
            return state.resolve_card_from(sprite.card_state.card)
        elif isinstance(sprite, PlaceholderSprite):
            if sprite.is_stock:
                return "FromStock"
        return None

    def lookup_to_position_at_sprite(
        self, state: SolitireState, sprite: Union[CardSprite, PlaceholderSprite]
    ) -> Optional[SolitireCardPositionTo]:
        if isinstance(sprite, CardSprite):
            return state.resolve_card_to(sprite.card_state.card)
        elif isinstance(sprite, PlaceholderSprite):
            if sprite.is_foundation and sprite.foundation_suit is not None:
                return ToFoundation(sprite.foundation_suit)
            elif sprite.is_piles and sprite.pile_index is not None:
                return ToPile(sprite.pile_index)
            elif sprite.is_stock:
                return "ToStock"
        return None

    def draw_sprite(self, screen: pygame.Surface):
        for sprite in self.spritse_updates.all_sprites_updates:
            sprite.draw(screen)

    def draw_sprite_map_card(
        self, screen: pygame.Surface, card_to_card_state_map: Dict[Card, CardState]
    ):
        if self.mapped_spritse_updates is None:
            self.mapped_spritse_updates = CardSpriteLayeredUpdates()
        for sprite in self.spritse_updates.all_sprites_updates:
            match sprite:
                case CardSprite():
                    mapped_card_state = card_to_card_state_map.get(
                        sprite.card_state.card
                    )
                    if mapped_card_state is not None:
                        mapped_sprite = self.mapped_spritse_updates.card_splites[
                            mapped_card_state.card
                        ]
                        mapped_sprite.set_position(*sprite.rect.topleft)
                        mapped_sprite.card_state = mapped_card_state
                        mapped_sprite.selected = sprite.selected
                        mapped_sprite.draw(screen)
                    else:
                        sprite.draw(screen)
                case PlaceholderSprite():
                    sprite.draw(screen)

    def select_cards(self, cards: List[Card]):
        """
        Selects the cards in the visualizer.
        :param cards: List of Card objects to select.
        """
        for card in cards:
            if card in self.spritse_updates.card_splites:
                self.spritse_updates.card_splites[card].selected = True

    def visualize_state_to_numpy_image(
        self,
        state: SolitireState,
        card_to_card_state_map: Optional[Dict[Card, CardState]] = None,
    ) -> np.ndarray:
        pygame.init()
        self.visualize(state)
        surface = pygame.Surface((self.width, self.height))
        surface.fill(background_color)
        if card_to_card_state_map is None:
            self.draw_sprite(surface)
        else:
            self.draw_sprite_map_card(surface, card_to_card_state_map)
        # numpy配列に変換（x, y, color）
        arr = pygame.surfarray.array3d(surface)

        # 軸を (height, width, color) に入れ替え
        arr = np.transpose(arr, (1, 0, 2))
        pygame.quit()
        return arr


class SolitireGameVisualizer:
    def __init__(self, game: Optional[SolitireGame] = None):
        if game is None:
            game = SolitireGame()
        self.game = game
        self.state_visualizer = SolitiereStateVisualizer()

    def visualize(self, top=0, left=0):
        self.state_visualizer.visualize(self.game.state, top, left)

    def lookup_sprite_from_position(self, from_position: SolitireCardPositionFrom):
        return self.state_visualizer.lookup_sprite_from_position(
            self.game.state, from_position
        )

    def lookup_sprite_to_position(self, to_position: SolitireCardPositionTo):
        return self.state_visualizer.lookup_sprite_to_position(
            self.game.state, to_position
        )

    def unselect_all(self):
        self.state_visualizer.unselect_all()

    def select_from_position(self, from_position: SolitireCardPositionFrom):
        self.state_visualizer.select_from_position(self.game.state, from_position)

    def select_to_position(self, to_position: SolitireCardPositionTo):
        self.state_visualizer.select_to_position(self.game.state, to_position)

    def find_sprite_at_pos(self, pos) -> Optional[Union[CardSprite, PlaceholderSprite]]:
        return self.state_visualizer.find_sprite_at_pos(pos)

    def lookup_from_position_at_sprite(
        self, sprite: Union[CardSprite, PlaceholderSprite]
    ) -> Optional[SolitireCardPositionFrom]:
        return self.state_visualizer.lookup_from_position_at_sprite(
            self.game.state, sprite
        )

    def lookup_to_position_at_sprite(
        self, sprite: Union[CardSprite, PlaceholderSprite]
    ) -> Optional[SolitireCardPositionTo]:
        return self.state_visualizer.lookup_to_position_at_sprite(
            self.game.state, sprite
        )

    def draw_sprite(self, screen: pygame.Surface):
        self.state_visualizer.draw_sprite(screen)

    def render_game_to_movie(self, filename: str = "game.mp4", fps: int = 30):
        import cv2

        # pygameは上でimport済みなら不要
        W, H, FPS = self.state_visualizer.width, self.state_visualizer.height, fps

        # 偶数解像度だと安心（必要なら丸め）
        W -= W % 2
        H -= H % 2

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # 環境により 'avc1' でも可
        writer = cv2.VideoWriter(filename, fourcc, FPS, (W, H))
        try:
            # 画面は開かずオフスクリーンで作成
            surface = pygame.Surface((W, H))

            for state in getattr(self.game, "states", [self.game.state]):
                # 各ステートを描画
                self.state_visualizer.visualize(state)
                surface.fill(background_color)
                self.draw_sprite(surface)

                # pygame(RGB) -> OpenCV(BGR)
                frame = pygame.surfarray.array3d(surface).transpose(1, 0, 2)
                frame = frame[:, :, ::-1]  # RGB->BGR（cv2.cvtColorでもOK）
                writer.write(frame.astype("uint8"))
        finally:
            writer.release()

    async def run_by_move_estimator(
        self,
        move_estimator: Callable[
            [SolitireGame],
            Optional[Tuple[SolitireCardPositionFrom, SolitireCardPositionTo]],
        ],
        max_moves: int = 200,
        is_loop: bool = False,
    ):
        """
        Run the game in auto-play mode.
        """
        pygame.init()
        screen = pygame.display.set_mode(
            (self.state_visualizer.width, self.state_visualizer.height)
        )
        pygame.display.set_caption("Solitire Game Visualizer")

        clock = pygame.time.Clock()
        running = True

        while True:
            move_count = 0
            while running:
                events = pygame.event.get()
                for event in events:
                    if event.type == pygame.QUIT:
                        running = False
                        is_loop = False

                if self.game.state.is_all_open():
                    print("All cards are open!")
                    # ここで全てのカードが開いている場合の処理を追加できます
                    # 例えば、ゲームを終了する、メッセージを表示するなど
                    running = False
                    break

                self.visualize()
                self.unselect_all()
                screen.fill(background_color)
                self.draw_sprite(screen)
                pygame.display.flip()

                clock.tick(60)  # Limit to 60 FPS
                move = await move_estimator(self.game)
                if move is None:
                    print("No valid moves available or game completed.")
                    running = False
                    break
                else:
                    self.game.checked_move(*move)
                    self.unselect_all()
                    self.select_from_position(move[0])
                    self.select_to_position(move[1])
                    move_count += 1
                    if move_count >= max_moves:
                        print(f"Reached maximum moves: {max_moves}. Stopping game.")
                        running = False
                        break

                self.visualize()
                screen.fill(background_color)
                self.draw_sprite(screen)
                pygame.display.flip()
                clock.tick(60)  # Limit to 60 FPS

            if is_loop:
                self.game = SolitireGame()  # Reset the game
                running = True
                move_count = 0
            else:
                break
        pygame.quit()

    def run(self, is_allow_same_state: bool = False):
        pygame.init()
        screen = pygame.display.set_mode(
            (self.state_visualizer.width, self.state_visualizer.height)
        )
        pygame.display.set_caption("Solitire Visualizer")

        clock = pygame.time.Clock()
        running = True
        from_position = None
        last_click_time = 0
        DOUBLE_CLICK_TIME = 500  # ms
        double_clicked = True

        while running:
            events = pygame.event.get()
            for event in events:
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    now = pygame.time.get_ticks()
                    if now - last_click_time <= DOUBLE_CLICK_TIME:
                        double_clicked = True
                    else:
                        double_clicked = False
                        print(event.pos)
                        self.state_visualizer.unselect_all()
                        from_position = None

                        sprite_at_pos = self.find_sprite_at_pos(event.pos)
                        if sprite_at_pos is not None:
                            sprite_at_pos.selected = True
                            from_position = self.lookup_from_position_at_sprite(
                                sprite_at_pos
                            )
                            print(f"From Position: {from_position}")
                            if from_position is not None:
                                if is_allow_same_state:
                                    valid_to_positions = (
                                        self.game.state.enumerate_valid_moves_from(
                                            from_position
                                        )
                                    )
                                else:
                                    valid_to_positions = self.game.enumerate_valid_moves_from_excluding_same_state(
                                        from_position
                                    )
                                print(f"Valid To Positions: {valid_to_positions}")
                                for to_position in valid_to_positions:
                                    self.select_to_position(to_position)
                            if isinstance(sprite_at_pos, CardSprite):
                                print(
                                    f"Selected: {sprite_at_pos.card_state.card} - Open: {sprite_at_pos.card_state.is_open}"
                                )
                            elif isinstance(sprite_at_pos, PlaceholderSprite):
                                info = (
                                    sprite_at_pos.foundation_suit
                                    if sprite_at_pos.is_foundation
                                    else (
                                        sprite_at_pos.pile_index
                                        if sprite_at_pos.is_piles
                                        else "stock"
                                    )
                                )
                                print(f"Selected Placeholder: {info}")
                    last_click_time = now
                elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
                    self.state_visualizer.unselect_all()
                    sprite_at_pos = self.find_sprite_at_pos(event.pos)
                    if double_clicked:
                        if from_position is not None:
                            if is_allow_same_state:
                                valid_to_positions = (
                                    self.game.state.enumerate_valid_moves_from(
                                        from_position
                                    )
                                )
                                if len(valid_to_positions) > 0:
                                    self.game.checked_move(
                                        from_position, valid_to_positions[0]
                                    )
                            else:
                                valid_to_positions = self.game.enumerate_valid_moves_from_excluding_same_state(
                                    from_position
                                )
                                if len(valid_to_positions) > 0:
                                    self.game.checked_move_excluding_same_state(
                                        from_position, valid_to_positions[0]
                                    )
                    else:
                        if sprite_at_pos is not None:
                            to_position = self.lookup_to_position_at_sprite(
                                sprite_at_pos
                            )
                            if to_position is not None and from_position is not None:
                                if is_allow_same_state:
                                    self.game.checked_move(from_position, to_position)
                                else:
                                    search_state = SolitireLightWeightState(
                                        self.game.state
                                    )
                                    self.game.checked_move_excluding_same_state(
                                        from_position, to_position
                                    )
                                    if self.game.state.is_valid_move(
                                        from_position, to_position
                                    ):
                                        new_states = search_state.move_uncertain_states(
                                            from_position, to_position
                                        )
                                        assert new_states[0].verify()

                    if self.game.state.is_all_open():
                        print("All cards are open!")
                        # ここで全てのカードが開いている場合の処理を追加できます
                        # 例えば、ゲームを終了する、メッセージを表示するなど
                        running = False

                    if is_allow_same_state:
                        valid_from_positions = (
                            self.game.state.enumerate_valid_from_positions()
                        )
                    else:
                        valid_from_positions = (
                            self.game.enumerate_valid_from_positions_excluding_same_state()
                        )
                    print(f"Valid From Positions: {valid_from_positions}")
                    for valid_from_position in valid_from_positions:
                        self.select_from_position(valid_from_position)
                    # for debug
                    self.visualize()
                    screen.fill(background_color)
                    self.draw_sprite(screen)
                    pygame.display.flip()  # Update the display
                    st_time = time.time_ns()
                    valid_moves = self.game.enumerate_valid_moves_excluding_same_state()
                    uncertain_states = {
                        from_to_position: self.game.state.move_uncertain_states(
                            *from_to_position
                        )
                        for from_to_position in valid_moves
                    }
                    ed_time = time.time_ns()
                    print(
                        f"Uncertain States Time(Game): {(ed_time - st_time) * 1e-9:.2f} seconds"
                    )
                    uncertain_states_count = {
                        from_to_position: len(states)
                        for from_to_position, states in uncertain_states.items()
                    }
                    sum_uncertain_states_count = sum(uncertain_states_count.values())
                    print(f"Uncertain States: {uncertain_states_count}")
                    print(
                        f"Sum of Uncertain States(Game): {sum_uncertain_states_count}"
                    )

                    # verify SolitireLightWeightGame
                    search_game = SolitireLightWeightGame.from_solitire_game(self.game)
                    st_time = time.time_ns()
                    valid_moves_verify = (
                        search_game.enumerate_valid_moves_excluding_same_state(
                            is_compatibility=True
                        )
                    )
                    uncertain_states_verify = {}
                    for valid_move in valid_moves_verify:
                        uncertain_states_verify[valid_move] = (
                            search_game.move_uncertain_states(*valid_move)
                        )
                    ed_time = time.time_ns()
                    print(
                        f"Uncertain States Time(LWGame): {(ed_time - st_time) * 1e-9:.6f} seconds"
                    )
                    uncertain_states_count_verify = {
                        from_to_position: len(states)
                        for from_to_position, states in uncertain_states_verify.items()
                    }
                    sum_uncertain_states_count_verify = sum(
                        uncertain_states_count_verify.values()
                    )
                    print(f"Uncertain States: {uncertain_states_count_verify}")
                    print(
                        f"Sum of Uncertain States(LWGame): {sum_uncertain_states_count_verify}"
                    )

                    print(f"Valid Moves: {set(valid_moves)}")
                    print(f"Valid Moves: {set(valid_moves_verify)}")
                    assert set(valid_moves) == set(valid_moves_verify)
                    assert (
                        sum_uncertain_states_count == sum_uncertain_states_count_verify
                    )
                    for move, states_verify in uncertain_states_verify.items():
                        states = uncertain_states.get(move)
                        assert states is not None
                        assert len(states) == len(states_verify), (
                            f"Move: {move}, "
                            f"{len(states)} != {len(states_verify)}, "
                            f"{states} != {states_verify}"
                        )
                        assert (
                            states[0].open_count() == states_verify[0].open_count()
                        ), f"{move}: {states[0].open_count()} != {states_verify[0].open_count()}"
                        for state in states_verify:
                            assert state.get_last_state().verify(), f"{move}"
                        token_indicess = [
                            state_to_token_indices(state) for state in states
                        ]
                        token_hash_set = set([hash(tuple(t)) for t in token_indicess])
                        token_indicess_verify = [
                            state.get_last_state().to_token_indices()
                            for state in states_verify
                        ]
                        token_hash_set_verify = set(
                            [hash(tuple(t)) for t in token_indicess_verify]
                        )
                        assert token_hash_set == token_hash_set_verify, f"{move}"

                    # verify SolitireLightWeightState
                    search_state = SolitireLightWeightState(self.game.state)
                    st_time = time.time_ns()
                    valid_moves_verify = search_state.enumerate_valid_moves()
                    uncertain_states_verify = {}
                    for valid_move in valid_moves_verify:
                        uncertain_states_verify[valid_move] = (
                            search_state.move_uncertain_states(*valid_move)
                        )
                    ed_time = time.time_ns()
                    print(
                        f"Uncertain States Time(LWState): {(ed_time - st_time) * 1e-9:.6f} seconds"
                    )
                    valid_moves = self.game.state.enumerate_valid_moves()
                    print(f"Valid Moves: {set(valid_moves)}")
                    print(f"Valid Moves: {set(valid_moves_verify)}")
                    assert set(valid_moves) == set(valid_moves_verify)
                    tokens_ref = state_to_token_indices(self.game.state)
                    tokens_verify = search_state.to_token_indices()
                    assert tokens_ref == list(
                        tokens_verify
                    ), f"{tokens_ref} != {tokens_verify}"
                    clock.tick(60)  # Limit to 60 FPS
                    continue

            self.visualize()
            screen.fill(background_color)
            self.draw_sprite(screen)
            pygame.display.flip()  # Update the display

            clock.tick(60)  # Limit to 60 FPS
        pygame.quit()
