import pygame
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from widgets.pieces.Pawn import Pawn
from widgets.informations.BoardDisplay import BoardDisplay
from widgets.informations.TimerDisplay import TimerDisplay
from widgets.informations.KillLogDisplay import KillLogDisplay
from widgets.informations.GameOverDisplay import GameOverDisplay
from widgets.buttons.MenuButtonDisplay import MenuButtonDisplay
from features.decision.ReplayBuffer import ReplayBuffer
from styles.PieceImage import PieceImage
from styles.BackgroundImage import BackgroundImage
from pages.Menu import Menu
from pages.Board import Board
from features.decision.Decision import Decision

class Game:
    def __init__(self, play_with_computer=False, computer_vs_computer=False):
        pygame.init()
        pygame.mixer.init()
        self.screen = pygame.display.set_mode((1000, 1000))
        self.background = BackgroundImage.load_image()
        self.background = pygame.transform.scale(self.background, (640, 640))  
        self.piece_images = PieceImage.load_image()
        self.board_display = BoardDisplay(self.screen, self.background, self.piece_images)
        self.board = Board(self.screen, self.background, self.board_display)
        self.current_turn = 'white'
        self.selected_piece = None
        self.selected_position = None
        self.turn_time_limit = 60
        self.turn_start_time = pygame.time.get_ticks()
        self.play_with_computer = play_with_computer
        self.computer_vs_computer = computer_vs_computer
        self.kill_log = []
        self.game_over_display = GameOverDisplay(self.screen)
        pygame.display.set_caption("White turn")
        self.clock = pygame.time.Clock()

        self.menu_button = pygame.image.load("assets/images/Buttons/Menu.png").convert_alpha()
        self.menu_button = pygame.transform.scale(self.menu_button, (50, 50))

        self.timer_display = TimerDisplay(self.screen, self.turn_time_limit)
        self.kill_log_display = KillLogDisplay(self.screen, self.piece_images)
        self.menu_button_display = MenuButtonDisplay(self.screen, self.menu_button)

    def play(self):
        while True:
            self.board_display.display_board(self.board, self.selected_piece, self.selected_position, self.current_turn)
            self.timer_display.display_timer(self.turn_start_time, self.current_turn, self.board)
            self.kill_log_display.display_kill_log(self.kill_log)
            self.menu_button_display.display_menu_button()

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    self.handle_mousebuttondown()
                elif event.type == pygame.KEYDOWN:
                    self.handle_keydown(event)

            pygame.display.flip()
            self.clock.tick(30)

    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == pygame.MOUSEBUTTONDOWN:
                self.handle_mousebuttondown()

    def handle_mousebuttondown(self):
        mouse_x, mouse_y = pygame.mouse.get_pos()
        if 900 <= mouse_x <= 980 and 20 <= mouse_y <= 100:
            menu = Menu()
            menu.run()
        else:
            row, col = (mouse_y - 100) // 80, (mouse_x - 100) // 80
            if 0 <= row < 8 and 0 <= col < 8:
                self.handle_board_click(row, col)
                
    def handle_board_click(self, row, col):
        if self.selected_piece:
            start_pos = self.selected_position
            end_pos = (row, col)
            if start_pos == end_pos:
                self.selected_piece = None
                self.selected_position = None
            else:
                self.board.move_piece(start_pos, end_pos)
                
                pygame.mixer.Sound("assets/sounds/Move.wav").play()
                
                self.selected_piece = None
                self.selected_position = None
                self.switch_turn()
                self.turn_start_time = pygame.time.get_ticks()

                if self.is_game_over():
                    return

                if self.play_with_computer:
                    if self.current_turn == 'black':
                        self.computer_decision()
                elif self.computer_vs_computer:
                    while True:
                        self.computer_decision()
                        if self.is_game_over():
                            return
        else:
            piece = self.board.board[row, col]
            if piece:
                if piece.color == self.current_turn:
                    self.selected_piece = piece
                    self.selected_position = (row, col)
    
    def handle_keydown(self, event):
        if event.key == pygame.K_ESCAPE:
            menu = Menu()
            menu.run()

    def switch_turn(self):
        self.current_turn = 'black' if self.current_turn == 'white' else 'white'
        pygame.display.set_caption(f"{self.current_turn.capitalize()} turn")
        for row in self.board.board:
            for piece in row:
                if isinstance(piece, Pawn):
                    self.en_passant_target = False

    def is_game_over(self):
        if self.board.is_checkmate(self.current_turn):
            self.game_over_display.display_game_over(self.board, self.current_turn)
            return True
        elif self.board.is_stalemate(self.current_turn):
            self.game_over_display.display_game_over(self.board, self.current_turn)
            return True
        elif not self.board.king_exists(self.current_turn):
            self.game_over_display.display_game_over(self.board, self.current_turn)
            return True
        return False
    

    def computer_decision(self):
        decision = Decision(self.board, self.current_turn, self.kill_log)
        decision.computer_decision()
        self.switch_turn()
        self.turn_start_time = pygame.time.get_ticks()
        self.board_display.display_board(self.board, self.selected_piece, self.selected_position, self.current_turn)

