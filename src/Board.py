import pygame
import numpy as np

from widgets.pieces.King import King
from widgets.pieces.Rook import Rook
from widgets.pieces.Bishop import Bishop
from widgets.pieces.Knight import Knight
from widgets.pieces.Pawn import Pawn
from widgets.pieces.Queen import Queen
from widgets.information.GameOverDisplay import GameOverDisplay

class Board:
    def __init__(self, screen, background, board_display):
        self.board = self.initialize_board()
        self.en_passant_target = None
        self.en_passant_turn = None 
        self.screen = screen
        self.background = background
        self.board_display = board_display
        self.game_over_display = GameOverDisplay(screen)  
        self.last_move_start = None
        self.last_move_end = None
        self.last_move_turn = None  
        self.computer_move_start = None
        self.computer_move_end = None
        
    def draw_computer_move(self, start_row, start_col, move):
        pass

    def initialize_board(self):
        board = np.full((8, 8), None)
        for i in range(8):
            board[1, i] = Pawn('black')
            board[6, i] = Pawn('white')
        board[0, [0, 7]] = Rook('black')
        board[0, [1, 6]] = Knight('black')
        board[0, [2, 5]] = Bishop('black')
        board[0, 3] = Queen('black')
        board[0, 4] = King('black')
        board[7, [0, 7]] = Rook('white')
        board[7, [1, 6]] = Knight('white')
        board[7, [2, 5]] = Bishop('white')
        board[7, 3] = Queen('white')
        board[7, 4] = King('white')
        return board

    def move_piece(self, start_pos, end_pos):
        piece = self.board[start_pos[0], start_pos[1]]
        if piece:
            possible_moves = piece.get_possible_moves(self.board, start_pos)
            if end_pos in possible_moves:
                target_piece = self.board[end_pos[0], end_pos[1]]
                if isinstance(target_piece, King):
                    self.game_over_display.display_game_over(self, piece.color)
                    return

                if isinstance(piece, King):
                    if abs(start_pos[1] - end_pos[1]) == 2:
                        if end_pos[1] == 6: 
                            self.board[start_pos[0], 5] = self.board[start_pos[0], 7]
                            self.board[start_pos[0], 7] = None
                        elif end_pos[1] == 2: 
                            self.board[start_pos[0], 3] = self.board[start_pos[0], 0]
                            self.board[start_pos[0], 0] = None

                if isinstance(piece, Pawn):
                    if end_pos == self.en_passant_target:
                        self.board[start_pos[0], end_pos[1]] = None

                original_piece = self.board[end_pos[0], end_pos[1]]
                self.board[end_pos[0], end_pos[1]] = piece
                self.board[start_pos[0], start_pos[1]] = None

                if self.is_in_check(piece.color):
                    self.board[start_pos[0], start_pos[1]] = piece
                    self.board[end_pos[0], end_pos[1]] = original_piece
                    return

                self.last_move_start = start_pos
                self.last_move_end = end_pos
                self.last_move_turn = self.get_turn_count()

                if isinstance(piece, Pawn):
                    if abs(start_pos[0] - end_pos[0]) == 2:
                        self.en_passant_target = (start_pos[0] + (end_pos[0] - start_pos[0]) // 2, start_pos[1])
                        self.en_passant_turn = self.get_turn_count()
                    else:
                        self.en_passant_target = None

                piece.has_moved = True

                if isinstance(piece, Pawn) and (end_pos[0] == 0 or end_pos[0] == 7):
                    piece.promote(self.board, end_pos)

    def get_turn(self):
        return 'white' if self.screen.get_at((0, 0)) == (255, 255, 255) else 'black'

    def get_turn_count(self):
        return pygame.time.get_ticks() // 1000  
    
    def king_exists(self, color):
        for row in self.board:
            for piece in row:
                if isinstance(piece, King) and piece.color == color:
                    return True
        return False

    def is_in_check(self, color):
        king_position = None
        for i in range(8):
            for j in range(8):
                piece = self.board[i, j]
                if piece:
                    if isinstance(piece, King):
                        if piece.color == color:
                            king_position = (i, j)
                            break
            if king_position:
                break

        for i in range(8):
            for j in range(8):
                piece = self.board[i, j]
                if piece:
                    if piece.color != color:
                        possible_moves = piece.get_possible_moves(self.board, (i, j))
                        if possible_moves:
                            if king_position in possible_moves:
                                return True
        return False

    def is_checkmate(self, color):
        if not self.is_in_check(color):
            return False

        for i in range(8):
            for j in range(8):
                piece = self.board[i, j]
                if piece: 
                    if piece.color == color:
                        possible_moves = piece.get_possible_moves(self.board, (i, j))
                        if possible_moves:
                            for move in possible_moves:
                                original_piece = self.board[move[0], move[1]]
                                self.board[move[0], move[1]] = piece
                                self.board[i, j] = None
                                if not self.is_in_check(color):
                                    self.board[i, j] = piece
                                    self.board[move[0], move[1]] = original_piece
                                    return False
                                self.board[i, j] = piece
                                self.board[move[0], move[1]] = original_piece
        return True

    def is_stalemate(self, color):
        # 현재 플레이어가 체크 상태
        if self.is_in_check(color):
            return False
        
        for i in range(8):
            for j in range(8):
                piece = self.board[i, j]
                # 현재 칸에 있는 말이 현재 플레이어의 말인지 확인
                if piece and piece.color == color:
                    # 현재 말의 가능한 이동 경로를 가져와서
                    possible_moves = piece.get_possible_moves(self.board, (i, j))
                    for move in possible_moves:
                        # 이동할 위치에 있는 말을 임시로 저장
                        original_piece = self.board[move[0], move[1]]
                        # 현재 말을 이동할 위치로 이동
                        self.board[move[0], move[1]] = piece
                        self.board[i, j] = None
                        # 이동 후 체크 상태가 아닌지 확인
                        if not self.is_in_check(color):
                            # 체크 상태가 아니라면 원래 위치로
                            self.board[i, j] = piece
                            self.board[move[0], move[1]] = original_piece
                            return False
                        # 체크 상태라면 원래 위치로 회귀
                        self.board[i, j] = piece
                        self.board[move[0], move[1]] = original_piece
        # 모든 가능한 이동 후에도 체크 상태가 아니라면 stalemate
        return True