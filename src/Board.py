import pygame
import sys
import numpy as np

from Pieces.King import King
from Pieces.Rook import Rook
from Pieces.Bishop import Bishop
from Pieces.Knight import Knight
from Pieces.Pawn import Pawn
from Pieces.Queen import Queen

class Board:
    def __init__(self, screen, background, board_display):
        self.board = self.initialize_board()
        self.en_passant_target = None
        self.en_passant_turn = None 
        self.screen = screen
        self.background = background
        self.board_display = board_display
        self.last_move_start = None
        self.last_move_end = None
        self.last_move_turn = None  
        self.computer_move_start = None
        self.computer_move_end = None
        
    def draw_computer_move(self, start_row, start_col, move):
        start_x, start_y = start_col * 80 + 140, start_row * 80 + 140
        end_x, end_y = move[1] * 80 + 140, move[0] * 80 + 140
        self.screen.fill((128, 128, 128))
        self.screen.blit(self.background, (100, 100))
        self.board_display.draw_board(self, [(255, 255, 255), (0, 0, 0)], pygame.font.SysFont(None, 24))  
        self.board_display.draw_dashed_line(self.screen, (0, 0, 255), (start_x, start_y), (end_x, end_y), 5)
        angle = np.arctan2(end_y - start_y, end_x - start_x)
        arrow_size = 10
        arrow_points = [
            (end_x, end_y),
            (end_x - arrow_size * np.cos(angle - np.pi / 6), end_y - arrow_size * np.sin(angle - np.pi / 6)),
            (end_x - arrow_size * np.cos(angle + np.pi / 6), end_y - arrow_size * np.sin(angle + np.pi / 6))
        ]
        pygame.draw.polygon(self.screen, (0, 0, 255), arrow_points)
        pygame.display.flip()

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
            if possible_moves: 
                if end_pos in possible_moves:
                    target_piece = self.board[end_pos[0], end_pos[1]]
                    if isinstance(target_piece, King):
                        print(f"{piece.color.capitalize()} wins")
                        pygame.quit()
                        sys.exit()

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
                    if self.last_move_turn is None or self.get_turn_count() - self.last_move_turn > 1:
                        return  
                    self.board[start_pos[0], end_pos[1]] = None

            original_piece = self.board[end_pos[0], end_pos[1]]
            self.board[end_pos[0], end_pos[1]] = piece
            self.board[start_pos[0], start_pos[1]] = None
            if self.is_in_check(piece.color):
                self.board[start_pos[0], start_pos[1]] = piece
                self.board[end_pos[0], end_pos[1]] = original_piece
                return
            self.board[start_pos[0], start_pos[1]] = piece
            self.board[end_pos[0], end_pos[1]] = original_piece

            self.board[end_pos[0], end_pos[1]] = piece
            self.board[start_pos[0], start_pos[1]] = None

            self.last_move_start = start_pos
            self.last_move_end = end_pos
            self.last_move_turn = self.get_turn_count()  

            if isinstance(piece, Pawn): 
                if abs(start_pos[0] - end_pos[0]) == 2:
                    self.en_passant_target = (start_pos[0] + (end_pos[0] - start_pos[0]) // 2, start_pos[1])
                    self.en_passant_turn = self.get_turn_count()  
                    piece.en_passant_target = True
                else:
                    piece.en_passant_target = None

            if isinstance(piece, Pawn): 
                if self.en_passant_target and end_pos == self.en_passant_target:
                    if self.get_turn_count() - self.en_passant_turn > 1:
                        return  
                    self.board[start_pos[0], end_pos[1]] = None

            piece.has_moved = True
                
            if isinstance(piece, Pawn):
                if end_pos[0] == 0 or end_pos[0] == 7:
                    piece.promote(self.board, end_pos)

    def get_turn(self):
        return 'white' if self.screen.get_at((0, 0)) == (255, 255, 255) else 'black'

    def get_turn_count(self):
        return pygame.time.get_ticks() // 1000  

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
        if self.is_in_check(color):
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
