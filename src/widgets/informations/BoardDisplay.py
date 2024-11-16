import pygame
import numpy as np

from widgets.pieces.King import King

class BoardDisplay:
    def __init__(self, screen, background, piece_images):
        self.screen = screen
        self.background = background
        self.piece_images = piece_images
        
    def draw_computer_move(self, start_row, start_col, move):
        start_x, start_y = start_col * 80 + 140, start_row * 80 + 140
        end_x, end_y = move[1] * 80 + 140, move[0] * 80 + 140
        self.draw_dashed_line(self.screen, (0, 0, 255), (start_x, start_y), (end_x, end_y), 5)
        angle = np.arctan2(end_y - start_y, end_x - start_x)
        arrow_size = 10
        arrow_points = [
            (end_x, end_y),
            (end_x - arrow_size * np.cos(angle - np.pi / 6), end_y - arrow_size * np.sin(angle - np.pi / 6)),
            (end_x - arrow_size * np.cos(angle + np.pi / 6), end_y - arrow_size * np.sin(angle + np.pi / 6))
        ]
        pygame.draw.polygon(self.screen, (0, 0, 255), arrow_points)
        pygame.display.flip()

    def display_board(self, board, selected_piece, selected_position, current_turn):
        self.screen.fill((128, 128, 128))
        self.screen.blit(self.background, (100, 100))  
        colors = [(255, 255, 255), (0, 0, 0)]
        font = pygame.font.SysFont(None, 24)
        mouse_position = pygame.mouse.get_pos()
        mouse_row, mouse_col = (mouse_position[1] - 100) // 80, (mouse_position[0] - 100) // 80

        self.draw_board(board, colors, font)
        self.highlight_mouse_position(mouse_row, mouse_col)
        self.highlight_selected_piece(board, selected_piece, selected_position)
        self.highlight_check(board, current_turn)
        self.highlight_last_move(board)

    def draw_board(self, board, colors, font):
        for row in range(8):
            for col in range(8):
                color = colors[(row + col) % 2]
                pygame.draw.rect(self.screen, color, pygame.Rect(col * 80 + 100, row * 80 + 100, 80, 80), 1)
                piece = board.board[row, col]
                if piece:
                    piece_image = self.piece_images[type(piece).__name__ + '_' + piece.color[0]]
                    self.screen.blit(piece_image, (col * 80 + 100, row * 80 + 100))
                
                coord_text = font.render(f"{chr(97 + col)}{8 - row}", True, (0, 0, 0, 128) if color == (255, 255, 255) else (255, 255, 255, 128))
                self.screen.blit(coord_text, (col * 80 + 105, row * 80 + 105))

    def highlight_mouse_position(self, mouse_row, mouse_col):
        if (0 <= mouse_row < 8) and (0 <= mouse_col < 8):
            highlight_surface = pygame.Surface((80, 80), pygame.SRCALPHA)
            highlight_surface.fill((255, 255, 255, 128))
            self.screen.blit(highlight_surface, (mouse_col * 80 + 100, mouse_row * 80 + 100))

    def highlight_selected_piece(self, board, selected_piece, selected_position):
        if selected_piece:
            if (pygame.time.get_ticks() // 200) % 2 == 0:
                pygame.draw.rect(self.screen, (0, 255, 0), pygame.Rect(selected_position[1] * 80 + 100, selected_position[0] * 80 + 100, 80, 80), 3)
            possible_moves = selected_piece.get_possible_moves(board.board, selected_position)
            if possible_moves:
                for move in possible_moves:
                    if board.board[move[0], move[1]] is None:
                        pygame.draw.circle(self.screen, (0, 255, 0), (move[1] * 80 + 140, move[0] * 80 + 140), 7)  
                    else:
                        pygame.draw.circle(self.screen, (255, 0, 0), (move[1] * 80 + 140, move[0] * 80 + 140), 7)  

    def highlight_check(self, board, current_turn):
        if board.is_in_check(current_turn):
            king_position = self.find_king_position(board, current_turn)
            for row in range(8):
                for col in range(8):
                    piece = board.board[row, col]
                    if piece and(piece.color != current_turn):
                        possible_moves = piece.get_possible_moves(board.board, (row, col))
                        if possible_moves:
                            if king_position in possible_moves:
                                pygame.draw.rect(self.screen, (255, 0, 0), pygame.Rect(col * 80 + 100, row * 80 + 100, 80, 80), 3)
                                
                                pygame.mixer.Sound("assets/sounds/Check.wav").play()

    def highlight_last_move(self, board):
        if board.last_move_start and board.last_move_end:
            start_x = board.last_move_start[1] * 80 + 140
            start_y = board.last_move_start[0] * 80 + 140
            end_x = board.last_move_end[1] * 80 + 140
            end_y = board.last_move_end[0] * 80 + 140
            self.draw_dashed_line(self.screen, (0, 0, 255), (start_x, start_y), (end_x, end_y), 5)
            angle = np.arctan2(end_y - start_y, end_x - start_x)
            arrow_size = 10
            arrow_points = [
                (end_x, end_y),
                (end_x - arrow_size * np.cos(angle - np.pi / 6), end_y - arrow_size * np.sin(angle - np.pi / 6)),
                (end_x - arrow_size * np.cos(angle + np.pi / 6), end_y - arrow_size * np.sin(angle + np.pi / 6))
            ]
            pygame.draw.polygon(self.screen, (0, 0, 255), arrow_points)
            
        if board.computer_move_start and board.computer_move_end:
            start_x = board.computer_move_start[1] * 80 + 140
            start_y = board.computer_move_start[0] * 80 + 140
            end_x = board.computer_move_end[1] * 80 + 140
            end_y = board.computer_move_end[0] * 80 + 140
            self.draw_dashed_line(self.screen, (0, 0, 255), (start_x, start_y), (end_x, end_y), 5)
            angle = np.arctan2(end_y - start_y, end_x - start_x)
            arrow_size = 10
            arrow_points = [
                (end_x, end_y),
                (end_x - arrow_size * np.cos(angle - np.pi / 6), end_y - arrow_size * np.sin(angle - np.pi / 6)),
                (end_x - arrow_size * np.cos(angle + np.pi / 6), end_y - arrow_size * np.sin(angle + np.pi / 6))
            ]
            pygame.draw.polygon(self.screen, (0, 0, 255), arrow_points)

    def draw_dashed_line(self, surface, color, start_pos, end_pos, width, dash_length=10):
        start_x, start_y = start_pos
        end_x, end_y = end_pos
        dash_len = dash_length

        distance = ((end_x - start_x)**2 + (end_y - start_y)**2)**0.5
        dash_num = int(distance / dash_len)

        for i in range(dash_num):
            dash_start = (start_x + (end_x - start_x) * i / dash_num, start_y + (end_y - start_y) * i / dash_num)
            dash_end = (start_x + (end_x - start_x) * (i + 0.5) / dash_num, start_y + (end_y - start_y) * (i + 0.5) / dash_num)
            pygame.draw.line(surface, color, dash_start, dash_end, width)

    def find_king_position(self, board, current_turn):
        for row in range(8):
            for col in range(8):
                piece = board.board[row, col]
                if piece:
                    if isinstance(piece, King):
                        if piece.color == current_turn:
                            return (row, col)
        return None
