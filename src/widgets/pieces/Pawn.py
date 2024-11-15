import numpy as np
from widgets.pieces.__init__ import Piece
from widgets.pieces.Queen import Queen

class Pawn(Piece):
    def __init__(self, color):
        super().__init__(color)
        self.en_passant_target = False

    def get_possible_moves(self, board, position):
        moves = []
        
        if self.color == 'white': 
            start_row = 6
            direction = -1
        elif self.color == 'black':
            start_row = 1
            direction = 1

        new_pos = position + np.array([direction, 0])
        if 0 <= new_pos[0] < 8:
            if board[new_pos[0], new_pos[1]] is None:
                moves.append(tuple(new_pos))

        if position[0] == start_row:
            new_pos = position + np.array([2 * direction, 0])
            
            
            if 0 <= new_pos[0] < 8:
                if board[new_pos[0], new_pos[1]] is None: 
                    if board[position[0] + direction, position[1]] is None:
                        moves.append(tuple(new_pos))
        
        if self.color == 'white':
            capture_directions = np.array([[-1, 1], [1, 1]])
        elif self.color == 'black':
            capture_directions = np.array([[-1, -1], [1, -1]])

        for d in capture_directions:
            new_pos = position + np.array([direction, d[0]])
            if 0 <= new_pos[0] < 8 and 0 <= new_pos[1] < 8:
                target_piece = board[new_pos[0], new_pos[1]]
                if target_piece is not None:
                    if target_piece.color != self.color:
                        moves.append(tuple(new_pos))
    
        for d in [-1, 1]:
            new_pos = position + np.array([0, d])
            if 0 <= new_pos[1] < 8:
                if isinstance(board[position[0], new_pos[1]], Pawn):
                    if board[position[0], new_pos[1]].color != self.color:
                            if (self.color == 'white' and position[0] == 3) or (self.color == 'black' and position[0] == 4):
                                moves.append((position[0] + direction, new_pos[1]))
                            
        return moves

    def promote(self, board, position):
        is_white_promotion = (self.color == 'white' and position[0] == 0)
        is_black_promotion = (self.color == 'black' and position[0] == 7)
        
        if is_white_promotion or is_black_promotion:
            board[position[0], position[1]] = Queen(self.color)