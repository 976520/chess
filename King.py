import numpy as np
from Piece import Piece
from Rook import Rook

class King(Piece):
    def __init__(self, color):
        super().__init__(color) 
        
        self.has_moved = False

    def get_possible_moves(self, board, position):
        moves = []
        directions = np.array([[-1, -1], [-1, 0], [-1, 1], [0, -1], [0, 1], [1, -1], [1, 0], [1, 1]])
        for d in directions:
            new_pos = position + d
            if np.all((0 <= new_pos) & (new_pos < 8)):
                if (board[new_pos[0], new_pos[1]] is None) or (board[new_pos[0], new_pos[1]].color != self.color):
                    moves.append(tuple(new_pos))

        if not self.has_moved:
            if self.color == 'white':
                row = 7
            elif self.color == 'black':
                row = 0

            if isinstance(board[row, 7], Rook):
                if not board[row, 7].has_moved:
                    if np.all(board[row, 5:7] == None):
                        moves.append((row, 6))

            if isinstance(board[row, 0], Rook):
                if not board[row, 0].has_moved:
                    if np.all(board[row, 1:4] == None):
                        moves.append((row, 2))

        return moves