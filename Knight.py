import numpy as np
from Piece import Piece

class Knight(Piece):
    def get_possible_moves(self, board, position):
        moves = []
        directions = np.array([[-2, -1], [-2, 1], [-1, -2], [-1, 2], [1, -2], [1, 2], [2, -1], [2, 1]])
        for d in directions:
            new_pos = position + d
            if np.all((0 <= new_pos) & (new_pos < 8)):
                if board[new_pos[0], new_pos[1]] is None or board[new_pos[0], new_pos[1]].color != self.color:
                    moves.append(tuple(new_pos))
        return moves