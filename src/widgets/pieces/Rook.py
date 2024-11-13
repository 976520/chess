import numpy as np
from widgets.pieces.__init__ import Piece

class Rook(Piece):
    def __init__(self, color):
        super().__init__(color)
        self.has_moved = False

    def get_possible_moves(self, board, position):
        moves = []
        directions = np.array([[-1, 0], [0, -1], [0, 1], [1, 0]])
        for d in directions:
            for i in range(1, 8):
                new_pos = position + d * i
                if np.all((0 <= new_pos) & (new_pos < 8)):
                    if board[new_pos[0], new_pos[1]] is None:
                        moves.append(tuple(new_pos))
                    elif board[new_pos[0], new_pos[1]].color != self.color:
                        moves.append(tuple(new_pos))
                        break
                    else:
                        break
                else:
                    break
        return moves