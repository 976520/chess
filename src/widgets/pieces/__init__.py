class Piece:
    def __init__(self, color):
        self.color = color

    def get_possible_moves(self, board, position):
        raise NotImplementedError()
