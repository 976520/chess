import numpy as np

class MonteCarloTreeSearchNode:
    def __init__(self, state, parent=None, action=None):
        self.state = state
        self.parent = parent
        self.action = action
        self.children = []
        self.visits = 0
        self.value = 0.0
        self.prior = 0.0

    def is_leaf(self):
        return len(self.children) == 0

    def expand(self, actions, priors):
        for action, prior in zip(actions, priors):
            next_state = self.get_next_state(self.state, action)
            child = MonteCarloTreeSearchNode(next_state, parent=self, action=action)
            child.prior = prior
            self.children.append(child)

    def get_next_state(self, state, action):
        new_board = np.copy(state).reshape(8, 8)
        (start_pos, end_pos) = action
        piece = new_board[start_pos[0], start_pos[1]]
        new_board[end_pos[0], end_pos[1]] = piece
        new_board[start_pos[0], start_pos[1]] = None
        return new_board

    def update(self, value):
        self.visits += 1
        self.value += (value - self.value) / self.visits
