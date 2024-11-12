import numpy as np
import torch
from mctx import mcts, SearchParams, PolicyOutput

class MCTSNode:
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
            child = MCTSNode(next_state, parent=self, action=action)
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

class MCTS:
    def __init__(self, policy_net, value_net, num_simulations=800):
        self.policy_net = policy_net
        self.value_net = value_net
        self.num_simulations = num_simulations
        
    def get_policy_value(self, state):
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            policy = self.policy_net(state_tensor).squeeze(0)
            value = self.value_net(state_tensor).item()
        return policy.numpy(), value

    def search(self, root_state, legal_actions):
        root = MCTSNode(root_state)
        
        policy, value = self.get_policy_value(root_state)
        root.expand(legal_actions, policy)
        
        for _ in range(self.num_simulations):
            node = root
            search_path = [node]
            
            while not node.is_leaf() and node.children:
                best_ucb = -float('inf')
                best_child = None
                
                total_visits = sum(child.visits for child in node.children)
                exploration_constant = 1.4
                
                for child in node.children:
                    if child.visits == 0:
                        ucb = float('inf')
                    else:
                        ucb = child.value + exploration_constant * child.prior * \
                             np.sqrt(total_visits) / (1 + child.visits)
                    
                    if ucb > best_ucb:
                        best_ucb = ucb
                        best_child = child
                        
                node = best_child
                search_path.append(node)
            
            if node.visits > 0:
                policy, value = self.get_policy_value(node.state)
                node.expand(legal_actions, policy)
                if node.children:
                    node = np.random.choice(node.children)
                    search_path.append(node)
            
            value = self.get_policy_value(node.state)[1]
            for node in reversed(search_path):
                node.update(value)
                value = -value 
        
        return max(root.children, key=lambda c: c.visits).action