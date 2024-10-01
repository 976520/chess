import numpy as np
import torch
import concurrent.futures

class MCTSNode:
    def __init__(self, state, parent=None, action=None):
        self.state = state
        self.parent = parent
        self.action = action
        self.children = []
        self.visits = 0
        self.value = 0.0

    def is_leaf(self):
        return len(self.children) == 0

    def expand(self, actions):
        for action in actions:
            next_state = self.get_next_state(self.state, action)
            self.children.append(MCTSNode(next_state, parent=self, action=action)) 

    def get_next_state(self, state, action):
        new_board = np.copy(state).reshape(8, 8)  
        (start_pos, end_pos) = action
        piece = new_board[start_pos[0], start_pos[1]]
        new_board[end_pos[0], end_pos[1]] = piece
        new_board[start_pos[0], start_pos[1]] = None
        return new_board

    def update(self, reward):
        self.visits += 1
        self.value += (reward - self.value) / self.visits

class MCTS:
    def __init__(self, policy_net, value_net):
        self.policy_net = policy_net
        self.value_net = value_net

    def best_child(self, node):
        exploration_constant = 1.4  
        best_score = -float('inf')
        best_child_node = None
        
        state_tensor = torch.tensor(node.state, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            policy = self.policy_net(state_tensor).squeeze(0).numpy()
            value = self.value_net(state_tensor).item()
        
        def compute_score(i, child_node):
            policy_score = policy[i] if i < len(policy) else 0
            score = child_node.value + exploration_constant * policy_score * np.sqrt(np.log(node.visits + 1) / (child_node.visits + 1))
            return score, child_node

        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [executor.submit(compute_score, i, child_node) for i, child_node in enumerate(node.children)]
            for future in concurrent.futures.as_completed(futures):
                score, child_node = future.result()
                if score > best_score:
                    best_score = score
                    best_child_node = child_node

        return best_child_node if best_child_node is not None else node 

    def best_action(self, root):
        return max(root.children, key=lambda child_node: child_node.visits).action