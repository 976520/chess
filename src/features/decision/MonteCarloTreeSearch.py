import numpy as np
import torch

from features.decision.MonteCarloTreeSearchNode import MonteCarloTreeSearchNode

class MonteCarloTreeSearch:
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
        root = MonteCarloTreeSearchNode(root_state)
        
        policy, value = self.get_policy_value(root_state)
        root.expand(legal_actions, policy)
        
        for _ in range(self.num_simulations):
            node = root
            search_path = [node]
            
            while node is not None and not node.is_leaf() and node.children:
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
            
            if node is None:
                continue  
            
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
        
        best_child = root.children[0]
        for child in root.children:
            if child.visits > best_child.visits:
                best_child = child
    
        return best_child.action
    
    def best_action(self, node):
        best_child = node.children[0]
        for child in node.children:
            if child.visits > best_child.visits:
                best_child = child
        return best_child.action