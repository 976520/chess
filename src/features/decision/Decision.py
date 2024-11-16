import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.optim as optim
import pickle
import concurrent.futures

from widgets.pieces.King import King
from widgets.pieces.Rook import Rook
from widgets.pieces.Bishop import Bishop
from widgets.pieces.Knight import Knight  
from widgets.pieces.Pawn import Pawn
from widgets.pieces.Queen import Queen
from features.decision.MonteCarloTreeSearch import MonteCarloTreeSearch
from features.decision.MonteCarloTreeSearchNode import MonteCarloTreeSearchNode
from features.decision.ReplayBuffer import ReplayBuffer
from features.decision.PolicyNetwork import PolicyNetwork
from features.decision.ValueNetwork import ValueNetwork


class Decision:
    def __init__(self, board, current_turn, kill_log, replay_buffer, alpha=0.01):
        self.board = board
        self.current_turn = current_turn
        self.kill_log = kill_log
        self.replay_buffer = replay_buffer
        self.alpha = alpha  
        self.replay_buffer = ReplayBuffer(10000)

    def computer_decision(self):
        state = self.board_to_numeric(self.board.board).flatten()
        reward = 0
        actions = []
        for row in range(8):
            for col in range(8):
                piece = self.board.board[row, col]
                if piece:
                    if piece.color == self.current_turn:
                        possible_moves = piece.get_possible_moves(self.board.board, (row, col))
                        for move in possible_moves:
                            actions.append(((row, col), move))

        if not actions:
            return
        policy_net = PolicyNetwork(len(actions))
        value_net = ValueNetwork()
        optimizer = optim.Adam(list(policy_net.parameters()) + list(value_net.parameters()), lr=0.0001)
        gamma = 0.99
        simulation_count = 4

        mcts = MonteCarloTreeSearch(policy_net, value_net)
        root = MonteCarloTreeSearchNode(state)

        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            priors = policy_net(state_tensor).squeeze(0).numpy()

        root.expand(actions, priors)

        def run_simulation(root, actions):
            for _ in range(simulation_count):
                node = root
                while not node.is_leaf():
                    node = mcts.best_child(node)
                if node.visits > 0:
                    node.expand(actions, )
                self.reward = self.evaluate_board()
                while node:
                    node.update(reward)
                    node = node.parent

        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = []
            for _ in range(simulation_count):
                futures.append(executor.submit(run_simulation, root, actions))
            concurrent.futures.wait(futures)

        best_action = mcts.best_action(root)
        if not best_action:
            return

        (start_row, start_col), move = best_action
        if self.board.board[move[0], move[1]]:
            self.kill_log.append((self.board.board[start_row, start_col], self.board.board[move[0], move[1]]))

        self.board.computer_move_piece((start_row, start_col), move)

        if isinstance(self.board.board[move[0], move[1]], Pawn):
            if move[0] in {0, 7}:
                self.board.board[move[0], move[1]].promote(self.board.board, move)

        self.board.computer_move_start = (start_row, start_col)
        self.board.computer_move_end = move

        torch.save(policy_net.state_dict(), 'policy_net.pth')
        torch.save(value_net.state_dict(), 'value_net.pth')
        with open('replay_buffer.pkl', 'wb') as f:
            pickle.dump(self.replay_buffer, f)
        
        next_state = self.board_to_numeric(self.board.board).flatten()
        self.update_policy_and_value_net(policy_net, value_net, optimizer, state, best_action, reward, next_state, gamma, self.alpha)

    def evaluate_board(self): # a
        piece_values = {
            King: 1000,
            Queen: 9,
            Rook: 5,
            Bishop: 3,
            Knight: 3,
            Pawn: 1
        }

        def evaluate_piece(piece):
            if piece:
                value = piece_values[type(piece)]
                return value if piece.color == 'black' else -value
            return 0

        with concurrent.futures.ThreadPoolExecutor() as executor:
            pieces = []
            for row in self.board.board:
                for piece in row:
                    pieces.append(piece)
            scores = executor.map(evaluate_piece, pieces)

        return sum(scores)

    def board_to_numeric(self, board): 
        numeric_board = np.zeros((8, 8), dtype=np.float32)
        for row in range(8):
            for col in range(8):
                piece = board[row, col]
                if piece is None:
                    numeric_board[row, col] = 0
                elif piece.color == 'black':
                    numeric_board[row, col] = -1
                else:
                    numeric_board[row, col] = 1
        return numeric_board

    def choose_action(self, state, actions, policy_net, value_net, epsilon=0.1): # a
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        
        with torch.no_grad():
            policy = policy_net(state_tensor).squeeze(0).numpy()
            value = value_net(state_tensor).item()

        action_probabilities = np.zeros(len(actions))

        if len(policy) == len(actions):
            action_probabilities = np.exp(policy) / np.sum(np.exp(policy)) 
        else:
            for idx, action in enumerate(actions):
                action_probabilities[idx] = policy[actions.index(action)]

            if np.sum(action_probabilities) > 0:
                action_probabilities /= np.sum(action_probabilities)
            else:
                action_probabilities = np.ones(len(actions)) / len(actions)  

        if np.random.rand() < epsilon:
            return np.random.choice(len(actions)), None  
        else:
            return np.random.choice(len(actions), p=action_probabilities), value 

    def update_policy_and_value_net(self, policy_net, value_net, optimizer, state, action, reward, next_state, gamma, alpha):
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        next_state_tensor = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0)
        action_tensor = torch.tensor([action], dtype=torch.int64)
        reward_tensor = torch.tensor([reward], dtype=torch.float32)

        with torch.no_grad():
            next_policy = policy_net(next_state_tensor).squeeze(0)
            next_value = value_net(next_state_tensor).item()
            next_action_probabilities = next_policy / torch.sum(next_policy)
            next_q_value = torch.sum(next_action_probabilities * next_value)

        policy = policy_net(state_tensor).squeeze(0)
        value = value_net(state_tensor).item()
        action_probability = policy[action_tensor]
        
        q_value = reward_tensor + gamma * next_q_value
        td_error = q_value - value

        policy_loss = (-torch.log(action_probability) * td_error).mean()
        value_loss = td_error.pow(2).mean()

        optimizer.zero_grad()
        (policy_loss + value_loss).backward()
        optimizer.step()

        for param in value_net.parameters():
            param.data.mul_(1 - alpha).add_(value * alpha)