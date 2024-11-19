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
    def __init__(self, chess_board, current_turn_color, captured_pieces_log, replay_memory_buffer, learning_rate_alpha=0.01):
        self.chess_board = chess_board
        self.current_turn_color = current_turn_color
        self.captured_pieces_log = captured_pieces_log
        self.replay_memory_buffer = replay_memory_buffer
        self.learning_rate_alpha = learning_rate_alpha  
        self.replay_memory_buffer = ReplayBuffer(10000)

    def make_computer_decision(self):
        current_state = self.convert_board_to_numeric(self.chess_board.board).flatten()
        possible_actions = self.get_possible_actions()

        if not possible_actions:
            return

        policy_network = PolicyNetwork(len(possible_actions))
        value_network = ValueNetwork()
        adam_optimizer = optim.Adam(list(policy_network.parameters()) + list(value_network.parameters()), lr=0.0001)
        discount_factor_gamma = 0.99
        number_of_simulations = 800

        monte_carlo_tree_search = MonteCarloTreeSearch(policy_network, value_network, num_simulations=number_of_simulations)
        action = monte_carlo_tree_search.search(current_state, possible_actions)

        if action is None:
            return

        (start_row_index, start_column_index), move = action
        if self.chess_board.board[move[0], move[1]]:
            self.captured_pieces_log.append((self.chess_board.board[start_row_index, start_column_index], self.chess_board.board[move[0], move[1]]))

        self.chess_board.computer_move_piece((start_row_index, start_column_index), move)

        if move[0] in {0, 7}:
            if isinstance(self.chess_board.board[move[0], move[1]], Pawn):
                self.chess_board.board[move[0], move[1]].promote(self.chess_board.board, move)

        torch.save(policy_network.state_dict(), 'policy_network.pth')
        torch.save(value_network.state_dict(), 'value_network.pth')
        with open('replay_memory_buffer.pkl', 'wb') as replay_buffer_file:
            pickle.dump(self.replay_memory_buffer, replay_buffer_file)

        next_state = self.convert_board_to_numeric(self.chess_board.board).flatten()
        immediate_reward = self.evaluate_chess_board()
        self.update_policy_and_value_network(policy_network, value_network, adam_optimizer, current_state, action, immediate_reward, next_state, discount_factor_gamma, self.learning_rate_alpha)

    def evaluate_chess_board(self): 
        piece_value_mapping = {
            King: 1000,
            Queen: 9,
            Rook: 5,
            Bishop: 3,
            Knight: 3,
            Pawn: 1
        }

        def evaluate_individual_piece(chess_piece):
            if chess_piece:
                piece_value = piece_value_mapping[type(chess_piece)]
                return piece_value if chess_piece.color == 'black' else -piece_value
            return 0

        with concurrent.futures.ThreadPoolExecutor() as thread_executor:
            all_pieces = []
            for row in self.chess_board.board:
                for chess_piece in row:
                    all_pieces.append(chess_piece)
            piece_scores = thread_executor.map(evaluate_individual_piece, all_pieces)

        return sum(piece_scores)

    def convert_board_to_numeric(self, chess_board): 
        numeric_chess_board = np.zeros((8, 8), dtype=np.float32)
        for row_index in range(8):
            for column_index in range(8):
                chess_piece = chess_board[row_index, column_index]
                if chess_piece is None:
                    numeric_chess_board[row_index, column_index] = 0
                elif chess_piece.color == 'black':
                    numeric_chess_board[row_index, column_index] = -1
                else:
                    numeric_chess_board[row_index, column_index] = 1
        return numeric_chess_board

    def select_action(self, current_state, possible_actions, policy_network, value_network, exploration_epsilon=0.1): 
        current_state_tensor = torch.tensor(current_state, dtype=torch.float32).unsqueeze(0)
        
        with torch.no_grad():
            action_policy = policy_network(current_state_tensor).squeeze(0).numpy()
            state_value = value_network(current_state_tensor).item()

        action_probabilities = np.zeros(len(possible_actions))

        if len(action_policy) == len(possible_actions):
            action_probabilities = np.exp(action_policy) / np.sum(np.exp(action_policy)) 
        else:
            for action_index, action in enumerate(possible_actions):
                action_probabilities[action_index] = action_policy[possible_actions.index(action)]

            if np.sum(action_probabilities) > 0:
                action_probabilities /= np.sum(action_probabilities)
            else:
                action_probabilities = np.ones(len(possible_actions)) / len(possible_actions)  

        if np.random.rand() < exploration_epsilon:
            return np.random.choice(len(possible_actions)), None  
        else:
            return np.random.choice(len(possible_actions), p=action_probabilities), state_value 

    def update_policy_and_value_network(self, policy_network, value_network, adam_optimizer, current_state, selected_action, immediate_reward, next_state, discount_factor_gamma, learning_rate_alpha):
        current_state_tensor = torch.tensor(current_state, dtype=torch.float32).unsqueeze(0)
        next_state_tensor = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0)
        selected_action_tensor = torch.tensor([selected_action], dtype=torch.int64)
        immediate_reward_tensor = torch.tensor([immediate_reward], dtype=torch.float32)

        with torch.no_grad():
            next_action_policy = policy_network(next_state_tensor).squeeze(0)
            next_state_value = value_network(next_state_tensor).item()
            next_action_probabilities = next_action_policy / torch.sum(next_action_policy)
            next_q_value = torch.sum(next_action_probabilities * next_state_value)

        current_action_policy = policy_network(current_state_tensor).squeeze(0)
        current_state_value = value_network(current_state_tensor).item()
        selected_action_probability = current_action_policy[selected_action_tensor]
        
        target_q_value = immediate_reward_tensor + discount_factor_gamma * next_q_value
        temporal_difference_error = target_q_value - current_state_value

        policy_loss = (-torch.log(selected_action_probability) * temporal_difference_error).mean()
        value_loss = temporal_difference_error.pow(2).mean()

        adam_optimizer.zero_grad()
        (policy_loss + value_loss).backward()
        adam_optimizer.step()

        for parameter in value_network.parameters():
            parameter.data.mul_(1 - learning_rate_alpha).add_(current_state_value * learning_rate_alpha)

    def get_possible_actions(self):
        possible_actions = []
        for row in range(8):
            for col in range(8):
                piece = self.chess_board.board[row, col]
                if piece and piece.color == self.current_turn_color:
                    moves = piece.get_possible_moves(self.chess_board.board, (row, col))
                    for move in moves:
                        possible_actions.append(((row, col), move))
        return possible_actions