import pygame
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.optim as optim
import pickle
import concurrent.futures
from pydub import AudioSegment
from pydub.playback import play

from widgets.pieces.King import King
from widgets.pieces.Rook import Rook
from widgets.pieces.Bishop import Bishop
from widgets.pieces.Knight import Knight  
from widgets.pieces.Pawn import Pawn
from widgets.pieces.Queen import Queen
from widgets.informations.BoardDisplay import BoardDisplay
from widgets.informations.TimerDisplay import TimerDisplay
from widgets.informations.KillLogDisplay import KillLogDisplay
from widgets.informations.GameOverDisplay import GameOverDisplay
from widgets.buttons.MenuButtonDisplay import MenuButtonDisplay
from features.decision.MonteCarloTreeSearch import MonteCarloTreeSearch
from features.decision.MonteCarloTreeSearchNode import MonteCarloTreeSearchNode
from features.decision.ReplayBuffer import ReplayBuffer
from features.decision.PolicyNetwork import PolicyNetwork
from features.decision.ValueNetwork import ValueNetwork
from styles.PieceImage import PieceImage
from styles.BackgroundImage import BackgroundImage
from pages.Menu import Menu
from pages.Board import Board


class Game:
    def __init__(self, play_with_computer=False, computer_vs_computer=False):
        pygame.init()
        pygame.mixer.init()
        self.screen = pygame.display.set_mode((1000, 1000))
        self.background = BackgroundImage.load_image()
        self.background = pygame.transform.scale(self.background, (640, 640))  
        self.piece_images = PieceImage.load_image()
        self.board_display = BoardDisplay(self.screen, self.background, self.piece_images)
        self.board = Board(self.screen, self.background, self.board_display)
        self.current_turn = 'white'
        self.selected_piece = None
        self.selected_position = None
        self.turn_time_limit = 60
        self.turn_start_time = pygame.time.get_ticks()
        self.play_with_computer = play_with_computer
        self.computer_vs_computer = computer_vs_computer
        self.kill_log = []
        self.replay_buffer = ReplayBuffer(10000)  
        self.game_over_display = GameOverDisplay(self.screen)
        pygame.display.set_caption("White turn")
        self.clock = pygame.time.Clock()

        self.menu_button = pygame.image.load("assets/images/Buttons/Menu.png").convert_alpha()
        self.menu_button = pygame.transform.scale(self.menu_button, (50, 50))

        self.timer_display = TimerDisplay(self.screen, self.turn_time_limit)
        self.kill_log_display = KillLogDisplay(self.screen, self.piece_images)
        self.menu_button_display = MenuButtonDisplay(self.screen, self.menu_button)

    def play(self):
        while True:
            self.board_display.display_board(self.board, self.selected_piece, self.selected_position, self.current_turn)
            self.timer_display.display_timer(self.turn_start_time, self.current_turn, self.board)
            self.kill_log_display.display_kill_log(self.kill_log)
            self.menu_button_display.display_menu_button()

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    self.handle_mousebuttondown()
                elif event.type == pygame.KEYDOWN:
                    self.handle_keydown(event)

            pygame.display.flip()
            self.clock.tick(30)

    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == pygame.MOUSEBUTTONDOWN:
                self.handle_mousebuttondown()

    def handle_mousebuttondown(self):
        mouse_x, mouse_y = pygame.mouse.get_pos()
        if 900 <= mouse_x <= 980 and 20 <= mouse_y <= 100:
            menu = Menu()
            menu.run()
        else:
            row, col = (mouse_y - 100) // 80, (mouse_x - 100) // 80
            if 0 <= row < 8 and 0 <= col < 8:
                self.handle_board_click(row, col)
                
    def handle_board_click(self, row, col):
        if self.selected_piece:
            start_pos = self.selected_position
            end_pos = (row, col)
            if start_pos == end_pos:
                self.selected_piece = None
                self.selected_position = None
            else:
                self.board.move_piece(start_pos, end_pos)
                
                move_sound = pygame.mixer.Sound("../../assets/sounds/Move.wav")
                move_sound.play()
                
                self.selected_piece = None
                self.selected_position = None
                self.switch_turn()
                self.turn_start_time = pygame.time.get_ticks()

                if self.is_game_over():
                    return

                if self.play_with_computer:
                    if self.current_turn == 'black':
                        self.computer_decision()
                elif self.computer_vs_computer:
                    while True:
                        self.computer_decision()
                        if self.is_game_over():
                            return
        else:
            piece = self.board.board[row, col]
            if piece and piece.color == self.current_turn:
                self.selected_piece = piece
                self.selected_position = (row, col)
    
    def handle_keydown(self, event):
        if event.key == pygame.K_ESCAPE:
            menu = Menu()
            menu.run()

    def switch_turn(self):
        self.current_turn = 'black' if self.current_turn == 'white' else 'white'
        pygame.display.set_caption(f"{self.current_turn.capitalize()} turn")
        for row in self.board.board:
            for piece in row:
                if isinstance(piece, Pawn):
                    self.en_passant_target = False

    def is_game_over(self):
        if self.board.is_checkmate(self.current_turn):
            self.game_over_display.display_game_over(self.board, self.current_turn)
            return True
        elif self.board.is_stalemate(self.current_turn):
            self.game_over_display.display_game_over(self.board, self.current_turn)
            return True
        elif not self.board.king_exists(self.current_turn):
            self.game_over_display.display_game_over(self.board, self.current_turn)
            return True
        return False
    

    def computer_decision(self):
        state = self.board_to_numeric(self.board.board).flatten()
        reward = 0
        actions = [
            ((row, col), move)
            for row in range(8)
            for col in range(8)
            if (piece := self.board.board[row, col]) and piece.color == self.current_turn
            for move in piece.get_possible_moves(self.board.board, (row, col))
        ]

        if not actions:
            return
        policy_net = PolicyNetwork(len(actions))
        value_net = ValueNetwork()
        optimizer = optim.Adam(list(policy_net.parameters()) + list(value_net.parameters()), lr=0.0001)
        gamma = 0.99
        simulation_count = 3

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
        self.switch_turn()
        self.turn_start_time = pygame.time.get_ticks()

        if isinstance(self.board.board[move[0], move[1]], Pawn) and move[0] in {0, 7}:
            self.board.board[move[0], move[1]].promote(self.board.board, move)

        self.board.computer_move_start = (start_row, start_col)
        self.board.computer_move_end = move
        self.board_display.display_board(self.board, self.selected_piece, self.selected_position, self.current_turn)

        torch.save(policy_net.state_dict(), 'policy_net.pth')
        torch.save(value_net.state_dict(), 'value_net.pth')
        with open('replay_buffer.pkl', 'wb') as f:
            pickle.dump(self.replay_buffer, f)
        
        next_state = self.board_to_numeric(self.board.board).flatten()
        self.update_policy_and_value_net(policy_net, value_net, optimizer, state, best_action, reward, next_state, gamma)

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

        value_net.load_state_dict(value_net.state_dict() * (1 - alpha) + value.item() * alpha)


