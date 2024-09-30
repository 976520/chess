import pygame
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from King import King
from Rook import Rook
from Bishop import Bishop
from Knight import Knight
from Pawn import Pawn
from Queen import Queen
from Board import Board

class Game:
    def __init__(self, play_with_computer=False):
        pygame.init()
        self.screen = pygame.display.set_mode((1000, 1000))
        self.board = Board(self.screen)
        self.current_turn = 'white'
        self.selected_piece = None
        self.selected_position = None
        self.turn_time_limit = 60
        self.turn_start_time = pygame.time.get_ticks()
        self.play_with_computer = play_with_computer

        pygame.display.set_caption("White turn")
        self.clock = pygame.time.Clock()
        self.background = pygame.image.load("assets/Background.png").convert()
        self.background = pygame.transform.scale(self.background, (640, 640))  

        self.piece_images = {
            'Pawn_b': pygame.transform.scale(pygame.image.load("assets/Pawn_b.png").convert_alpha(), (80, 80)),
            'Pawn_w': pygame.transform.scale(pygame.image.load("assets/Pawn_w.png").convert_alpha(), (80, 80)),
            'Rook_b': pygame.transform.scale(pygame.image.load("assets/Rook_b.png").convert_alpha(), (80, 80)),
            'Rook_w': pygame.transform.scale(pygame.image.load("assets/Rook_w.png").convert_alpha(), (80, 80)),
            'Knight_b': pygame.transform.scale(pygame.image.load("assets/Knight_b.png").convert_alpha(), (80, 80)),
            'Knight_w': pygame.transform.scale(pygame.image.load("assets/Knight_w.png").convert_alpha(), (80, 80)),
            'Bishop_b': pygame.transform.scale(pygame.image.load("assets/Bishop_b.png").convert_alpha(), (80, 80)),
            'Bishop_w': pygame.transform.scale(pygame.image.load("assets/Bishop_w.png").convert_alpha(), (80, 80)),
            'Queen_b': pygame.transform.scale(pygame.image.load("assets/Queen_b.png").convert_alpha(), (80, 80)),
            'Queen_w': pygame.transform.scale(pygame.image.load("assets/Queen_w.png").convert_alpha(), (80, 80)),
            'King_b': pygame.transform.scale(pygame.image.load("assets/King_b.png").convert_alpha(), (80, 80)),
            'King_w': pygame.transform.scale(pygame.image.load("assets/King_w.png").convert_alpha(), (80, 80)),
        }

    def play(self):
        while not self.is_game_over():
            self.handle_events()
            self.display_board()
            self.display_timer()
            pygame.display.flip()
            self.clock.tick(30)
                
            if self.current_turn == 'black':
                if self.play_with_computer:
                    self.computer_move()
                    self.turn_start_time = pygame.time.get_ticks()

        self.display_board()
        pygame.display.flip()
        self.display_game_over()

    def display_board(self):
        self.screen.fill((128, 128, 128))
        self.screen.blit(self.background, (100, 100))  
        colors = [(255, 255, 255), (0, 0, 0)]
        font = pygame.font.SysFont(None, 24)
        mouse_position = pygame.mouse.get_pos()
        mouse_row, mouse_col = (mouse_position[1] - 100) // 80, (mouse_position[0] - 100) // 80

        self.draw_board(colors, font)
        self.highlight_mouse_position(mouse_row, mouse_col)
        self.highlight_selected_piece()
        self.highlight_check()
        self.highlight_last_move()

    def draw_board(self, colors, font):
        for row in range(8):
            for col in range(8):
                color = colors[(row + col) % 2]
                pygame.draw.rect(self.screen, color, pygame.Rect(col * 80 + 100, row * 80 + 100, 80, 80), 1)
                piece = self.board.board[row, col]
                if piece:
                    piece_image = self.piece_images[type(piece).__name__ + '_' + piece.color[0]]
                    self.screen.blit(piece_image, (col * 80 + 100, row * 80 + 100))
                
                coord_text = font.render(f"{chr(97 + col)}{8 - row}", True, (0, 0, 0, 128) if color == (255, 255, 255) else (255, 255, 255, 128))
                self.screen.blit(coord_text, (col * 80 + 105, row * 80 + 105))

    def highlight_mouse_position(self, mouse_row, mouse_col):
        if (0 <= mouse_row < 8) and (0 <= mouse_col < 8):
            s = pygame.Surface((80, 80), pygame.SRCALPHA)
            s.fill((255, 255, 255, 128))
            self.screen.blit(s, (mouse_col * 80 + 100, mouse_row * 80 + 100))

    def highlight_selected_piece(self):
        if self.selected_piece:
            pygame.draw.rect(self.screen, (0, 0, 225), pygame.Rect(self.selected_position[1] * 80 + 100, self.selected_position[0] * 80 + 100, 80, 80), 3)
            possible_moves = self.selected_piece.get_possible_moves(self.board.board, self.selected_position)
            if possible_moves:
                for move in possible_moves:
                    if self.board.board[move[0], move[1]] is None:
                        pygame.draw.circle(self.screen, (0, 255, 0), (move[1] * 80 + 140, move[0] * 80 + 140), 7)  
                    else:
                        pygame.draw.circle(self.screen, (255, 0, 0), (move[1] * 80 + 140, move[0] * 80 + 140), 7)  

    def highlight_check(self):
        if self.board.is_in_check(self.current_turn):
            king_position = self.find_king_position()
            for i in range(8):
                for j in range(8):
                    piece = self.board.board[i, j]
                    if piece and piece.color != self.current_turn:
                        possible_moves = piece.get_possible_moves(self.board.board, (i, j))
                        if possible_moves and king_position in possible_moves:
                            pygame.draw.rect(self.screen, (255, 0, 0), pygame.Rect(j * 80 + 100, i * 80 + 100, 80, 80), 3)

    def highlight_last_move(self):
        if self.board.last_move_start and self.board.last_move_end:
            start_x = self.board.last_move_start[1] * 80 + 140
            start_y = self.board.last_move_start[0] * 80 + 140
            end_x = self.board.last_move_end[1] * 80 + 140
            end_y = self.board.last_move_end[0] * 80 + 140
            self.draw_dashed_line(self.screen, (0, 0, 255), (start_x, start_y), (end_x, end_y), 5)
            angle = np.arctan2(end_y - start_y, end_x - start_x)
            arrow_size = 10
            arrow_points = [
                (end_x, end_y),
                (end_x - arrow_size * np.cos(angle - np.pi / 6), end_y - arrow_size * np.sin(angle - np.pi / 6)),
                (end_x - arrow_size * np.cos(angle + np.pi / 6), end_y - arrow_size * np.sin(angle + np.pi / 6))
            ]
            pygame.draw.polygon(self.screen, (0, 0, 255), arrow_points)
            pygame.draw.circle(self.screen, (0, 0, 255), (start_x, start_y), 7)  

        if self.board.computer_move_start and self.board.computer_move_end:
            start_x = self.board.computer_move_start[1] * 80 + 140
            start_y = self.board.computer_move_start[0] * 80 + 140
            end_x = self.board.computer_move_end[1] * 80 + 140
            end_y = self.board.computer_move_end[0] * 80 + 140
            self.draw_dashed_line(self.screen, (0, 0, 255), (start_x, start_y), (end_x, end_y), 5)
            angle = np.arctan2(end_y - start_y, end_x - start_x)
            arrow_size = 10
            arrow_points = [
                (end_x, end_y),
                (end_x - arrow_size * np.cos(angle - np.pi / 6), end_y - arrow_size * np.sin(angle - np.pi / 6)),
                (end_x - arrow_size * np.cos(angle + np.pi / 6), end_y - arrow_size * np.sin(angle + np.pi / 6))
            ]
            pygame.draw.polygon(self.screen, (0, 0, 255), arrow_points)

    def draw_dashed_line(self, surface, color, start_pos, end_pos, width, dash_length=10):
        x1, y1 = start_pos
        x2, y2 = end_pos
        dl = dash_length

        distance = ((x2 - x1)**2 + (y2 - y1)**2)**0.5
        dash_num = int(distance / dl)

        for i in range(dash_num):
            start = (x1 + (x2 - x1) * i / dash_num, y1 + (y2 - y1) * i / dash_num)
            end = (x1 + (x2 - x1) * (i + 0.5) / dash_num, y1 + (y2 - y1) * (i + 0.5) / dash_num)
            pygame.draw.line(surface, color, start, end, width)

    def find_king_position(self):
        for i in range(8):
            for j in range(8):
                piece = self.board.board[i, j]
                if piece and isinstance(piece, King) and piece.color == self.current_turn:
                    return (i, j)
        return None

    def display_timer(self):
        elapsed_time = (pygame.time.get_ticks() - self.turn_start_time) / 1000
        remaining_time = max(0, self.turn_time_limit - elapsed_time)
        if (remaining_time == 0):
            self.switch_turn() 
            self.turn_start_time = pygame.time.get_ticks() 

        timer_width = int((remaining_time / self.turn_time_limit) * 640)
        
        if self.current_turn == 'white': 
            if self.board.is_in_check('white'):
                timer_color = (255, 0, 0)
            else:
                timer_color = (0, 0, 255)
            pygame.draw.rect(self.screen, timer_color, pygame.Rect(100, 750, timer_width, 5)) 
        elif self.current_turn == 'black':
            if self.board.is_in_check('black'):
                timer_color = (255, 0, 0)
            else:
                timer_color = (0, 0, 255)
            pygame.draw.rect(self.screen, timer_color, pygame.Rect(100, 85, timer_width, 5))  

        pygame.display.update()

    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == pygame.MOUSEBUTTONDOWN:
                pos = pygame.mouse.get_pos()
                row, col = (pos[1] - 100) // 80, (pos[0] - 100) // 80
                if self.selected_piece:
                    possible_moves = self.selected_piece.get_possible_moves(self.board.board, self.selected_position)
                    if possible_moves and (row, col) in possible_moves:
                        self.board.move_piece(self.selected_position, (row, col))
                        self.switch_turn()
                        self.turn_start_time = pygame.time.get_ticks()
                    self.selected_piece = None
                    self.selected_position = None
                else:
                    piece = self.board.board[row, col]
                    if piece:
                        if piece.color == self.current_turn:
                            self.selected_piece = piece
                            self.selected_position = (row, col)

    def switch_turn(self):
        self.current_turn = 'black' if self.current_turn == 'white' else 'white' 
        pygame.display.set_caption(f"{self.current_turn.capitalize()} turn")

        for row in self.board.board:
            for piece in row:
                if isinstance(piece, Pawn):
                    piece.en_passant_target = False

    def is_game_over(self):
        return self.board.is_checkmate(self.current_turn) or self.board.is_stalemate(self.current_turn)

    def display_game_over(self):
        font = pygame.font.SysFont(None, 74)
        if self.board.is_checkmate(self.current_turn):
            text = font.render(f"Checkmate, {self.current_turn} loses.", True, (255, 0, 0))
        else:
            text = font.render("Stalemate", True, (255, 255, 0))

        modal_surface = pygame.Surface((400, 200))
        modal_surface.fill((0, 0, 0))
        modal_surface.blit(text, (50, 80))

        self.screen.blit(modal_surface, (300, 400))
        pygame.display.flip()

        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                elif event.type == pygame.KEYDOWN or event.type == pygame.MOUSEBUTTONDOWN:
                    return
    
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
            new_board = np.copy(state)
            (start_pos, end_pos) = action
            piece = new_board[start_pos[0], start_pos[1]]
            new_board[end_pos[0], end_pos[1]] = piece
            new_board[start_pos[0], start_pos[1]] = None
            return new_board

        def update(self, reward):
            self.visits += 1
            self.value += (reward - self.value) / self.visits

    def mcts(root, policy_net, value_net, num_simulations):
        for _ in range(num_simulations):
            node = root
            while not node.is_leaf():
                node = best_child(node, policy_net, value_net)
            
            actions = get_actions_for_state(node.state)
            if actions:
                node.expand(actions)
            
            reward = simulate_random_game(node.state)
            
            while node:
                node.update(reward)
                node = node.parent

        return best_action(root)

    def best_child(node, policy_net, value_net):
        c = 1.4  
        best_score = -float('inf')
        best_child = None
        
        for child in node.children:
            score = child.value + c * np.sqrt(np.log(node.visits + 1) / (child.visits + 1))
            if score > best_score:
                best_score = score
                best_child = child

        return best_child

    def best_action(root):
        return max(root.children, key=lambda x: x.visits).action

    class ReplayBuffer:
        def __init__(self, capacity):
            self.buffer = []
            self.capacity = capacity
            self.position = 0

        def push(self, state, action, reward, next_state):
            if len(self.buffer) < self.capacity:
                self.buffer.append(None)
            self.buffer[self.position] = (state, action, reward, next_state)
            self.position = (self.position + 1) % self.capacity

        def sample(self, batch_size):
            return random.sample(self.buffer, batch_size)

        def __len__(self):
            return len(self.buffer)

    def board_to_numeric(board):
        numeric_board = np.zeros((8, 8), dtype=np.float32)
        for i in range(8):
            for j in range(8):
                piece = board[i, j]
                if piece is None:
                    numeric_board[i, j] = 0
                elif piece.color == 'black':
                    numeric_board[i, j] = -1
                else:
                    numeric_board[i, j] = 1
        return numeric_board

    def computer_move(self):
        
        def choose_action(state, actions, policy_net, value_net, epsilon=0.1):
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            
            with torch.no_grad():
                policy = policy_net(state_tensor).squeeze(0).numpy()
                value = value_net(state_tensor).item()

            action_probs = np.zeros(len(actions))

            if len(policy) == len(actions):
                action_probs = np.exp(policy) / np.sum(np.exp(policy)) 
            else:
                for idx, action in enumerate(actions):
                    action_probs[idx] = policy[idx]

                if np.sum(action_probs) > 0:
                    action_probs /= np.sum(action_probs)
                else:
                    action_probs = np.ones(len(actions)) / len(actions)  

            if np.random.rand() < epsilon:
                return np.random.choice(len(actions)), None  
            else:
                return np.random.choice(len(actions), p=action_probs), value 

        def update_policy_and_value_net(policy_net, value_net, optimizer, state, action, reward, next_state, next_actions, gamma):
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            next_state_tensor = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0)
            action_tensor = torch.tensor([action], dtype=torch.int64)
            reward_tensor = torch.tensor([reward], dtype=torch.float32)

            with torch.no_grad():
                next_policy = policy_net(next_state_tensor).squeeze(0)
                next_value = value_net(next_state_tensor).item()
                next_action_probs = next_policy / torch.sum(next_policy)
                next_value = torch.sum(next_action_probs * next_value)

            policy = policy_net(state_tensor).squeeze(0)
            value = value_net(state_tensor).item()
            action_prob = policy[action_tensor]
            advantage = reward_tensor + gamma * next_value - value
            policy_loss = -torch.log(action_prob) * advantage
            value_loss = advantage.pow(2)

            optimizer.zero_grad()
            (policy_loss + value_loss).backward()
            optimizer.step()

        class PolicyNetwork(nn.Module):
            def __init__(self, num_actions):
                super(PolicyNetwork, self).__init__()
                self.conv1 = nn.Conv2d(1, 64, kernel_size=3, padding=1)
                self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
                self.fc1 = nn.Linear(128 * 8 * 8, 512)
                self.fc2 = nn.Linear(512, num_actions)

            def forward(self, x):
                x = torch.relu(self.conv1(x.view(-1, 1, 8, 8)))
                x = torch.relu(self.conv2(x))
                x = x.view(-1, 128 * 8 * 8)
                x = torch.relu(self.fc1(x))
                return self.fc2(x)


        class ValueNetwork(nn.Module):
            def __init__(self):
                super(ValueNetwork, self).__init__()
                self.fc1 = nn.Linear(64, 512)
                self.fc2 = nn.Linear(512, 512)
                self.fc3 = nn.Linear(512, 1)

            def forward(self, x):
                x = torch.relu(self.fc1(x))
                x = torch.relu(self.fc2(x))
                x = self.fc3(x)
                return x

        def board_to_numeric(board):
            numeric_board = np.zeros((8, 8), dtype=np.float32)
            for i in range(8):
                for j in range(8):
                    piece = board[i, j]
                    if piece is None:
                        numeric_board[i, j] = 0
                    elif piece.color == 'black':
                        numeric_board[i, j] = -1
                    else:
                        numeric_board[i, j] = 1
            return numeric_board

        state = board_to_numeric(self.board.board).flatten()

        actions = []
        for i in range(8):
            for j in range(8):
                piece = self.board.board[i, j]
                if piece and piece.color == 'black':
                    possible_moves = piece.get_possible_moves(self.board.board, (i, j))
                    if possible_moves:
                        for move in possible_moves:
                            actions.append(((i, j), move))

        num_actions = len(actions)
        policy_net = PolicyNetwork(num_actions)
        value_net = ValueNetwork()
        optimizer = optim.Adam(list(policy_net.parameters()) + list(value_net.parameters()), lr=0.000001)
        gamma = 0.99

        if actions:
            action_index, value = choose_action(state, actions, policy_net, value_net)

            action = actions[action_index]
            if action:
                (i, j), move = action


                self.board.board[move[0], move[1]] = self.board.board[i, j]
                self.board.board[i, j] = None

                reward = self.evaluate_board()
                next_state = board_to_numeric(self.board.board).flatten()
                next_actions = []

                for x in range(8):
                    for y in range(8):
                        piece = self.board.board[x, y]
                        if piece and piece.color == 'black':
                            possible_moves = piece.get_possible_moves(self.board.board, (x, y))
                            if possible_moves:
                                for next_move in possible_moves:
                                    next_actions.append(((x, y), next_move))

                update_policy_and_value_net(policy_net, value_net, optimizer, state, action_index, reward, next_state, next_actions, gamma)

                self.board.move_piece((i, j), move)
                self.switch_turn()
                self.turn_start_time = pygame.time.get_ticks()

                if isinstance(self.board.board[move[0], move[1]], Pawn):
                    if move[0] == 0 or move[0] == 7:
                        self.board.board[move[0], move[1]].promote(self.board.board, move)

                self.board.computer_move_start = (i, j)
                self.board.computer_move_end = move
                start_x = j * 80 + 140
                start_y = i * 80 + 140
                end_x = move[1] * 80 + 140
                end_y = move[0] * 80 + 140
                self.screen.fill((128, 128, 128))  
                self.screen.blit(self.background, (100, 100))  
                self.draw_board([(255, 255, 255), (0, 0, 0)], pygame.font.SysFont(None, 24))  
                self.draw_dashed_line(self.screen, (0, 0, 255), (start_x, start_y), (end_x, end_y), 5)
                angle = np.arctan2(end_y - start_y, end_x - start_x)
                arrow_size = 10
                arrow_points = [
                    (end_x, end_y),
                    (end_x - arrow_size * np.cos(angle - np.pi / 6), end_y - arrow_size * np.sin(angle - np.pi / 6)),
                    (end_x - arrow_size * np.cos(angle + np.pi / 6), end_y - arrow_size * np.sin(angle + np.pi / 6))
                ]
                pygame.draw.polygon(self.screen, (0, 0, 255), arrow_points)
                pygame.display.flip()
    def evaluate_board(self):
        piece_values = {
            King: 1000,
            Queen: 9,
            Rook: 5,
            Bishop: 3,
            Knight: 3,
            Pawn: 1
        }
        score = 0
        for row in self.board.board:
            for piece in row:
                if piece:
                    value = piece_values[type(piece)]
                    if piece.color == 'black':
                        score += value
                    else:
                        score -= value
        return score