import pygame
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
import pickle

from Pieces.King import King
from Pieces.Rook import Rook
from Pieces.Bishop import Bishop
from Pieces.Knight import Knight
from Pieces.Pawn import Pawn
from Pieces.Queen import Queen
from Board import Board
from PieceImages import PieceImages
from BoardDisplay import BoardDisplay
from TimerDisplay import TimerDisplay
from KillLogDisplay import KillLogDisplay
from MenuButtonDisplay import MenuButtonDisplay

class Menu:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((1000, 1000))
        pygame.display.set_caption("Chess")
        self.clock = pygame.time.Clock()

        self.title_font = pygame.font.SysFont(None, 74)
        self.title_text = self.title_font.render("chess in python", True, (255, 255, 255))
        self.title_rect = self.title_text.get_rect(center=(self.screen.get_width() // 2, 100))

        self.human_img = pygame.image.load("assets/Buttons/Human.png").convert_alpha()
        self.computer_img = pygame.image.load("assets/Buttons/Computer.png").convert_alpha()
        self.computer_vs_computer_img = pygame.image.load("assets/Buttons/Mirror.png").convert_alpha()
        self.exit_img = pygame.image.load("assets/Buttons/Exit.png").convert_alpha()

        self.human_img_with_bg = self.create_button_with_bg(self.human_img)
        self.computer_img_with_bg = self.create_button_with_bg(self.computer_img)
        self.mirror_img_with_bg = self.create_button_with_bg(self.computer_vs_computer_img)
        self.exit_img_with_bg = self.create_button_with_bg(self.exit_img)

        self.options = [self.human_img_with_bg, self.computer_img_with_bg, self.mirror_img_with_bg, self.exit_img_with_bg]
        self.option_texts = ["human vs human", "human vs computer", "computer vs computer", "exit"]
        self.option_rects = [pygame.Rect(100 + i * (option.get_width() + 100), 400, option.get_width(), option.get_height()) for i, option in enumerate(self.options)]
        self.selected_option = 0
        self.blink = True
        self.blink_timer = 0

    def create_button_with_bg(self, img):
        img_with_bg = pygame.Surface((img.get_width() + 20, img.get_height() + 20), pygame.SRCALPHA)
        img_with_bg.fill((255, 255, 255))
        img_with_bg.blit(img, (10, 10))
        return img_with_bg

    def run(self):
        while True:
            self.screen.fill((0, 0, 0))
            self.screen.blit(self.title_text, self.title_rect)
            for i, option in enumerate(self.options):
                self.screen.blit(option, self.option_rects[i].topleft)
                if i == self.selected_option and self.blink:
                    pygame.draw.rect(self.screen, (0, 255, 0), self.option_rects[i], 5)
                text_surface = pygame.font.SysFont(None, 26).render(self.option_texts[i], True, (255, 255, 255))
                text_rect = text_surface.get_rect(center=(self.option_rects[i].centerx, self.option_rects[i].bottom + 30))
                self.screen.blit(text_surface, text_rect)

            pygame.display.flip()

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                elif event.type == pygame.KEYDOWN:
                    self.handle_keydown(event)
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    self.handle_mousebuttondown()

            self.blink_timer += 1
            if self.blink_timer % 5 == 0:
                self.blink = not self.blink

            self.clock.tick(30)

    def handle_keydown(self, event):
        if event.key == pygame.K_LEFT:
            self.selected_option = (self.selected_option - 1) % len(self.options)
        elif event.key == pygame.K_RIGHT:
            self.selected_option = (self.selected_option + 1) % len(self.options)
        elif event.key == pygame.K_RETURN:
            self.execute_selected_option()

    def handle_mousebuttondown(self):
        mouse_x, mouse_y = pygame.mouse.get_pos()
        for i, option_rect in enumerate(self.option_rects):
            if option_rect.collidepoint(mouse_x, mouse_y):
                self.selected_option = i
                self.execute_selected_option()

    def execute_selected_option(self):
        if self.selected_option == 0:
            game = Game()
            game.play()
        elif self.selected_option == 1:
            game = Game(play_with_computer=True)
            game.play()
        elif self.selected_option == 2:
            game = Game(computer_vs_computer=True)
            game.play()
        elif self.selected_option == 3:
            pygame.quit()
            sys.exit()


class Game:
    def __init__(self, play_with_computer=False, computer_vs_computer=False):
        pygame.init()
        self.screen = pygame.display.set_mode((1000, 1000))
        self.board = Board(self.screen)
        self.current_turn = 'white'
        self.selected_piece = None
        self.selected_position = None
        self.turn_time_limit = 60
        self.turn_start_time = pygame.time.get_ticks()
        self.play_with_computer = play_with_computer
        self.computer_vs_computer = computer_vs_computer
        self.kill_log = []
        self.replay_buffer = ReplayBuffer(10000)  

        pygame.display.set_caption("White turn")
        self.clock = pygame.time.Clock()
        self.background = pygame.image.load("assets/Background.png").convert()
        self.background = pygame.transform.scale(self.background, (640, 640))  

        self.piece_images = PieceImages.load_images()

        self.menu_button = pygame.image.load("assets/Buttons/Menu.png").convert_alpha()
        self.menu_button = pygame.transform.scale(self.menu_button, (50, 50))

        self.board_display = BoardDisplay(self.screen, self.background, self.piece_images)
        self.timer_display = TimerDisplay(self.screen, self.turn_time_limit)
        self.kill_log_display = KillLogDisplay(self.screen, self.piece_images)
        self.menu_button_display = MenuButtonDisplay(self.screen, self.menu_button)

    def play(self):
        while not self.is_game_over():
            self.handle_events()
            self.board_display.display_board(self.board, self.selected_piece, self.selected_position, self.current_turn)
            self.timer_display.display_timer(self.turn_start_time, self.current_turn, self.board)
            self.menu_button_display.display_menu_button()
            self.kill_log_display.display_kill_log(self.kill_log)
            pygame.display.flip()
            self.clock.tick(30)
                
            if self.current_turn == 'black' or self.computer_vs_computer:
                if self.play_with_computer or self.computer_vs_computer:
                    self.computer_move()
                    self.turn_start_time = pygame.time.get_ticks()

        self.board_display.display_board(self.board, self.selected_piece, self.selected_position, self.current_turn)
        pygame.display.flip()
        self.display_game_over()

    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == pygame.MOUSEBUTTONDOWN:
                self.handle_mousebuttondown()

    def handle_mousebuttondown(self):
        mouse_position = pygame.mouse.get_pos()
        if 900 <= mouse_position[0] <= 950 and 20 <= mouse_position[1] <= 70:
            Menu().run()
            return
        row, col = (mouse_position[1] - 100) // 80, (mouse_position[0] - 100) // 80
        if self.selected_piece:
            possible_moves = self.selected_piece.get_possible_moves(self.board.board, self.selected_position)
            if possible_moves and (row, col) in possible_moves:
                if self.board.board[row, col] is not None:
                    self.kill_log.append((self.selected_piece, self.board.board[row, col]))
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
        if self.current_turn == 'white':
            self.current_turn = 'black'
        elif self.current_turn == 'black':
            self.current_turn = 'white'
            
        pygame.display.set_caption(f"{self.current_turn.capitalize()} turn")

        for row in self.board.board:
            for piece in row:
                if isinstance(piece, Pawn):
                    self.en_passant_target = False

    def is_game_over(self):
        return self.board.is_checkmate(self.current_turn) or self.board.is_stalemate(self.current_turn) or not self.king_exists(self.current_turn)

    def king_exists(self, color):
        for row in self.board.board:
            for piece in row:
                if isinstance(piece, King) and piece.color == color:
                    return True
        return False

    def display_game_over(self):
        font_title = pygame.font.SysFont(None, 74)
        font_subtitle = pygame.font.SysFont(None, 50)
        
        if self.board.is_checkmate(self.current_turn):
            title_text = font_title.render(f"{self.current_turn.capitalize()} loses", True, (255, 0, 0))
            subtitle_text = font_subtitle.render("Checkmate", True, (255, 255, 255))
        elif not self.king_exists(self.current_turn):
            title_text = font_title.render(f"{self.current_turn.capitalize()} loses", True, (255, 0, 0))
            subtitle_text = font_subtitle.render("King captured", True, (255, 255, 255))
        else:
            title_text = font_title.render("Stalemate", True, (255, 255, 0))
            subtitle_text = None

        modal_surface = pygame.Surface((400, 200), pygame.SRCALPHA)
        modal_surface.fill((0, 0, 0, 128))  
        modal_surface.blit(title_text, (50, 50))
        if subtitle_text:
            modal_surface.blit(subtitle_text, (50, 120))

        self.screen.blit(modal_surface, (300, 400))
        pygame.display.flip()

        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                elif event.type == pygame.KEYDOWN or event.type == pygame.MOUSEBUTTONDOWN:
                    return

    def computer_move(self):
        state = self.board_to_numeric(self.board.board).flatten()

        actions = []
        for row in range(8):
            for col in range(8):
                piece = self.board.board[row, col]
                if piece and piece.color == self.current_turn:
                    possible_moves = piece.get_possible_moves(self.board.board, (row, col))
                    if possible_moves:
                        for move in possible_moves:
                            actions.append(((row, col), move))

        num_actions = len(actions)
        policy_net = PolicyNetwork(num_actions)
        value_net = ValueNetwork()
        optimizer = optim.Adam(list(policy_net.parameters()) + list(value_net.parameters()), lr=0.000001)
        gamma = 0.99  

        if actions:
            mcts = MCTS(policy_net, value_net)
            root = MCTSNode(state)
            root.expand(actions)

            for _ in range(100): # 하...  
                node = root
                while not node.is_leaf():
                    node = mcts.best_child(node)
                if node.visits > 0:
                    node.expand(actions)
                reward = self.evaluate_board()
                while node is not None:
                    node.update(reward)
                    node = node.parent

            best_action = mcts.best_action(root)
            if best_action:
                (start_row, start_col), move = best_action

                if self.board.board[move[0], move[1]] is not None:
                    self.kill_log.append((self.board.board[start_row, start_col], self.board.board[move[0], move[1]]))

                self.board.board[move[0], move[1]] = self.board.board[start_row, start_col]
                self.board.board[start_row, start_col] = None

                reward = self.evaluate_board()
                next_state = self.board_to_numeric(self.board.board).flatten()

                next_actions = []
                for row in range(8):
                    for col in range(8):
                        piece = self.board.board[row, col]
                        if piece and piece.color == self.current_turn:
                            possible_moves = piece.get_possible_moves(self.board.board, (row, col))
                            if possible_moves:
                                for next_move in possible_moves:
                                    next_actions.append(((row, col), next_move))

                self.update_policy_and_value_net(policy_net, value_net, optimizer, state, actions.index(best_action), reward, next_state, next_actions, gamma)

                self.board.move_piece((start_row, start_col), move)
                self.switch_turn()
                self.turn_start_time = pygame.time.get_ticks()

                if isinstance(self.board.board[move[0], move[1]], Pawn):
                    if move[0] == 0 or move[0] == 7:
                        self.board.board[move[0], move[1]].promote(self.board.board, move)

                self.board.computer_move_start = (start_row, start_col)
                self.board.computer_move_end = move
                start_x = start_col * 80 + 140
                start_y = start_row * 80 + 140
                end_x = move[1] * 80 + 140
                end_y = move[0] * 80 + 140
                self.screen.fill((128, 128, 128))  
                self.screen.blit(self.background, (100, 100))  
                self.board_display.draw_board(self.board, [(255, 255, 255), (0, 0, 0)], pygame.font.SysFont(None, 24))  
                self.board_display.draw_dashed_line(self.screen, (0, 0, 255), (start_x, start_y), (end_x, end_y), 5)
                angle = np.arctan2(end_y - start_y, end_x - start_x)
                arrow_size = 10
                arrow_points = [
                    (end_x, end_y),
                    (end_x - arrow_size * np.cos(angle - np.pi / 6), end_y - arrow_size * np.sin(angle - np.pi / 6)),
                    (end_x - arrow_size * np.cos(angle + np.pi / 6), end_y - arrow_size * np.sin(angle + np.pi / 6))
                ]
                pygame.draw.polygon(self.screen, (0, 0, 255), arrow_points)
                pygame.display.flip()

        torch.save(policy_net.state_dict(), 'policy_net.pth')
        torch.save(value_net.state_dict(), 'value_net.pth')

        with open('replay_buffer.pkl', 'wb') as f:
            pickle.dump(self.replay_buffer, f)

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

    def choose_action(self, state, actions, policy_net, value_net, epsilon=0.1):
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        
        with torch.no_grad():
            policy = policy_net(state_tensor).squeeze(0).numpy()
            value = value_net(state_tensor).item()

        action_probabilities = np.zeros(len(actions))

        if len(policy) == len(actions):
            action_probabilities = np.exp(policy) / np.sum(np.exp(policy)) 
        else:
            for idx, action in enumerate(actions):
                action_probabilities[idx] = policy[idx]

            if np.sum(action_probabilities) > 0:
                action_probabilities /= np.sum(action_probabilities)
            else:
                action_probabilities = np.ones(len(actions)) / len(actions)  

        if np.random.rand() < epsilon:
            return np.random.choice(len(actions)), None  
        else:
            return np.random.choice(len(actions), p=action_probabilities), value 

    def update_policy_and_value_net(self, policy_net, value_net, optimizer, state, action, reward, next_state, next_actions, gamma):
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        next_state_tensor = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0)
        action_tensor = torch.tensor([action], dtype=torch.int64)
        reward_tensor = torch.tensor([reward], dtype=torch.float32)

        with torch.no_grad():
            next_policy = policy_net(next_state_tensor).squeeze(0)
            next_value = value_net(next_state_tensor).item()
            next_action_probabilities = next_policy / torch.sum(next_policy)
            next_value = torch.sum(next_action_probabilities * next_value)

        policy = policy_net(state_tensor).squeeze(0)
        value = value_net(state_tensor).item()
        action_probability = policy[action_tensor]
        advantage = reward_tensor + gamma * next_value - value
        policy_loss = -torch.log(action_probability) * advantage
        value_loss = advantage.pow(2)

        optimizer.zero_grad()
        (policy_loss + value_loss).backward()
        optimizer.step()

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
        
        for i, child_node in enumerate(node.children):
            policy_score = policy[i] if i < len(policy) else 0
            score = child_node.value + exploration_constant * policy_score * np.sqrt(np.log(node.visits + 1) / (child_node.visits + 1))
            if score > best_score:
                best_score = score
                best_child_node = child_node

        return best_child_node if best_child_node is not None else node 

    def best_action(self, root):
        return max(root.children, key=lambda child_node: child_node.visits).action

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
        x = x.view(-1, 64)  
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

if __name__ == "__main__":
    while True:
        Menu().run()