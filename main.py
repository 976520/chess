import pygame
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import time 

from King import King
from Rook import Rook
from Bishop import Bishop
from Knight import Knight
from Pawn import Pawn
from Queen import Queen

class Board:
    def __init__(self, screen):
        self.board = self.create_initial_board()
        self.en_passant_target = None
        self.screen = screen

    def create_initial_board(self):
        board = np.full((8, 8), None)
        for i in range(8):
            board[1, i] = Pawn('black')
            board[6, i] = Pawn('white')
        board[0, [0, 7]] = Rook('black')
        board[0, [1, 6]] = Knight('black')
        board[0, [2, 5]] = Bishop('black')
        board[0, 3] = Queen('black')
        board[0, 4] = King('black')
        board[7, [0, 7]] = Rook('white')
        board[7, [1, 6]] = Knight('white')
        board[7, [2, 5]] = Bishop('white')
        board[7, 3] = Queen('white')
        board[7, 4] = King('white')
        return board

    def move_piece(self, start_pos, end_pos):
        piece = self.board[start_pos[0], start_pos[1]]
        if piece:
            possible_moves = piece.get_possible_moves(self.board, start_pos)
            if possible_moves and end_pos in possible_moves:
                target_piece = self.board[end_pos[0], end_pos[1]]
                if isinstance(target_piece, King):
                    print(f"{piece.color.capitalize()} wins")
                    pygame.quit()
                    sys.exit()

            if isinstance(piece, King):
                if abs(start_pos[1] - end_pos[1]) == 2:
                    if end_pos[1] == 6: 
                        self.board[start_pos[0], 5] = self.board[start_pos[0], 7]
                        self.board[start_pos[0], 7] = None
                    elif end_pos[1] == 2: 
                        self.board[start_pos[0], 3] = self.board[start_pos[0], 0]
                        self.board[start_pos[0], 0] = None

            if isinstance(piece, Pawn):
                if end_pos == self.en_passant_target:
                    self.board[start_pos[0], end_pos[1]] = None

            self.board[end_pos[0], end_pos[1]] = piece
            self.board[start_pos[0], start_pos[1]] = None

            if isinstance(piece, Pawn):
                if abs(start_pos[0] - end_pos[0]) == 2:
                    self.en_passant_target = (start_pos[0] + (end_pos[0] - start_pos[0]) // 2, start_pos[1])
                    piece.en_passant_target = True
                else:
                    self.en_passant_target = None

            piece.has_moved = True
                
            if isinstance(piece, Pawn):
                if end_pos[0] == 0 or end_pos[0] == 7:
                    piece.promote(self.board, end_pos)

    def is_check(self, color):
        king_pos = None
        for i in range(8):
            for j in range(8):
                piece = self.board[i, j]
                if piece:
                    if isinstance(piece, King):
                        if piece.color == color:
                            king_pos = (i, j)
                            break
            if king_pos:
                break

        for i in range(8):
            for j in range(8):
                piece = self.board[i, j]
                if piece and piece.color != color:
                    possible_moves = piece.get_possible_moves(self.board, (i, j))
                    if possible_moves and king_pos in possible_moves:
                        return True
        return False

    def is_checkmate(self, color):
        if not self.is_check(color):
            return False

        for i in range(8):
            for j in range(8):
                piece = self.board[i, j]
                if piece and piece.color == color:
                    possible_moves = piece.get_possible_moves(self.board, (i, j))
                    if possible_moves:
                        for move in possible_moves:
                            original_piece = self.board[move[0], move[1]]
                            self.board[move[0], move[1]] = piece
                            self.board[i, j] = None
                            if not self.is_check(color):
                                self.board[i, j] = piece
                                self.board[move[0], move[1]] = original_piece
                                return False
                            self.board[i, j] = piece
                            self.board[move[0], move[1]] = original_piece
        return True

    def is_stalemate(self, color):
        if self.is_check(color):
            return False

        for i in range(8):
            for j in range(8):
                piece = self.board[i, j]
                if piece and piece.color == color:
                    possible_moves = piece.get_possible_moves(self.board, (i, j))
                    if possible_moves:
                        for move in possible_moves:
                            original_piece = self.board[move[0], move[1]]
                            self.board[move[0], move[1]] = piece
                            self.board[i, j] = None
                            if not self.is_check(color):
                                self.board[i, j] = piece
                                self.board[move[0], move[1]] = original_piece
                                return False
                            self.board[i, j] = piece
                            self.board[move[0], move[1]] = original_piece
        return True

class Game:
    def __init__(self, play_with_computer=False):
        pygame.init()
        self.screen = pygame.display.set_mode((1000, 1000))
        self.board = Board(self.screen)
        self.current_turn = 'white'
        self.selected_piece = None
        self.selected_pos = None
        self.turn_time = 60
        self.start_time = pygame.time.get_ticks()
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

        self.display_board()
        pygame.display.flip()
        self.display_game_over()

    def display_board(self):
        self.screen.fill((128, 128, 128))
        self.screen.blit(self.background, (100, 100))  
        colors = [(255, 255, 255), (0, 0, 0)]
        font = pygame.font.SysFont(None, 24)
        mouse_pos = pygame.mouse.get_pos()
        mouse_row, mouse_col = (mouse_pos[1] - 100) // 80, (mouse_pos[0] - 100) // 80

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

        if (0 <= mouse_row < 8) and (0 <= mouse_col < 8):
            s = pygame.Surface((80, 80), pygame.SRCALPHA)
            s.fill((255, 255, 255, 128))
            self.screen.blit(s, (mouse_col * 80 + 100, mouse_row * 80 + 100))

        if self.selected_piece:
            pygame.draw.rect(self.screen, (0, 0, 225), pygame.Rect(self.selected_pos[1] * 80 + 100, self.selected_pos[0] * 80 + 100, 80, 80), 3)
            possible_moves = self.selected_piece.get_possible_moves(self.board.board, self.selected_pos)
            if possible_moves:
                for move in possible_moves:
                    if self.board.board[move[0], move[1]] is None:
                        pygame.draw.circle(self.screen, (0, 255, 0), (move[1] * 80 + 140, move[0] * 80 + 140), 10)
                    else:
                        pygame.draw.circle(self.screen, (255, 0, 0), (move[1] * 80 + 140, move[0] * 80 + 140), 10)

        if self.board.is_check(self.current_turn):
            king_pos = None
            for i in range(8):
                for j in range(8):
                    piece = self.board.board[i, j]
                    if piece and isinstance(piece, King):
                        if piece.color == self.current_turn:
                            king_pos = (i, j)
                            break
                if king_pos:
                    break

            for i in range(8):
                for j in range(8):
                    piece = self.board.board[i, j]
                    if piece:
                        if piece.color != self.current_turn:
                            possible_moves = piece.get_possible_moves(self.board.board, (i, j))
                            if possible_moves and king_pos in possible_moves:
                                pygame.draw.rect(self.screen, (255, 0, 0), pygame.Rect(j * 80 + 100, i * 80 + 100, 80, 80), 3)

    def display_timer(self):
        elapsed_time = (pygame.time.get_ticks() - self.start_time) / 1000
        remaining_time = max(0, self.turn_time - elapsed_time)
        if remaining_time == 0:
            self.switch_turn() 
            self.start_time = pygame.time.get_ticks() 

        timer_width = int((remaining_time / self.turn_time) * 640)
        
        if self.current_turn == 'white': 
            if self.board.is_check('white'):
                timer_color = (255, 0, 0)
            else:
                timer_color = (0, 0, 255)
            pygame.draw.rect(self.screen, timer_color, pygame.Rect(100, 750, timer_width, 5)) 
        elif self.current_turn == 'black':
            if self.board.is_check('black'):
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
                    possible_moves = self.selected_piece.get_possible_moves(self.board.board, self.selected_pos)
                    if possible_moves and (row, col) in possible_moves:
                        self.board.move_piece(self.selected_pos, (row, col))
                        self.switch_turn()
                        self.start_time = pygame.time.get_ticks()
                    self.selected_piece = None
                    self.selected_pos = None
                else:
                    piece = self.board.board[row, col]
                    if piece:
                        if piece.color == self.current_turn:
                            self.selected_piece = piece
                            self.selected_pos = (row, col)

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
                


    def computer_move(self):
        def choose_action(state, actions, policy_net):
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0) 
            with torch.no_grad():
                policy = policy_net(state_tensor).squeeze(0).numpy()
            
            action_probs = np.zeros(len(actions))  
            
            if len(policy) != len(actions):
                for idx, action in enumerate(actions):
                    action_probs[idx] = policy[idx] 

            # action_probs를 정규화
            if np.sum(action_probs) > 0:
                action_probs /= np.sum(action_probs)  # 확률 정규화
            else:
                action_probs = np.ones(len(actions)) / len(actions)  

            return np.random.choice(len(actions), p=action_probs)



        def update_policy_net(policy_net, optimizer, state, action, reward, next_state, next_actions, gamma):
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            next_state_tensor = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0)
            action_tensor = torch.tensor([action], dtype=torch.int64)
            reward_tensor = torch.tensor([reward], dtype=torch.float32)

            with torch.no_grad():
                next_policy = policy_net(next_state_tensor).squeeze(0)
                next_action_probs = next_policy / torch.sum(next_policy)
                next_value = torch.sum(next_action_probs * reward_tensor)

            policy = policy_net(state_tensor).squeeze(0)
            action_prob = policy[action_tensor]
            loss = -torch.log(action_prob) * (reward_tensor + gamma * next_value)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        class PolicyNetwork(nn.Module):
            def __init__(self, num_actions):
                super(PolicyNetwork, self).__init__()
                self.fc1 = nn.Linear(64, 128)
                self.fc2 = nn.Linear(128, 128)
                self.fc3 = nn.Linear(128, num_actions)  

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
        optimizer = optim.Adam(policy_net.parameters(), lr=0.001)  
        gamma = 0.99

        if actions:  
            action_index = choose_action(state, actions, policy_net)
            
            action = actions[action_index]
            if action:
                (i, j), move = action

                delay = np.random.uniform(0.5, 2.0)
                time.sleep(delay)

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

                update_policy_net(policy_net, optimizer, state, action_index, reward, next_state, next_actions, gamma)

                self.board.move_piece((i, j), move)
                self.switch_turn()
                self.start_time = pygame.time.get_ticks()

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

def main_menu():
    pygame.init()
    screen = pygame.display.set_mode((1000, 1000))
    pygame.display.set_caption("Chess Main Menu")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont(None, 74)
    options = ["1. play with human (local)", "2. play with computer", "3. exit"]
    selected_option = 0

    while True:
        screen.fill((0, 0, 0))
        for i, option in enumerate(options):
            color = (255, 0, 0) if i == selected_option else (255, 255, 255)
            text = font.render(option, True, color)
            screen.blit(text, (100, 200 + i * 100))

        pygame.display.flip()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP:
                    selected_option = (selected_option - 1) % len(options)
                elif event.key == pygame.K_DOWN:
                    selected_option = (selected_option + 1) % len(options)
                elif event.key == pygame.K_RETURN:
                    if selected_option == 0:
                        game = Game()
                        game.play()
                    elif selected_option == 1:
                        game = Game(play_with_computer=True)
                        game.play()
                    elif selected_option == 2:
                        pygame.quit()
                        sys.exit()
        clock.tick(30)

if __name__ == "__main__":
    while True:
        main_menu()