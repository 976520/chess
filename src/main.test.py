import unittest
from main import Game
import pygame

class TestGameInitialization(unittest.TestCase):
    def setUp(self):
        pygame.init()
        self.game = Game()
        self.screen = pygame.display.get_surface()
        
    def test_default_initialization(self):
        self.assertEqual(self.game.current_turn, 'white')
        self.assertIsNone(self.game.selected_piece)
        self.assertIsNone(self.game.selected_position)
        self.assertEqual(self.game.turn_time_limit, 60)
        self.assertEqual(len(self.game.kill_log), 0)
        self.assertEqual(len(self.game.replay_buffer.buffer), 0)
        self.assertFalse(self.game.play_with_computer)
        self.assertFalse(self.game.computer_vs_computer)
        
    def test_initial_turn(self):
        game = Game()
        self.assertEqual(game.current_turn, 'white')
        
    def test_handle_mousebuttondown(self):
        pygame.mouse.set_pos(150, 150) 
        pygame.event.post(pygame.event.Event(pygame.MOUSEBUTTONDOWN, pos=(150, 150)))

        self.game.handle_events()
        self.assertEqual(self.game.selected_piece, self.game.board.board[1, 1])
        self.assertEqual(self.game.selected_position, (1, 1))

        
if __name__ == '__main__':
    unittest.main()