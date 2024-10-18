import unittest
from main import Game

class TestGameInitialization(unittest.TestCase):
    def setUp(self):
        self.game = Game()

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

if __name__ == '__main__':
    unittest.main()