import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from features.game.Game import Game
from pages.Menu import Menu

if __name__ == "__main__":
    game = Game()
    game.play()
    Menu().run()
