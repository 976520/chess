import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from features.game.Game import Game

if __name__ == "__main__":
    from pages.Menu import Menu
    
    game = Game()
    game.play()
    Menu().run()
