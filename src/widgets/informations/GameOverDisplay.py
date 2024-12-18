import pygame
import sys
from widgets.pieces.King import King

class GameOverDisplay:
    def __init__(self, screen):
        self.screen = screen

    def display_game_over(self, board, current_turn):
        font_title = pygame.font.SysFont(None, 74)
        font_subtitle = pygame.font.SysFont(None, 50)
        
        if board.is_checkmate(current_turn):
            title_text = font_title.render(f"{current_turn.capitalize()} loses", True, (255, 0, 0))
            subtitle_text = font_subtitle.render("Checkmate", True, (255, 255, 255))
        elif board.king_exists(current_turn):
            if current_turn == 'white':
                title_text = font_title.render(f"Black loses", True, (255, 0, 0))
            elif current_turn == 'black':
                title_text = font_title.render(f"White loses", True, (255, 0, 0))
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

        waiting_for_event = True
        while waiting_for_event:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                elif event.type == pygame.MOUSEBUTTONUP:
                    waiting_for_event = False
                    from pages.Menu import Menu
                    Menu().run()
                    break
                    
                

    def king_exists(self, board, color):
        for row in board.board:
            for piece in row:
                if isinstance(piece, King):
                    if piece.color == color:
                        return True
        return False