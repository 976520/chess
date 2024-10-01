import pygame
import sys
from Pieces.King import King

class GameOverDisplay:
    def __init__(self, screen):
        self.screen = screen

    def display_game_over(self, board, current_turn):
        font_title = pygame.font.SysFont(None, 74)
        font_subtitle = pygame.font.SysFont(None, 50)
        
        if board.is_checkmate(current_turn):
            title_text = font_title.render(f"{current_turn.capitalize()} loses", True, (255, 0, 0))
            subtitle_text = font_subtitle.render("Checkmate", True, (255, 255, 255))
        elif not self.king_exists(board, current_turn):
            title_text = font_title.render(f"{current_turn.capitalize()} loses", True, (255, 0, 0))
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

    def king_exists(self, board, color):
        for row in board.board:
            for piece in row:
                if isinstance(piece, King) and piece.color == color:
                    return True
        return False
