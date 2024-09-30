import pygame

class PieceImages:
    @staticmethod
    def load_images():
        return {
            'Pawn_b': pygame.transform.scale(pygame.image.load("assets/Pieces_b/Pawn_b.png").convert_alpha(), (80, 80)),
            'Pawn_w': pygame.transform.scale(pygame.image.load("assets/Pieces_w/Pawn_w.png").convert_alpha(), (80, 80)),
            'Rook_b': pygame.transform.scale(pygame.image.load("assets/Pieces_b/Rook_b.png").convert_alpha(), (80, 80)),
            'Rook_w': pygame.transform.scale(pygame.image.load("assets/Pieces_w/Rook_w.png").convert_alpha(), (80, 80)),
            'Knight_b': pygame.transform.scale(pygame.image.load("assets/Pieces_b/Knight_b.png").convert_alpha(), (80, 80)),
            'Knight_w': pygame.transform.scale(pygame.image.load("assets/Pieces_w/Knight_w.png").convert_alpha(), (80, 80)),
            'Bishop_b': pygame.transform.scale(pygame.image.load("assets/Pieces_b/Bishop_b.png").convert_alpha(), (80, 80)),
            'Bishop_w': pygame.transform.scale(pygame.image.load("assets/Pieces_w/Bishop_w.png").convert_alpha(), (80, 80)),
            'Queen_b': pygame.transform.scale(pygame.image.load("assets/Pieces_b/Queen_b.png").convert_alpha(), (80, 80)),
            'Queen_w': pygame.transform.scale(pygame.image.load("assets/Pieces_w/Queen_w.png").convert_alpha(), (80, 80)),
            'King_b': pygame.transform.scale(pygame.image.load("assets/Pieces_b/King_b.png").convert_alpha(), (80, 80)),
            'King_w': pygame.transform.scale(pygame.image.load("assets/Pieces_w/King_w.png").convert_alpha(), (80, 80)),
        }
