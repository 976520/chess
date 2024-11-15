import pygame

class BackgroundImage:
    @staticmethod
    def load_image():
        return pygame.image.load("assets/images/Background.png").convert()

