import pygame
import sys

class Menu:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((1000, 1000))
        pygame.display.set_caption("Chess")
        self.clock = pygame.time.Clock()

        self.title_font = pygame.font.SysFont(None, 74)
        self.title_text = self.title_font.render("chess in python", True, (255, 255, 255))
        self.title_rect = self.title_text.get_rect(center=(self.screen.get_width() // 2, 100))

        self.options = [self.create_button_with_bg(pygame.image.load("assets/images/Buttons/Human.png").convert_alpha()), self.create_button_with_bg(pygame.image.load("assets/images/Buttons/Computer.png").convert_alpha()), self.create_button_with_bg(pygame.image.load("assets/images/Buttons/Mirror.png").convert_alpha()), self.create_button_with_bg(pygame.image.load("assets/images/Buttons/Exit.png").convert_alpha())]
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
                if i == self.selected_option:
                    if self.blink:
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
        from features.game.Game import Game

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
