import pygame
import sys
from Game import Game

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
            elif event.type == pygame.MOUSEBUTTONDOWN:
                mouse_x, mouse_y = pygame.mouse.get_pos()
                for i, option in enumerate(options):
                    option_rect = pygame.Rect(100, 200 + i * 100, 800, 74)
                    if option_rect.collidepoint(mouse_x, mouse_y):
                        selected_option = i
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