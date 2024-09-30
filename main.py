import pygame
import sys
from Game import Game

def main_menu():
    pygame.init()
    screen = pygame.display.set_mode((1000, 1000))
    pygame.display.set_caption("Chess Main Menu")
    clock = pygame.time.Clock()

    human_img = pygame.image.load("assets/Human.png").convert_alpha()
    computer_img = pygame.image.load("assets/Computer.png").convert_alpha()
    exit_img = pygame.image.load("assets/Exit.png").convert_alpha()

    human_img_with_bg = pygame.Surface((human_img.get_width() + 20, human_img.get_height() + 20), pygame.SRCALPHA)
    human_img_with_bg.fill((255, 255, 255))
    human_img_with_bg.blit(human_img, (10, 10))

    computer_img_with_bg = pygame.Surface((computer_img.get_width() + 20, computer_img.get_height() + 20), pygame.SRCALPHA)
    computer_img_with_bg.fill((255, 255, 255))
    computer_img_with_bg.blit(computer_img, (10, 10))

    exit_img_with_bg = pygame.Surface((exit_img.get_width() + 20, exit_img.get_height() + 20), pygame.SRCALPHA)
    exit_img_with_bg.fill((255, 255, 255))
    exit_img_with_bg.blit(exit_img, (10, 10))

    options = [human_img_with_bg, computer_img_with_bg, exit_img_with_bg]
    option_texts = ["play with human(local)", "play with computer", "exit"]
    selected_option = 0
    blink = True
    blink_timer = 0

    while True:
        screen.fill((0, 0, 0))
        for i, option in enumerate(options):
            x_position = 100 + i * (option.get_width() + 230)  
            screen.blit(option, (x_position, 400))
            if i == selected_option and blink:
                pygame.draw.rect(screen, (0, 255, 0), (x_position, 400, option.get_width(), option.get_height()), 5)
            text_surface = pygame.font.SysFont(None, 26).render(option_texts[i], True, (255, 255, 255))
            text_rect = text_surface.get_rect(center=(x_position + option.get_width() // 2, 400 + option.get_height() + 30))
            screen.blit(text_surface, text_rect)

        pygame.display.flip()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT:
                    selected_option = (selected_option - 1) % len(options)
                elif event.key == pygame.K_RIGHT:
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
                    option_rect = pygame.Rect(100 + i * (option.get_width() + 300), 400, option.get_width(), option.get_height()) 
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

        blink_timer += 1
        if blink_timer % 5 == 0:
            blink = not blink

        clock.tick(30)

if __name__ == "__main__":
    while True:
        main_menu()