import pygame

class TimerDisplay:
    def __init__(self, screen, turn_time_limit):
        self.screen = screen
        self.turn_time_limit = turn_time_limit

    def display_timer(self, turn_start_time, current_turn, board):
        elapsed_time = (pygame.time.get_ticks() - turn_start_time) / 1000
        remaining_time = max(0, self.turn_time_limit - elapsed_time)
        if remaining_time == 0:
            return True 

        timer_width = int((remaining_time / self.turn_time_limit) * 640)
        
        if current_turn == 'white': 
            if board.is_in_check('white'):
                timer_color = (255, 0, 0)
            else:
                timer_color = (0, 0, 255)
            pygame.draw.rect(self.screen, timer_color, pygame.Rect(100, 750, timer_width, 5)) 
        elif current_turn == 'black':
            if board.is_in_check('black'):
                timer_color = (255, 0, 0)
            else:
                timer_color = (0, 0, 255)
            pygame.draw.rect(self.screen, timer_color, pygame.Rect(100, 85, timer_width, 5))  

        pygame.display.update()
        return False  