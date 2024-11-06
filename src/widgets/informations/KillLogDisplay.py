import numpy as np
import pygame

class KillLogDisplay:
    def __init__(self, screen, piece_images):
        self.screen = screen
        self.piece_images = piece_images

    def display_kill_log(self, kill_log):
        x_offset = 750
        y_offset = 100 + (min(len(kill_log), 6) - 1) * 100 
        for killer_piece, killed_piece in kill_log[-5:]:  
            killer_image = self.piece_images[type(killer_piece).__name__ + '_' + killer_piece.color[0]]
            killed_image = self.piece_images[type(killed_piece).__name__ + '_' + killed_piece.color[0]]
            self.screen.blit(killer_image, (x_offset, y_offset))
            start_pos = (x_offset + 80, y_offset + 40)
            end_pos = (x_offset + 110, y_offset + 40)  
            angle = np.arctan2(end_pos[1] - start_pos[1], end_pos[0] - start_pos[0])
            arrow_size = 10
            arrow_points = [
                (end_pos[0] + 10, end_pos[1]), 
                (end_pos[0] - arrow_size * np.cos(angle - np.pi / 4), end_pos[1] - arrow_size * np.sin(angle - np.pi / 4)),
                (end_pos[0] - arrow_size * np.cos(angle + np.pi / 4), end_pos[1] - arrow_size * np.sin(angle + np.pi / 4))
            ]
            pygame.draw.line(self.screen, (255, 0, 0), start_pos, end_pos, 5)
            pygame.draw.polygon(self.screen, (255, 0, 0), arrow_points)
            self.screen.blit(killed_image, (x_offset + 120, y_offset))
            y_offset -= 100  
