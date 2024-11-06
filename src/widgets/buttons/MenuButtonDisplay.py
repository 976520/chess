class MenuButtonDisplay:
    def __init__(self, screen, menu_button):
        self.screen = screen
        self.menu_button = menu_button

    def display_menu_button(self):
        self.screen.blit(self.menu_button, (900, 20))