import pygame

pygame.init()
screen = pygame.display.set_mode((600, 400))
img = pygame.Surface((103, 177))
img.fill((200, 200, 200))
rect = img.get_rect(topleft=(100, 100))
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
    screen.fill((50, 180, 50))
    screen.blit(img, rect)
    pygame.draw.rect(screen, (255, 0, 0), rect, width=2)
    pygame.display.flip()
pygame.quit()
