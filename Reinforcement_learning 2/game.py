import math
import random
import numpy as np
import pygame
import sys

class GameManager:
    def __init__(self):
        pygame.init()

        self.WIDTH, self.HEIGHT = 800, 600
        self.BLACK = (0, 0, 0)
        self.WHITE = (255, 255, 255)
        self.PADDLE_SPEED = 10
        self.BALL_SPEED = 5

        self.screen = pygame.display.set_mode((self.WIDTH, self.HEIGHT))
        pygame.display.set_caption("Pong")
        
        self.paddle2 = self.Paddle(20, self)
        self.recta = self.Rectangle(self.WIDTH - 5, self)
        self.ball = self.Ball(self)

        self.all_sprites = pygame.sprite.Group()
        self.all_sprites.add(self.recta, self.paddle2, self.ball)

        self.clock = pygame.time.Clock()

        self.reward = 0
        self.score = 0

    def reset(self):
        self.reward = 0
        self.score = 0
        self.ball.reset()

    def play_step(self, action) -> tuple:
        self.reward = 0
        game_over = False
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
        
        if np.array_equal(action, [1, 0, 0]):
            pass
        elif np.array_equal(action, [0, 1, 0]):
            self.paddle2.move(-self.PADDLE_SPEED)
        else:
            self.paddle2.move(self.PADDLE_SPEED)
        self.ball.move()

        if self.ball.rect.left <= 0:
            self.reward = -15
            game_over = True
        elif self.ball.rect.right >= self.WIDTH:
            game_over = True
        
        if self.ball.rect.colliderect(self.recta):
            self.ball.speed_x *= -1
            self.ball.accelerate()
            
        elif self.ball.rect.colliderect(self.paddle2):
            self.ball.speed_x *= -1
            self.ball.accelerate()
            self.reward = 10
            self.score += 1


        self.screen.fill(self.BLACK)
        self.all_sprites.draw(self.screen)
        pygame.display.flip()
        self.clock.tick(120)

        return self.reward, game_over, self.score


    class Paddle(pygame.sprite.Sprite):
        def __init__(self, x, game_manager):
            super().__init__()
            self.game_manager = game_manager
            self.image = pygame.Surface((10, 100))
            self.image.fill(self.game_manager.WHITE)
            self.rect = self.image.get_rect()
            self.rect.center = (x, self.game_manager.HEIGHT // 2)

        def move(self, dy):
            if 0 < self.rect.top + dy < self.game_manager.HEIGHT - 100:
                self.rect.y += dy
    
    class Rectangle(pygame.sprite.Sprite):
        def __init__(self, x, game_manager):
            super().__init__()
            self.game_manager = game_manager
            self.image = pygame.Surface((10, 800))
            self.image.fill(self.game_manager.WHITE)
            self.rect = self.image.get_rect()
            self.rect.center = (x, self.game_manager.HEIGHT // 2)

    class Ball(pygame.sprite.Sprite):
        def __init__(self, game_manager):
            super().__init__()
            self.game_manager = game_manager
            self.image = pygame.Surface((10, 10))
            self.image.fill(self.game_manager.WHITE)
            self.rect = self.image.get_rect()
            self.rect.center = (self.game_manager.WIDTH // 2, self.game_manager.HEIGHT // 2)
            self.speed_x = self.game_manager.BALL_SPEED
            self.speed_y = self.game_manager.BALL_SPEED

        def reset(self):
            self.rect.center = (self.game_manager.WIDTH // 2, self.game_manager.HEIGHT // 2)
            self.speed_x = self.game_manager.BALL_SPEED
            self.speed_y = self.game_manager.BALL_SPEED * random.uniform(-1, 1)
            if self.speed_y > 1 and self.speed_y < 1:
                self.speed_y = self.game_manager.BALL_SPEED
        
        def move(self):
            self.rect.x += self.speed_x
            self.rect.y += self.speed_y

            if self.rect.top <= 0 or self.rect.bottom >= self.game_manager.HEIGHT:
                self.speed_y *= -1
        
        def accelerate(self):
            #self.speed_x += 0.5 if self.speed_x > 0 else -0.5
            pass
