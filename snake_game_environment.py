"""
NOTE : This code was initially downloaded from the YouTube Tutorial for a Basic Snake Game.
Some adaptations have been made to turn the environment into an AI environment instead.
The arial.ttf file is just used to import a Font into this PyGame, I believe.
"""

import pygame
import random
from enum import Enum
from collections import namedtuple
import numpy as np

# Initialising the PyGame environment
pygame.init()
font = pygame.font.Font("arial.ttf", 25)
# font = pygame.font.SysFont('arial', 25)


# Defining the different directions the agent could go in
class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4


# Defining a block of a snake
Point = namedtuple("Point", "x, y")

# rgb colors
WHITE = (255, 255, 255)
RED = (200, 0, 0)
BLUE1 = (0, 0, 255)
BLUE2 = (0, 100, 255)
BLACK = (0, 0, 0)

BLOCK_SIZE = 20
SPEED = 20


# Defining the SnakeGame as a class
class SnakeGameAI:

    def __init__(self, w=640, h=480):
        # Defining the height and width of the game
        self.w = w
        self.h = h
        # init display
        self.display = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption("Snake")
        self.clock = pygame.time.Clock()
        # Initialising the game state
        self.reset()

    # Code to reset the game state back to default
    def reset(self):
        # init game state
        self.direction = Direction.RIGHT

        self.head = Point(self.w / 2, self.h / 2)
        # Defining the snake object itself
        self.snake = [
            self.head,
            Point(self.head.x - BLOCK_SIZE, self.head.y),
            Point(self.head.x - (2 * BLOCK_SIZE), self.head.y),
        ]
        # Initialising Score and Food for the snake
        self.score = 0
        self.food = None
        self._place_food()
        # Keeping count of the number of frame iterations
        self.frame_iteration = 0

    # Function to randomly place food on the board
    def _place_food(self):
        x = random.randint(0, (self.w - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
        y = random.randint(0, (self.h - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
        self.food = Point(x, y)
        if self.food in self.snake:
            self._place_food()

    # Defining the different actions that the snake can take
    def play_step(self, action):
        # Updating the frame iteration
        self.frame_iteration += 1

        # 1. collect user input
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        # 2. move
        self._move(action)  # update the head
        self.snake.insert(0, self.head)

        # Initialising the reward for the agent
        reward = 0

        # 3. check if game over
        game_over = False
        # This if statement also implements code to stop if the snake makes too many actions without eating fruit
        # If the snake makes >100 per fruit, its game over
        # This incentives the agent to prioritise moving towards fruit, rather than moving aimlessly
        if self.is_collision() or self.frame_iteration > 100 * len(self.snake):
            game_over = True
            reward = -10  # Setting award to -10 as lost
            return reward, game_over, self.score

        # 4. place new food or just move
        if self.head == self.food:
            reward = 10
            self.score += 1
            self._place_food()
        else:
            self.snake.pop()

        # 5. update ui and clock
        self._update_ui()
        self.clock.tick(SPEED)
        # 6. return game over and score
        return reward, game_over, self.score

    # Function to check whether the snake has collided with a wall or itself
    # pt stands for Point, initially set to None
    def is_collision(self, pt=None):
        if pt is None:
            pt = self.head
        # hits boundary
        if (
            pt.x > self.w - BLOCK_SIZE
            or pt.x < 0
            or pt.y > self.h - BLOCK_SIZE
            or pt.y < 0
        ):
            return True
        # hits itself
        if pt in self.snake[1:]:
            return True

        return False

    # Updating the ui that is displayed to the screen
    def _update_ui(self):
        self.display.fill(BLACK)

        for pt in self.snake:
            pygame.draw.rect(
                self.display, BLUE1, pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE)
            )
            pygame.draw.rect(
                self.display, BLUE2, pygame.Rect(pt.x + 4, pt.y + 4, 12, 12)
            )

        pygame.draw.rect(
            self.display,
            RED,
            pygame.Rect(self.food.x, self.food.y, BLOCK_SIZE, BLOCK_SIZE),
        )

        text = font.render("Score: " + str(self.score), True, WHITE)
        self.display.blit(text, [0, 0])
        pygame.display.flip()

    # Defining the snake making a move and how it affects the head of the snake
    def _move(self, action):
        # Defining all the possible directions in a clockwise order
        clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        # Getting the current direction from the action
        idx = clock_wise.index(self.direction)

        # Using the Action to define what direction we want to go in now
        # Using np.array_equal() to check if the two arrays are equal or not
        if np.array_equal(action, [1, 0, 0]):
            # As we want to go straight, continue in the same direction
            new_dir = clock_wise[idx]
        elif np.array_equal(action, [0, 1, 0]):
            # Going right, so need to go +1 clockwise. Modulus 4 so that you go back to the beggining at the end
            new_idx = (idx + 1) % 4
            new_dir = clock_wise[new_idx]
        else:
            # Going left, so need to go -1 clockwise
            new_idx = (idx - 1) % 4
            new_dir = clock_wise[new_idx]

        # Setting self.direction to the found direction
        self.direction = new_dir

        # Getting the location of the snakes head
        x = self.head.x
        y = self.head.y

        # Performing the action on the snake's head
        if self.direction == Direction.RIGHT:
            x += BLOCK_SIZE
        elif self.direction == Direction.LEFT:
            x -= BLOCK_SIZE
        elif self.direction == Direction.DOWN:
            y += BLOCK_SIZE
        elif self.direction == Direction.UP:
            y -= BLOCK_SIZE

        self.head = Point(x, y)
