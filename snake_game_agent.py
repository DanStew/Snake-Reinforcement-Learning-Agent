import torch
import random
import numpy as np
from collections import deque
from snake_game_environment import SnakeGameAI, Direction, Point

# Defining some constant parameters used throughout the agent
MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001  # Learning Rate


class SnakeGameAgent:

    # Function to initialise the Snake Game Agent
    def __init__(self):
        self.n_games = 0  # Number of games the agent has played
        self.epsilon = 0  # Parameter to control the randomness of the agent
        self.gamma = 0  # Discount rate (included as part of the model)
        # Defining some memory structure for the agent
        self.memory = deque(
            maxlen=MAX_MEMORY
        )  # If you exceed MAX_MEMORY, it automatically pops items from the deque
        # TODO : model, trainer

    # Function to get the current state of the agent from the game
    def get_state(self, game):
        pass

    # Function to remember actions, state, rewards, etc. Any previous things the agent has done
    # The Done variable represents Game_Over here
    def remember(self, state, action, reward, next_state, done):
        pass

    # Function to train the agent based on long memory
    def train_long_memory(self):
        pass

    # Function to train the agent based on short memory, most recent
    def train_short_memory(self, state, action, reward, next_state, done):
        pass

    # Function to get the next action the agent should do, based on the current state
    def get_action(self, state):
        pass


# The Train function used to train the agent on the Snake Game
def train():
    plot_scores = []  # Keeping track of the scores the agent has gotten
    plot_mean_scores = []  # Keeping track of the different means scores for the agent
    total_score = 0  # All scores combined
    record = 0  # Highest score

    # Making the agent and the game
    agent = SnakeGameAgent()
    game = SnakeGameAI()

    while True:
        # Getting the old state
        state_old = agent.get_state(game)
        # Getting the move from the current state
        final_move = agent.get_action(state_old)
        # Performing the move and get new State
        reward, done, score = game.play_step(final_move)
        # Defining the new state
        state_new = agent.get_state(game)

        # Training the short memory
        agent.train_short_memory(state_old, final_move, reward, state_new, done)

        # Remember this information
        agent.remember(state_old, final_move, reward, state_new, done)

        # Checking if gameover
        if done:
            # Train long memory, Plot Result
            game.reset()
            agent.n_games += 1
            agent.train_long_memory()

            if score > record:
                record = score
                # agent.model.save()

        print("Game ", agent.n_games, " Score ", score, " Record ", record)

        # TODO : Plot the Graph


if __name__ == "__main__":
    train()
