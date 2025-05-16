import torch
import random
import numpy as np
from collections import deque
from snake_game_environment import SnakeGameAI, Direction, Point
from snake_game_model import Linear_QNet, QTrainer
from plotHelper import plot

# Defining some constant parameters used throughout the agent
MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001  # Learning Rate


class SnakeGameAgent:

    # Function to initialise the Snake Game Agent
    def __init__(self):
        self.n_games = 0  # Number of games the agent has played
        self.epsilon = 0  # Parameter to control the randomness of the agent
        self.gamma = 0.9  # Discount rate (included as part of the model and trainer)
        # Defining some memory structure for the agent
        self.memory = deque(
            maxlen=MAX_MEMORY
        )  # If you exceed MAX_MEMORY, it automatically pops items from the deque
        self.model = Linear_QNet(
            11, 256, 3
        )  # Needs input size, hidden layer size and output size
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)

    # Function to get the current state of the agent from the game
    def get_state(self, game):
        # Getting the head point of the snake
        head = game.snake[0]
        # Finding the points up/down/left/right of the head
        point_l = Point(head.x - 20, head.y)
        point_r = Point(head.x + 20, head.y)
        point_u = Point(head.x, head.y - 20)
        point_d = Point(head.x, head.y + 20)

        # Boolean variables used to check whether we are heading in that direction or not
        # Could do this through an if statement however guess this is easier
        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN

        # Defining the current state for the agent
        state = [
            # Danger Straight
            (dir_r and game.is_collision(point_r))
            or (dir_l and game.is_collision(point_l))
            or (dir_u and game.is_collision(point_u))
            or (dir_d and game.is_collision(point_d)),
            # Danger Right
            (dir_u and game.is_collision(point_r))
            or (dir_d and game.is_collision(point_l))
            or (dir_l and game.is_collision(point_u))
            or (dir_r and game.is_collision(point_d)),
            # Danger Left
            (dir_d and game.is_collision(point_r))
            or (dir_u and game.is_collision(point_l))
            or (dir_r and game.is_collision(point_u))
            or (dir_l and game.is_collision(point_d)),
            # Move Direction (Defining what direction the agent is moving in)
            dir_l,
            dir_r,
            dir_u,
            dir_d,
            # Food Location
            game.food.x < game.head.x,  # food left
            game.food.x > game.head.x,  # food right
            game.food.y < game.head.y,  # food up
            game.food.y > game.head.y,  # food down
        ]

        # Returning the state array, however all items are ints (0 or 1) rather than boolean (true or false)
        return np.array(state, dtype=int)

    # Function to remember actions, state, rewards, etc. Any previous things the agent has done
    # The Done variable represents Game_Over here
    def remember(self, state, action, reward, next_state, done):
        # Appending all the items recieved to memory
        # NOTE : All items are stored as part of 1 tuple, not stored separately
        self.memory.append((state, action, reward, next_state, done))

    # Function to train the agent based on short memory, most recent
    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    # Function to train the agent based on long memory
    def train_long_memory(self):
        # If we have enough items in memory, get a random batch from memory
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE)
        else:
            mini_sample = self.memory

        # Extracting the separate states, action, etc from each of the tuples extracted
        # As each tuple has a state, action, etc. This code just goes through each tuple and extracts the value
        # It combines them into separate arrays for each of the values
        states, actions, rewards, next_states, dones = zip(*mini_sample)
        # Passing these extracted states into the train_step function
        self.trainer.train_step(states, actions, rewards, next_states, dones)

    # Function to get the next action the agent should do, based on the current state
    def get_action(self, state):
        # Make actions with a balance between randomness and exploitation
        self.epsilon = 80 - self.n_games  # Lower randomness as more games
        final_move = [0, 0, 0]
        # If the randomly generated number is <epsilon, move in a random direction
        # NOTE : If we play >80 games, epsilon is negative and agent will no longer choose any random move
        if random.randint(0, 200) < self.epsilon:
            # Choosing a random move and setting it to 1
            move = random.randint(0, 2)
            final_move[move] = 1
        # Using the model to choose the move
        else:
            # Converting the current state into a tensor (as model.predict wants tensor)
            state0 = torch.tensor(state, dtype=torch.float)
            # Making a prediction for the next move, using the state
            prediction = self.model(state0)
            # Getting the maximum value of prediction and setting it to 1
            move = torch.argmax(
                prediction
            ).item()  # .item() is used to convert outputted tensor into a number
            final_move[move] = 1

        return final_move


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
                agent.model.save()

            print("Game ", agent.n_games, " Score ", score, " Record ", record)

            # Updating the stored scores and mean_scores for the agent
            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.n_games
            plot_mean_scores.append(mean_score)

            # Updating the outputted graph
            plot(plot_scores, plot_mean_scores)


if __name__ == "__main__":
    train()
