import torch
import random
import pygame
import numpy as np
from snake_game import Score, Snake, Apple
from snake_game import play_step, reset_game, screen1
from plot_graph import plot
from deep_q_learning import Linear_Network, QTrainer
from collections import deque

# TWO ISSUES :
# GAME WINDOW NOT SET UP PROPERLY 
# RESET FUNCTION NOT WORKING PROPERLY

# CONSTANTS
MAX_MEMORY = 100_000
BATCH_SIZE = 1000
ALPHA = 0.001


# CLASS INSTANCES
snake = Snake()
apple = Apple()
score = Score()

exploring = 0
exploiting = 0

class Agent:

    def __init__(self):
        self.num_games = 0
        self.epsilon = 0
        self.gamma = 0.9
        self.memory = deque(maxlen=MAX_MEMORY)
        self.model = Linear_Network(11, 256, 2)
        self.trainer = QTrainer(self.model, alpha = ALPHA, gamma = self.gamma)


    def get_state(self, snake):
        head_x, head_y = snake.body[0]

        apple_x, apple_y = apple.position

        # Defines points around the head
        point_l = (head_x - 20, head_y)
        point_r = (head_x + 20, head_y)
        point_u = (head_x, head_y - 20)
        point_d = (head_x, head_y + 20)

        dir_l = 0
        dir_r = 0
        dir_d = 0
        dir_u = 0

        if snake.direction == (1, 0): # GOES RIGHT
            dir_r = 1
        if snake.direction == (-1, 0): # GOES LEFT
            dir_l = 1
        if snake.direction == (0, 1): # GOES DOWN
            dir_d = 1
        if snake.direction == (0, -1): # GOES UP
            dir_u = 1

        # Don't need to check for danger down because snake cannot perform an 180 rotation
        # Array that stores binary representation of the state of the snake
        state = [(dir_r and (snake.collide_self(point_r) or snake.collide_wall(point_r))) or # Danger straight
            (dir_l and (snake.collide_self(point_l) or snake.collide_wall(point_l))) or
            (dir_u and (snake.collide_self(point_u) or snake.collide_wall(point_u))) or
            (dir_d and (snake.collide_self(point_d) or snake.collide_wall(point_d))),

            (dir_u and (snake.collide_self(point_r) or snake.collide_wall(point_r))) or # Danger right
            (dir_d and (snake.collide_self(point_l) or snake.collide_wall(point_l))) or
            (dir_l and (snake.collide_self(point_u) or snake.collide_wall(point_u))) or
            (dir_r and (snake.collide_self(point_d) or snake.collide_wall(point_d))),

            (dir_d and (snake.collide_self(point_r) or snake.collide_wall(point_r))) or # Danger left
            (dir_u and (snake.collide_self(point_l) or snake.collide_wall(point_l))) or
            (dir_r and (snake.collide_self(point_u) or snake.collide_wall(point_u))) or
            (dir_l and (snake.collide_self(point_d) or snake.collide_wall(point_d))),

            dir_l, #direction
            dir_r,
            dir_u,
            dir_d,

            apple_x < head_x, # Apple to the left
            apple_x > head_x, # Apple to the right
            apple_y < head_y, # Apple above
            apple_y > head_y  # Apple below
        ]

        return np.array(state, dtype = int)

    # Stores the current state, action, reward, next state, done
    def remember(self, current_state, action, reward, next_state, done):
        self.memory.append((current_state, action, reward, next_state, done))

    # Trains the Agent's Q-net using either a random sample from mem or the entire mem (depending on its size)
    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            sample = random.sample(self.memory, BATCH_SIZE)
        else:
            sample = self.memory

        state, action, reward, next_state, done = zip(*sample)
        self.trainer.train_step(state, action, reward, next_state, done)

    # Trains Q-net 
    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    # Implements the exploration vs exploitation trade-off
    def get_action(self, state):
        global exploiting, exploring
        self.epsilon = 120 - self.num_games
        final_move = [0, 0]

        move_dict = {
            0: (1, 0), # RIGHT
            1: (-1, 0), # LEFT
            2: (0, 1), # DOWN
            3: (0, -1) # UP
        }

        opposite_moves = {
            (1, 0): (-1, 0),
            (-1, 0): (1, 0),
            (0, 1): (0, -1),
            (0, -1): (0, 1)
        }

        if random.randint(0, 200) < self.epsilon: # Explore
            move = random.randint(0, 3)
            exploring += 1
        else: # Exploit
            exploiting += 1
            current_state = torch.tensor(state, dtype = torch.float)
            prediction = self.model(current_state)
            probabilities = torch.nn.functional.softmax(prediction, dim=0)
            move = torch.argmax(probabilities).item()

        final_move = move_dict[move]
        # print(type(final_move))
        # print(final_move)

        # Prevent the snake from moving in the opposite direction
        current_direction = snake.direction
        if final_move == opposite_moves[current_direction]:
            final_move = current_direction

        return final_move


def train():
    global exploring, exploiting
    score_lst = []
    mean_score_lst = []
    total_score = 0
    max_score = 0
    agent = Agent()
    score = Score()

    while True:
        
        curr_state = agent.get_state(snake)
        final_move = agent.get_action(curr_state)
        reward, game_over, curr_score = play_step(snake, apple, score, final_move)
        new_state = agent.get_state(snake)
        agent.train_short_memory(curr_state, final_move, reward, new_state, game_over)
        agent.remember(curr_state, final_move, reward, new_state, game_over)

        # Train long memory only after a loss
        if game_over:

            # print("CURR SCORE: ", curr_score)
            # Resets the game
            pygame.display.set_mode((720, 480)) # Does not seem to be working
            snake.body = [(15, 10), (14, 10), (13, 10)]
            snake.direction = (1, 0)
            apple.random_pos()
            score.reset()

            agent.num_games += 1
            agent.train_long_memory()
            if curr_score > max_score:
                max_score = curr_score
                agent.model.save()


            print('Game', agent.num_games, 'Score', curr_score, 'Max Score:', max_score)


            score_lst.append(curr_score)
            total_score += curr_score
            mean_score = total_score / agent.num_games
            mean_score_lst.append(mean_score)
            plot(score_lst, mean_score_lst)

            explore_exploit_ratio = exploring / exploiting
            print("RATIO: EXPLORE/EXPLOIT: ", explore_exploit_ratio)
            





if __name__ == "__main__":
    train()

