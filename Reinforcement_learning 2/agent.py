import time
import torch
import random
import numpy as np
from collections import deque
from game import GameManager
from model import DeepNetwork, QTrainer

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001

class Agent:
    def __init__(self, num_hiddens):
        self.n_games = 0
        self.epsilon = 1
        self.min_epsilon = 0.1
        self.epsilon_decay = 0.95
        self.gamma = 0.9
        self.num_hiddens = num_hiddens
        self.memory = deque(maxlen=MAX_MEMORY)
        self.model = DeepNetwork(4, self.num_hiddens, 3)
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma,)

    def get_state(self, game):
        state = [game.ball.rect.x/600, game.ball.rect.y/800, game.ball.speed_y/10, game.paddle2.rect.y/600]
        
        return np.array(state, dtype=int)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE)
        else:
            mini_sample = self.memory
        
        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    def decay_epsilon(self):
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)
        
    def get_action(self, state):
        final_move = [0,0,0]
        if np.random.random() < self.epsilon:
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float).unsqueeze(0)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1
        
        return final_move


def train():
    agent = Agent(64)
    game = GameManager()
    while True:
        state_old = agent.get_state(game)
        final_move = agent.get_action(state_old)
        reward, done, score = game.play_step(final_move)
        state_new = agent.get_state(game)
        agent.train_short_memory(state_old, final_move, reward, state_new, done)
        agent.remember(state_old, final_move, reward, state_new, done)

        if done:
            agent.decay_epsilon()
            game.reset()
            agent.n_games += 1
            agent.train_long_memory()
            print('Game ', agent.n_games, 'Score ', score)

if __name__ == "__main__":

    train()