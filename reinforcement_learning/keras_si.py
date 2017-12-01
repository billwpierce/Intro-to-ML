#Heavily copied from this code: https://keon.io/deep-q-learning/

# -*- coding: utf-8 -*-
# Import necessary libraries.e
import random
import csv
import gym
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam

# Define number of episodes to train on.
EPISODES = 1001

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size # The size of each frame of the game in number of pixels.
        self.action_size = action_size # The number of possible actions
        self.memory = deque(maxlen=2000) # The memory being used to remember previous states and actions.
        self.gamma = 0.95    # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01 # Minimum exploration variable value.
        self.epsilon_decay = 0.995 # Decay rate
        self.learning_rate = 0.001 # Used by adam for optimization
        self.model = self._build_model() # The actual model

    def _build_model(self): # Define the model
        # Neural Net for Deep-Q learning Model
        model = Sequential() # Make it sequential
        # Define a model with 3 hidden layers.
        model.add(Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_size, activation='softmax'))
        model.compile(loss='mse',
                      optimizer=Adam(lr=self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state, done): # Record previous states.
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon: # If a random number from 1-10 is less than the current learning value
            return random.randrange(self.action_size) # Choose a random value, exploring
        act_values = self.model.predict(state) # predicts best action
        return np.argmax(act_values[0])  # returns action based on which has a higher % of working based on model prediction

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size) # Sample a random number of previous states to replay them/
        for state, action, reward, next_state, done in minibatch:
            target = reward # Set the resulting value to the reward
            if not done: # If the action never finished
                # Predict what the reward would be
                target = (reward + self.gamma *
                          np.amax(self.model.predict(next_state)[0]))
            target_f = self.model.predict(state) # what the model chose to do
            target_f[0][action] = target # Sets predicted reward for an action to predicted actual reward.
            self.model.fit(state, target_f, epochs=1, verbose=0) # trains network to learn that this state has this reward
        if self.epsilon > self.epsilon_min: # Decay model if able to
            self.epsilon *= self.epsilon_decay

    # Remembering the weights (not implemented):
    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)

def addToFile(file, what): # from https://stackoverflow.com/questions/13203868/how-to-write-to-csv-and-not-overwrite-past-text
    f = csv.writer(open(file, 'a')).writerow(what) # appends to csv file

if __name__ == "__main__":
    env = gym.make('SpaceInvaders-v0') # Open space invaders
    state_size = 100800 # set the state size to the screen
    action_size = env.action_space.n # set the numbr of actions to the possible actions
    agent = DQNAgent(state_size, action_size) # initialize a DQN model
    # agent.load("./save/cartpole-dqn.h5")
    done = False
    batch_size = 16 # set the number of episodes before training

    for e in range(EPISODES): # for every episode
        state = env.reset() # reset the environment
        state = np.ravel(state).reshape((1, -1)) # Turn the environment into a list
        score = 0
        e+=1
        while True: # while the game is running
            # time += 1
            if e % 100 == 0: # Every 100 episodes, render them to view progress
                env.render()
            action = agent.act(state) # choose an action
            next_state, reward, done, _ = env.step(action) # see how that action changes the environment
            reward = reward if not done else -10 # set reward to reward, or to -10 if done
            # print(reward)
            if not done:
                score += reward # Add your reward to the score
            next_state = np.ravel(next_state).reshape((1, -1)) # ravel the next state into correct size
            agent.remember(state, action, reward, next_state, done) # remember current states
            state = next_state
            if done: # print current situation
                print("episode: {}/{}, score: {}, e: {:.2}"
                      .format(e, EPISODES, score, agent.epsilon))
                addToFile("test.csv",([e, score])) # add data to file for later analization
                break
        if len(agent.memory) > batch_size: # after a certain number of repeats, retrain
            agent.replay(batch_size)