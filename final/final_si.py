# Heavily copied from this code: https://keon.io/deep-q-learning/

# -*- coding: utf-8 -*-
import random
import gym
import csv
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
from keras.optimizers import SGD
from keras.losses import categorical_crossentropy
from functools import reduce
from operator import mul

EPISODES = 1001


class DQNAgent:
    def __init__(self, state_shape, action_size):
        self.state_size = reduce(mul, state_shape, 1)
        self.input_shape = state_shape
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.99    # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.999
        self.learning_rate = 0.001
        self.model = self._build_model()

    def _build_model(self):
        # Neural Net for Deep-Q learning Model, based off http://adventuresinmachinelearning.com/keras-tutorial-cnn-11-lines/
        model = Sequential()
        model.add(Conv2D(64, kernel_size=(5, 4), strides=(1, 1),
                 activation='relu',
                 input_shape=self.input_shape))
        print(self.input_shape)
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        model.add(Conv2D(64, (7, 5), activation='relu'))
        model.add(MaxPooling2D(pool_size=(1, 2)))
        model.add(Flatten())
        model.add(Dense(1000, activation='relu'))
        model.add(Dense(self.action_size, activation='softmax'))
        model.compile(loss=categorical_crossentropy,
            optimizer=SGD(lr=0.01),
            metrics=['accuracy'])
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state.reshape(1, state.shape[0], state.shape[1], state.shape[2]))
        return np.argmax(act_values[0])  # returns action

    def replay(self, batch_size):
        minibatch = random.sample(list(self.memory), batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = (reward + self.gamma *
                          np.amax(self.model.predict(next_state.reshape(1, next_state.shape[0], next_state.shape[1], next_state.shape[2]))[0]))
            target_f = self.model.predict(state.reshape(1, state.shape[0], state.shape[1], state.shape[2]))
            target_f[0][action] = target
            self.model.fit(state.reshape(1, state.shape[0], state.shape[1], state.shape[2]), target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)

def addToFile(file, what): # from https://stackoverflow.com/questions/13203868/how-to-write-to-csv-and-not-overwrite-past-text
    f = csv.writer(open(file, 'a')).writerow(what) # appends to csv file

if __name__ == "__main__":
    env = gym.make('SpaceInvaders-v0')
    state_shape = env.observation_space.shape
    state_size = reduce(mul, state_shape, 1)
    action_size = env.action_space.n
    agent = DQNAgent(state_shape, action_size)
    # agent.load("./save/cartpole-dqn.h5")
    done = False
    batch_size = 32

    for e in range(EPISODES):
        state = env.reset()
        score = 0
        # state = np.reshape(state, [1, state_size])
        while True: # while the game is running
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            reward = reward if not done else -10
            if not done:
                score += reward # Add your reward to the score
            # next_state = np.reshape(next_state, [1, state_size])
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            if done:
                print("episode: {}/{}, score: {}, e: {:.2}"
                    .format(e, EPISODES, score, agent.epsilon))
                addToFile("test.csv",([e, score])) # add data to file for later analization
                break
        if len(agent.memory) > batch_size:
            agent.replay(batch_size)
