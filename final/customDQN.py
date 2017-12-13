# Modified from https://keon.io/deep-q-learning/
# Based partially off https://www.intelnervana.com/demystifying-deep-reinforcement-learning/

# -*- coding: utf-8 -*-
# Import libraries
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

EPISODES = 100

def addToFile(file, what): # from https://stackoverflow.com/questions/13203868/how-to-write-to-csv-and-not-overwrite-past-text
    f = csv.writer(open(file, 'a')).writerow(what) # appends to csv file

class QNet:
	# Define Q neural network
	def __init__(self, state_shape, action_size):
		self.state_size = reduce(mul, state_shape, 1)
		self.input_shape = state_shape
		self.action_size = action_size
		self.memory = deque(maxlen=2000)
		self.discount_rate = 0.99
		self.exploration = 1.0
		self.exploration_decay = 0.99954
		self.exploration_min = 0.05
		self.learning_rate = 0.9
		self.model = self._build_model()

	# Build a model
	def _build_model(self):
		model = Sequential()
		model.add(Conv2D(32, kernel_size=(10, 10), strides=(5, 5), activation='relu', input_shape=self.input_shape))
		model.add(Conv2D(32, kernel_size=(8, 8), strides=(4, 4), activation='relu'))
		model.add(Conv2D(64, kernel_size=(4, 4), strides=(2, 2), activation='relu'))
		model.add(Conv2D(64, kernel_size=(3, 3), strides=(1, 1), activation='relu'))
		model.add(Flatten())
		model.add(Dense(512, activation='relu'))
		model.add(Dense(self.action_size, activation='linear'))
		model.compile(loss=categorical_crossentropy,
			optimizer=SGD(lr=0.08),
			metrics=['accuracy'])
		return model

	def process_state(self, state):
		return state.reshape(1, state.shape[0], state.shape[1], state.shape[2])

	def remember(self, state, action, reward, next_state, done):# remember our information
		self.memory.append((state, action, reward, next_state, done))# 	Just add it to memory
		
	def act(self, state):
		if self.exploration > np.random.rand():# if we are exploring
			return random.randrange(self.action_size) # random action
		else:
			outcomes = self.model.predict(self.process_state(state)) # choose the action in our current state that returns the highest value
			return np.argmax(outcomes[0])
	
	def train(self, action_num): # train our model
		# training_data = list(self.memory)[len(self.memory)-action_num: len(self.memory)] # take the most recent iterations of the model
		training_data = random.sample(list(self.memory), batch_size)
		for state, action, reward, next_state, done in training_data: # cycling through every moment that we are referencing
			true_reward = reward # define the model expected output as the current reward,
			if not done:
				predicted_now = self.model.predict(self.process_state(state))[0][action]
				predicted_next = self.model.predict(self.process_state(next_state))[0][action]
				true_reward = predicted_now + self.learning_rate * (reward + self.discount_rate * predicted_next - predicted_now) # set the Q function output as: current reward + timerate * best action based on next state
			predicted_rewards = self.model.predict(self.process_state(state)) # set the reward for our action in this state to the reward we just got
			predicted_rewards[0][action] = true_reward
			self.model.fit(self.process_state(state), predicted_rewards, epochs=1, verbose=0) # fit our model based on our state and target value
		# decay exploration if possible
		if self.exploration > self.exploration_min:
			self.exploration *= self.exploration_decay

if __name__ == "__main__": # Main part of game:
	env = gym.make('SpaceInvaders-v0')
	state_shape = env.observation_space.shape
	action_size = env.action_space.n
	agent = QNet(state_shape, action_size)
	done = False
	batch_size = 32

	for e in range(EPISODES): 
		state = env.reset()
		score = 0
		while True:
			action = agent.act(state)
			next_state, reward, done, _ = env.step(action)
			reward = reward if not done else -10
			if not done:
				score += reward # Add your reward to the score
			agent.remember(state, action, reward, next_state, done)
			state = next_state
			if done:
				print("episode: {}/{}, score: {}, e: {:.2}"
					.format(e, EPISODES, score, agent.exploration))
				addToFile("test.csv",([e, score])) # add data to file for later analization
				break
		if len(agent.memory) > batch_size:
			agent.train(batch_size)
	numpy.save('weights', agent.model.get_weights())