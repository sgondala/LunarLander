import gym
import numpy as np 
import tensorflow
import keras
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import Adam
import random as random
from collections import deque
import math
import datetime
import keras.backend as K
from tensorflow import set_random_seed
set_random_seed(0)
np.random.seed(0)
random.seed(0)




class DQN():
	def __init__(self):
		self.env = gym.make('LunarLander-v2')
		self.env.seed(0)
		# self.env = gym.make('CartPole-v0')
		self.memory = deque(maxlen=100000)
		self.scores = deque(maxlen=100)
		self.numberOfActions = self.env.action_space.n
		self.numberOfAttributesInState = self.env.observation_space.shape[0]
		print(self.numberOfAttributesInState, self.numberOfActions)
		self.gamma = 1.0
		self.batch_size = 64
		self.epsilon_min=0.01
		self.epsilon_decay=0.995
		self.epsilon = 1.0
		self.alpha = 0.01

		self.model = Sequential()
		self.model.add(Dense(24, input_dim=self.numberOfAttributesInState, activation='relu'))
		self.model.add(Dense(24,activation='relu'))
		self.model.add(Dense(self.numberOfActions, activation='linear'))
		self.model.add(Activation('softmax'))
		self.model.compile(loss=self.myLoss(gamma=self.gamma),optimizer=Adam(lr=self.alpha))

	def myLoss(self, gamma):
		def loss(y_true, y_pred):
			return K.categorical_crossentropy(y_true, y_pred)		
		return loss

	def getEpsilon(self, episodeNum):
		return max(self.epsilon_min, min(self.epsilon, 1.0 - math.log10((episodeNum + 1) * self.epsilon_decay)))
		# return max(self.epsilon_min, min(self.epsilon, 1.0 - (episodeNum + 1)*0.005))


	def getAction(self, state, epsilon):
		if(np.random.random() < epsilon):
			return self.env.action_space.sample() 
		return np.argmax(self.model.predict(state))

	def store(self, inputVal):
		self.memory.append(inputVal)

	def train(self):
		batch = random.sample(self.memory, min(self.batch_size, len(self.memory)))
		xvals = []
		yvals = []
		for prevState,action,reward,currState,done in batch:
			y = reward
			if not done:
				y = reward + self.gamma*np.max(self.model.predict(currState)[0])
			qValue = self.model.predict(prevState)
			qValue[0][action] = y
			xvals.append(prevState[0])
			yvals.append(qValue[0])
		
		self.model.fit(np.array(xvals), np.array(yvals), batch_size=len(xvals), verbose=0)
		if self.epsilon > self.epsilon_min:
			self.epsilon *= self.epsilon_decay


	# env = gym.make('LunarLander-v2')
	def run(self):
		for episodeNum in range(100000):
			score = 0
			currState = self.env.reset()
			currState = np.reshape(currState, [1, self.numberOfAttributesInState])
			done = False
			while not done:
				prevState = currState
				action = self.getAction(prevState, self.getEpsilon(episodeNum))
				currState,reward,done,_ = self.env.step(action)
				currState = np.reshape(currState, [1,self.numberOfAttributesInState])
				self.store((prevState, action, reward, currState, done))
				score += reward

			self.scores.append(score)

			mean_score = np.mean(self.scores)
			if episodeNum > 100 and mean_score > 195:
				self.model.save(str(episodeNum) + ":" + str(mean_score))
				print("Solved ", mean_score)
				return

			if episodeNum % 100 == 0:
				currentDT = datetime.datetime.now()
				print("Mean for 100 episodes ", episodeNum, np.mean(self.scores), str(currentDT))

			if episodeNum % 1000 == 0:
				self.model.save(str(episodeNum) + ":" + str(mean_score))


			self.train()

a = DQN()
a.run()



