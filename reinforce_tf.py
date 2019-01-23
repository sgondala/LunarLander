import gym
import numpy as np 
import tensorflow as tf
import random as random
import math
import datetime
import pickle
import math
import datetime

from tensorflow import set_random_seed
set_random_seed(0)
np.random.seed(0)
random.seed(0)


class Reinforce():
	def __init__(self):
		self.env = gym.make('LunarLander-v2')
		# self.env = gym.make('LunarLander-v0')		
		self.env.seed(0)		
		self.scores = [0]*100
		self.numberOfActions = self.env.action_space.n
		self.numberOfAttributesInState = self.env.observation_space.shape[0]
		self.gamma = 0.99
		self.alpha = 0.005
		self.states = list()
		self.actions = list()
		self.rewards = list()
		self.stateValues = list()

		# self.stateValues = np.array([0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0])

		self.stateVectorInput = tf.placeholder(tf.float32, [None, self.numberOfAttributesInState])
		self.actionVector = tf.placeholder(tf.int32, [None, ])
		self.gVector = tf.placeholder(tf.float32, [None, ])

		self.layer1 = tf.layers.dense(inputs=self.stateVectorInput, units=16, activation=tf.nn.relu)
		self.layer2 = tf.layers.dense(inputs=self.layer1, units=16, activation=tf.nn.relu)
		self.outputLayer = tf.layers.dense(inputs=self.layer2, units=self.numberOfActions, activation=None)
		# Chosing output layer to be linear, not sigmoid

		self.partialDerivativeStep = tf.nn.sparse_softmax_cross_entropy_with_logits(logits = self.outputLayer, labels=self.actionVector)
		self.loss = tf.reduce_mean(self.partialDerivativeStep * self.gVector)
		self.optimizer = tf.train.AdamOptimizer(self.alpha).minimize(self.loss)

		self.softmaxOutputProb = tf.nn.softmax(self.outputLayer)

		self.session = tf.Session()
		self.session.run(tf.global_variables_initializer())		

	def getAction(self, state):
		out = self.session.run(self.softmaxOutputProb, feed_dict = {self.stateVectorInput: np.reshape(state, [1, self.numberOfAttributesInState]) })
		action = np.random.choice(range(self.numberOfActions), p=out.ravel())
		return action

	def store(self, state, action, reward):
		self.states.append(state)
		self.actions.append(action)
		self.rewards.append(reward)
		
	def computeGValues(self):
		self.stateValues = np.zeros(len(self.rewards))
		self.stateValues[-1] = self.rewards[-1]
		for index in reversed(range(0,len(self.rewards)-1)):
			self.stateValues[index] = self.gamma*self.stateValues[index+1] + self.rewards[index]

		self.stateValues -= np.mean(self.stateValues)
		self.stateValues /= np.std(self.stateValues)

	def train(self):
		self.computeGValues()
		# print("States", self.states)
		# print("Actions", self.actions)
		# print("State Values", self.stateValues)
		
		self.session.run([self.optimizer, self.loss], feed_dict= {self.stateVectorInput:np.vstack(self.states), self.actionVector:np.array(self.actions), self.gVector:np.array(self.stateValues)})

		self.states = list()
		self.actions = list()
		self.rewards = list()
		self.stateValues = list()

	def loadFromCheckpoint(self):
		# tf.train.Saver().restore(self.session, fileName)
		tf.train.Saver().restore(self.session, "weights/final_output")


	def test(self):
		scoreValues = []
		for episodeNum in range(100):
			score = 0
			currState = self.env.reset()
			currState = np.reshape(currState, [1, self.numberOfAttributesInState])
			done = False
			while not done:
				prevState = currState
				action = self.getAction(prevState)
				currState,reward,done,_ = self.env.step(action)
				currState = np.reshape(currState, [1,self.numberOfAttributesInState])
				self.store(prevState, action, reward)
				score += reward
			scoreValues.append(score)
		# print(scoreValues)
		return scoreValues


	def run(self):
		rewardsVectorForDumping = []
		meanScoreVector = []
		# rewardsVectorForDumping = np.array(rewardsVectorForDumping)
		for episodeNum in range(4010):
			score = 0
			currState = self.env.reset()
			currState = np.reshape(currState, [1, self.numberOfAttributesInState])
			done = False
			while not done:
				prevState = currState
				action = self.getAction(prevState)
				currState,reward,done,_ = self.env.step(action)
				currState = np.reshape(currState, [1,self.numberOfAttributesInState])
				self.store(prevState, action, reward)
				score += reward

			rewardsVectorForDumping.append(score)	
			self.scores[episodeNum%100] = score

			meanScore = np.mean(self.scores)

			if episodeNum > 100 and meanScore >= 200:
				tf.train.Saver().save(self.session, "weights/final_output")
				meanScoreVector.append(meanScore)
				file = open("rewards/LunarLander-rewardsfinal", 'wb')
				pickle.dump(rewardsVectorForDumping, file)
				pickle.dump(meanScoreVector, 
					open("rewards/LunarLander-meanScoreFinal",'wb'))
				print("Solved ", meanScore)
				return

			if episodeNum%100 == 0:
				meanScoreVector.append(meanScore)
				print("Episode ", episodeNum, "Score", meanScore, str(datetime.datetime.now()))
				# print(str(datetime.datetime.now()))

			if episodeNum%1000 == 0:
				tf.train.Saver().save(self.session, "weights/LunarLander-" + str(episodeNum))
				fileName = "rewards/LunarLander-rewards" + str(episodeNum)
				file = open(fileName, 'wb')
				pickle.dump(rewardsVectorForDumping, file)
				pickle.dump(meanScoreVector, 
					open("rewards/LunarLander-meanScore" +str(episodeNum),'wb'))

			self.train()

agent = Reinforce()

# To train, comment out the next 3 lines and uncomment last line
agent.loadFromCheckpoint()
scorevaluestest = agent.test()
print(scorevaluestest)

# to train, uncomment the below line 
# agent.run()
