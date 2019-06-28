#!/usr/bin/env python3
# encoding utf-8

from DiscreteHFO.HFOAttackingPlayer import HFOAttackingPlayer
from DiscreteHFO.Agent import Agent
import argparse
import numpy as np
import math
class QLearningAgent(Agent):
	def __init__(self, learningRate, discountFactor, epsilon, initVals=0.0):
		super(QLearningAgent, self).__init__()
		self.learningRate = learningRate
		self.discountFactor = discountFactor
		self.epsilon = epsilon
		self.q_table = {}
		self.A = ["DRIBBLE_UP","DRIBBLE_DOWN","DRIBBLE_LEFT","DRIBBLE_RIGHT",'KICK']
		self.reward = 0.0
		self.status = 0
		self.state =((0,0),(0,0))
		self.nextState =((0,0),(0,0))
		self.action = ""
	def learn(self):
		num = self.A.index(self.action)
		q_predict = self.q_table[self.state][num]
		old_value = self.q_table[self.state][num]
		if self.status != 0:
			q_target = self.reward + self.discountFactor* np.max(self.q_table[self.nextState])
		else:
			q_target = self.reward

		self.q_table[self.state][num] += self.learningRate * (q_target - q_predict)
		return self.q_table[self.state][num]-old_value

	def act(self):
		temp =[]
		if np.random.uniform() <(1-self.epsilon):
			state_action = self.q_table[self.state]
			for i in range(5):
				if state_action[i] == np.max(state_action):
					temp.append(self.A[i])
			action = np.random.choice(temp)
		else:
			action = np.random.choice(self.A)
		return action

	def toStateRepresentation(self, state):
		state_t = tuple(state)
		if state_t not in self.q_table.keys():
			self.q_table[state_t]= [0.0,0.0,0.0,0.0,0.0]
		return state_t

	def setState(self, state):
		self.state = state

	def setExperience(self, state, action, reward, status, nextState):
		self.state = state
		self.action = action
		self.reward = reward
		self.status = status
		self.nextState = nextState

	def setLearningRate(self, learningRate):
		self.learningRate = learningRate

	def setEpsilon(self, epsilon):
		self.epsilon = epsilon

	def reset(self):
		pass

	def computeHyperparameters(self, numTakenActions, episodeNumber):
		lr_s = 0.5
		lr_e = 0.05
		lr_d = 3e4
		eps_s = 1
		eps_e = 0.01
		eps_d =3e4 
		self.learningRate = lr_e + (lr_s - lr_e) * np.exp(-1 * numTakenActions / lr_d)
		self.epsilon = eps_e + (eps_s - eps_e)* np.exp(-1 * numTakenActions / eps_d)
		self.epislon = max(math.exp(-1*6/2000*episodeNumber),0.1)#0.1
		return self.learningRate,self.epsilon

if __name__ == '__main__':

	parser = argparse.ArgumentParser()
	parser.add_argument('--id', type=int, default=0)
	parser.add_argument('--numOpponents', type=int, default=0)
	parser.add_argument('--numTeammates', type=int, default=0)
	parser.add_argument('--numEpisodes', type=int, default=5000)

	args=parser.parse_args()

	# Initialize connection with the HFO server
	hfoEnv = HFOAttackingPlayer(numOpponents = args.numOpponents, numTeammates = args.numTeammates, agentId = args.id)
	hfoEnv.connectToServer()

	# Initialize a Q-Learning Agent
	agent = QLearningAgent(learningRate = 0.5, discountFactor = 0.99, epsilon = 1.0)
	numEpisodes = args.numEpisodes

	# Run training using Q-Learning
	numTakenActions = 0
	for episode in range(numEpisodes):
		status = 0
		observation = hfoEnv.reset()

		while status==0:
			learningRate, epsilon = agent.computeHyperparameters(numTakenActions, episode)
			agent.setEpsilon(epsilon)
			agent.setLearningRate(learningRate)

			obsCopy = observation.copy()
			agent.setState(agent.toStateRepresentation(obsCopy))
			action = agent.act()
			numTakenActions += 1

			nextObservation, reward, done, status = hfoEnv.step(action)
			agent.setExperience(agent.toStateRepresentation(obsCopy), action, reward, status, agent.toStateRepresentation(nextObservation))
			update = agent.learn()

			observation = nextObservation
