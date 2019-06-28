#!/usr/bin/env python3
# encoding utf-8

from DiscreteHFO.HFOAttackingPlayer import HFOAttackingPlayer
from DiscreteHFO.Agent import Agent
import argparse
import numpy as np
import math


class SARSAAgent(Agent):
	def __init__(self, learningRate, discountFactor, epsilon, initVals=0.0):
		super(SARSAAgent, self).__init__()
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
		self.last_reward = 0.0
		self.last_action = 'KICK'
		self.last_state = ((0,0),(0,0))
		self.action_n = 'KICK'


	def learn(self):
		num = self.A.index(self.last_action)
		q_predict = self.q_table[self.last_state][num]
		old_value = self.q_table[self.last_state][num]
		if self.action_n != None:
			if self.status != 0 :
				num_n = self.A.index(self.action)
				q_target = self.last_reward + self.discountFactor* self.q_table[self.state][num_n]
			else:
				q_target = self.last_reward
			self.q_table[self.last_state][num] += self.learningRate * (q_target - q_predict)
		else:
			num_n = self.A.index(self.action)
			q_target = self.reward
			old_value = self.q_table[self.state][num_n]
			q_predict = self.q_table[self.state][num_n]
			self.q_table[self.state][num_n] += self.learningRate * (q_target- q_predict)
			return self.q_table[self.state][num_n]-old_value

		return self.q_table[self.last_state][num]-old_value

	def act(self):
		temp =[]
		if np.random.uniform() < (1-self.epsilon):
			state_action = self.q_table[self.state]
			for i in range(5):
				if state_action[i] == np.max(state_action):
					temp.append(self.A[i])
			action = np.random.choice(temp)
		else:
			action = np.random.choice(self.A)
		return action

	def setState(self, state):
		self.state= state
		self.last_action = self.action
		self.last_reward = self.reward
		self.last_state = self.state


	def setExperience(self, state, action, reward, status, nextState):
		if action != None:
			self.state = state
			self.action = action
			self.reward = reward
			self.status = status
			self.nextState = nextState
		else:
			self.action_n = action


	def computeHyperparameters(self, numTakenActions, episodeNumber):
		lr_s = 0.5
		lr_e = 0.05
		lr_d = 5e3
		eps_s = 1
		eps_e = 0.01
		eps_d = 5e3
		self.learningRate = lr_e + (lr_s - lr_e) * np.exp(-1 * numTakenActions / lr_d)
		if self.epsilon >0.1:
			self.epsilon =1.2-(1 / (1 + math.exp(-0.002*(episodeNumber-500))))
		return self.learningRate,self.epsilon
		'''
		lr_s = 0.5
		lr_e = 0.05
		lr_d = 5e3
		eps_s = 1
		eps_e = 0.01
		eps_d = 5e3
		self.learningRate = lr_e + (lr_s - lr_e) * np.exp(-1 * numTakenActions / lr_d)
		self.epislon = max(math.exp(-1*5/2000*episodeNumber),0.05)
		return self.learningRate,self.epsilon

		#if self.epsilon >0.1:
			#self.epsilon =  1 / (1 + math.exp(0.01*(episodeNumber-1500)))
		#self.epsilon =1.2-(1 / (1 + math.exp(-0.002*(episodeNumber-600))))
		'''
	def toStateRepresentation(self, state):
		state_t = tuple(state)
		if state_t not in self.q_table.keys():
			self.q_table[state_t]= [0.0,0.0,0.0,0.0,0.0]
		return state_t

	def reset(self):
		pass

	def setLearningRate(self, learningRate):
		self.learningRate = learningRate

	def setEpsilon(self, epsilon):
		self.epsilon = epsilon

if __name__ == '__main__':

	parser = argparse.ArgumentParser()
	parser.add_argument('--id', type=int, default=0)
	parser.add_argument('--numOpponents', type=int, default=0)
	parser.add_argument('--numTeammates', type=int, default=0)
	parser.add_argument('--numEpisodes', type=int, default=5000)

	args=parser.parse_args()

	numEpisodes = args.numEpisodes
	# Initialize connection to the HFO environment using HFOAttackingPlayer
	hfoEnv = HFOAttackingPlayer(numOpponents = args.numOpponents, numTeammates = args.numTeammates, agentId = args.id)
	hfoEnv.connectToServer()

	# Initialize a SARSA Agent
	agent = SARSAAgent(0.75, 0.99,1.0)

	# Run training using SARSA
	numTakenActions = 0
	for episode in range(numEpisodes):
		agent.reset()
		status = 0

		observation = hfoEnv.reset()
		nextObservation = None
		epsStart = True

		while status==0:
			learningRate, epsilon = agent.computeHyperparameters(numTakenActions, episode)
			agent.setEpsilon(epsilon)
			agent.setLearningRate(learningRate)

			obsCopy = observation.copy()
			agent.setState(agent.toStateRepresentation(obsCopy))
			action = agent.act()

			numTakenActions += 1

			nextObservation, reward, done, status = hfoEnv.step(action)
			print(obsCopy, action, reward, nextObservation)
			agent.setExperience(agent.toStateRepresentation(obsCopy), action, reward, status, agent.toStateRepresentation(nextObservation))

			if not epsStart :
				agent.learn()
			else:
				epsStart = False

			observation = nextObservation

		agent.setExperience(agent.toStateRepresentation(nextObservation), None, None, None, None)
		agent.learn()
