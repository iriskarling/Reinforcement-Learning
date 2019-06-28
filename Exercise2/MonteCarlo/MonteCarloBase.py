#!/usr/bin/env python3
# encoding utf-8

from DiscreteHFO.HFOAttackingPlayer import HFOAttackingPlayer
from DiscreteHFO.Agent import Agent
import argparse
import numpy as np
from collections import defaultdict
import math

class MonteCarloAgent(Agent):
	def __init__(self, discountFactor, epsilon, initVals=0.0):
		super(MonteCarloAgent, self).__init__()
		self.discountFactor = discountFactor
		self.epsilon = epsilon
		self.q_table = {}
		self.episode = []
		self.R =defaultdict(lambda:[])
		self.A = ["DRIBBLE_UP","DRIBBLE_DOWN","DRIBBLE_LEFT","DRIBBLE_RIGHT",'KICK']
		self.reward = 0.0
		self.status = 0
		self.state =((0,0),(0,0))
		self.nextState =((0,0),(0,0))
		self.action = ""
		self.G = {}


	def learn(self):
		C = 0.0
		temp = []
		output = []
		for i in reversed(self.episode):
			state,action,reward = i
			C = reward + self.discountFactor*C
			self.G[(state,action)]= C

		for j in self.episode:
			state,action,reward = j
			if (state,action) not in temp:
				temp.append((state,action))
				self.R[(state,action)].append(self.G[(state,action)])
				a = np.average(self.R[(state,action)])
				num = self.A.index(action)
				self.q_table[state][num] = a
				output.append(self.q_table[state][num])

		return self.q_table,output




	def toStateRepresentation(self, state):
		state_t = tuple(state)
		if state_t not in self.q_table.keys():
			self.q_table[state_t]= [0.0,0.0,0.0,0.0,0.0]
		return state_t

	def setExperience(self, state, action, reward, status, nextState):
		self.state = state
		self.action = action
		self.reward = reward
		self.status = status
		self.nextState = nextState
		self.episode.append((self.state,self.action,self.reward))


	def setState(self, state):
		self.state = state

	def reset(self):
		self.G = {}
		self.episode = []

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

	def setEpsilon(self, epsilon):
		self.epsilon = epsilon

	def computeHyperparameters(self, numTakenActions, episodeNumber):
		#self.epsilon =  1 / (1 + math.exp(0.005*(episodeNumber-100))) + 0.2
		if self.epsilon >0.1:
			self.epsilon =1.2-(1 / (1 + math.exp(-0.002*(episodeNumber-500))))
		return self.epsilon


if __name__ == '__main__':

	parser = argparse.ArgumentParser()
	parser.add_argument('--id', type=int, default=0)
	parser.add_argument('--numOpponents', type=int, default=0)
	parser.add_argument('--numTeammates', type=int, default=0)
	parser.add_argument('--numEpisodes', type=int, default=5000)

	args=parser.parse_args()

	#Init Connections to HFO Server
	hfoEnv = HFOAttackingPlayer(numOpponents = args.numOpponents, numTeammates = args.numTeammates, agentId = args.id)
	hfoEnv.connectToServer()

	# Initialize a Monte-Carlo Agent
	agent = MonteCarloAgent(discountFactor = 0.99, epsilon = 1.0)
	numEpisodes = args.numEpisodes
	numTakenActions = 0
	# Run training Monte Carlo Method
	for episode in range(numEpisodes):
		agent.reset()
		observation = hfoEnv.reset()
		status = 0

		while status==0:
			epsilon = agent.computeHyperparameters(numTakenActions, episode)
			agent.setEpsilon(epsilon)
			obsCopy = observation.copy()
			agent.setState(agent.toStateRepresentation(obsCopy))
			action = agent.act()
			numTakenActions += 1
			nextObservation, reward, done, status = hfoEnv.step(action)
			agent.setExperience(agent.toStateRepresentation(obsCopy), action, reward, status, agent.toStateRepresentation(nextObservation))
			observation = nextObservation

		agent.learn()
