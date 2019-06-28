#!/usr/bin/env python3
# encoding utf-8

import random
import argparse
from DiscreteMARLUtils.Environment import DiscreteMARLEnvironment
from DiscreteMARLUtils.Agent import Agent
from copy import deepcopy
import itertools
import argparse
import numpy as np

class JointQLearningAgent(Agent):
	def __init__(self, learningRate, discountFactor, epsilon, numTeammates, initVals=0.0):
		super(JointQLearningAgent, self).__init__()
		self.learningRate = learningRate
		self.discountFactor = discountFactor
		self.numTeammates = numTeammates
		self.initVals = initVals
		self.epsilon = epsilon
		self.A = ['MOVE_UP','MOVE_DOWN','MOVE_LEFT','MOVE_RIGHT','KICK','NO_OP']
		self.reward = 0.0
		self.status = 0
		self.state =[[[0,0],[0,0]],[[0,0]],[[0,0]]]
		self.nextState =[[[0,0],[0,0]],[[0,0]],[[0,0]]]
		self.action = ""
		self.q_table = {}
		self.c_table = {}
		self.n ={}
		self.action = 'KICK'
		self.oppoActions = ['NO_OP']


	def setExperience(self, state, action, oppoActions, reward, status, nextState):
		self.state = state
		self.action = action
		self.oppoActions = oppoActions

		self.reward = reward
		self.status = status
		self.nextState = nextState
		if nextState not in self.q_table:
			self.q_table[nextState]= np.zeros((6,6))
			self.c_table[nextState]= [0.0,0.0,0.0,0.0,0.0,0.0]
			self.n[nextState] = 0.0

	def learn(self):
		num = self.A.index(self.action)
		oppoActions = self.oppoActions[0]
		num_opponents = self.A.index(oppoActions)
		q_predict = self.q_table[self.state][num][num_opponents]
		old_value = self.q_table[self.state][num][num_opponents]
		maxx = -100000000
		action =self.action
		for i in range(6):
			summ = 0.0
			for i in range(6):
				if self.n[self.state] == 0:
					summ += self.q_table[self.state][num][num_opponents]* 1/36
				else:
					summ += (self.q_table[self.state][num][num_opponents]*self.c_table[self.state][num_opponents])/self.n[self.state]
			if summ >= maxx:
				maxx = summ

		if self.status != 0:
			q_target = self.reward + self.discountFactor* maxx
		else:
			q_target = self.reward
		self.q_table[self.state][num][num_opponents] += self.learningRate * (q_target - q_predict)
		self.c_table[self.state][num_opponents] += 1.0
		self.n[self.state] += 1

		return self.q_table[self.state][num][num_opponents]-old_value

	def act(self):
		num = self.A.index(self.action)
		oppoAction= self.oppoActions[0]
		num_oppo = self.A.index(oppoAction)
		maxx = -100000000
		action = self.action
		if np.random.uniform() < self.epsilon:
			for i in range(6):
				summ = 0.0
				for i in range(6):
					if self.n[self.state] == 0:
						summ += self.q_table[self.state][num][num_oppo]*1/36
					else:
						summ += (self.q_table[self.state][num][num_oppo]*self.c_table[self.state][num_oppo])/self.n[self.state]
				if summ >= maxx:
					maxx = summ
					action = self.action
		else:
			action = np.random.choice(self.A)

		return action

	def setEpsilon(self, epsilon) :
		return self.epsilon

	def setLearningRate(self, learningRate) :
		return self.learningRate

	def setState(self, state):
		self.state = state

	def toStateRepresentation(self, rawState):
		temp =[]
		for i in range(3):
		    temp1 = []
		    a =0
		    for k in rawState[i]:
		        str1 = tuple(k)
		        temp1.append(str1)
		        a +=1
		    str2 = tuple(temp1)
		    res =str2
		    if a==1:
		        res,=str2
		    temp.append(res)

		state_t = tuple(temp)
		if state_t not in self.q_table:
			self.q_table[state_t]= np.zeros((6,6))
			self.c_table[state_t]= [0.0,0.0,0.0,0.0,0.0,0.0]
			self.n[state_t] = 0.0
		return state_t

	def computeHyperparameters(self, numTakenActions, episodeNumber):
		lr_s= 0.5
		lr_e = 0.05
		lr_d = 3e5
		eps_s = 1
		eps_e= 0.05
		eps_d =3e5
		self.learningRate = lr_e + (lr_s - lr_e) * np.exp(-1 * numTakenActions / lr_d)
		self.epsilon = eps_e + (eps_s - eps_e)* np.exp(-1 * numTakenActions / eps_d)
		return self.learningRate,self.epsilon


def describe(data):
    tmp_status = data['status'][-500:]
    tmp_n_steps_episode = data['steps_in_episode'][-500:]
    ratio_goal = np.sum(np.array(tmp_status) == "GOAL") / len(tmp_status)
    ratio_oob = np.sum(np.array(tmp_status) == "OUT_OF_BOUNDS") / len(tmp_status)
    ratio_oot = np.sum(np.array(tmp_status) == "OUT_OF_TIME") / len(tmp_status)

    avg_n_steps_episode = sum(tmp_n_steps_episode) / len(tmp_n_steps_episode)

    print('============ INFO ============')
    print('%-25s:  %d' % ("TOTAL EPISODE NUM", len(data['status'])))
    print('===== LATEST 500 EPISODE =====')
    for info in zip(['GOAL', 'OUT_OF_BOUNDS', 'OUT_OF_TIME'],
                    [ratio_goal, ratio_oob, ratio_oot]):
        print('%-25s:  %.3f' % (info[0], info[1] * 100))

    print('\n')
    print('%-25s:  %.3f' % ('AVG STEPS TO FINISH', avg_n_steps_episode))

if __name__ == '__main__':

	parser = argparse.ArgumentParser()
	parser.add_argument('--numOpponents', type=int, default=1)
	parser.add_argument('--numAgents', type=int, default=2)
	parser.add_argument('--numEpisodes', type=int, default=10000)

	args=parser.parse_args()

	MARLEnv = DiscreteMARLEnvironment(numOpponents = args.numOpponents, numAgents = args.numAgents)
	agents = []
	numAgents = args.numAgents
	numEpisodes = args.numEpisodes
	for i in range(numAgents):
		agent = JointQLearningAgent(learningRate = 0.5, discountFactor = 0.9, epsilon = 1.0, numTeammates=args.numAgents-1)
		agents.append(agent)
	numEpisodes = 10000
	numTakenActions = 0
	numTakenActionCKPT = 0
	status_lst = []
	n_episode = []
	data = {'status':status_lst,'steps_in_episode':n_episode}

	for episode in range(numEpisodes):
		status = ["IN_GAME","IN_GAME","IN_GAME"]
		observation = MARLEnv.reset()
		totalReward = 0.0
		timeSteps = 0
		while status[0]=="IN_GAME":
			for agent in agents:
				learningRate, epsilon = agent.computeHyperparameters(numTakenActions, episode)
				agent.setEpsilon(epsilon)
				agent.setLearningRate(learningRate)
			actions = []
			stateCopies = []
			for agentIdx in range(args.numAgents):
				obsCopy = deepcopy(observation[agentIdx])
				stateCopies.append(obsCopy)
				agents[agentIdx].setState(agents[agentIdx].toStateRepresentation(obsCopy))
				actions.append(agents[agentIdx].act())

			nextObservation, reward, done, status = MARLEnv.step(actions)
			numTakenActions += 1

			for agentIdx in range(args.numAgents):
				oppoActions = actions.copy()
				del oppoActions[agentIdx]
				agents[agentIdx].setExperience(agents[agentIdx].toStateRepresentation(stateCopies[agentIdx]), actions[agentIdx], oppoActions,
					reward[agentIdx], status[agentIdx], agent.toStateRepresentation(nextObservation[agentIdx]))
				agents[agentIdx].learn()

			observation = nextObservation
		status_lst.append(status[0])
		n_episode.append(numTakenActions - numTakenActionCKPT)
		numTakenActionCKPT = numTakenActions
		if episode % 100 == 0:
			print('action number %d, episode numer %d' % (numTakenActions, episode))
	describe(data)
 	#describe(data)
