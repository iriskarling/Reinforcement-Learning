#!/usr/bin/env python3
# encoding utf-8

import random
import argparse
from DiscreteMARLUtils.Environment import DiscreteMARLEnvironment
from DiscreteMARLUtils.Agent import Agent
from copy import deepcopy
from collections import defaultdict
import numpy as np
import math
class IndependentQLearningAgent(Agent):
	def __init__(self, learningRate, discountFactor, epsilon, initVals=0.0):
		super(IndependentQLearningAgent, self).__init__()
		self.learningRate = learningRate
		self.discountFactor = discountFactor
		self.epsilon = epsilon
		self.q_table = {}
		self.A = ['MOVE_UP','MOVE_DOWN','MOVE_LEFT','MOVE_RIGHT','KICK','NO_OP']
		self.reward = 0.0
		self.status = 0
		self.state =[[[0,0],[0,0]],[[0,0]],[[0,0]]]
		self.nextState =[[[0,0],[0,0]],[[0,0]],[[0,0]]]
		self.action = "KICK"

	def setExperience(self, state, action, reward, status, nextState):
		self.state = state
		self.action = action
		self.reward = reward
		self.status = status
		self.nextState = nextState
		if nextState not in self.q_table:
			self.q_table[nextState]= [0.0,0.0,0.0,0.0,0.0,0.0]

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
		if np.random.uniform() < self.epsilon:
			state_action = self.q_table[self.state]
			for i in range(6):
				if state_action[i] == np.max(state_action):
					temp.append(self.A[i])
			action = np.random.choice(temp)
		else:
			action = np.random.choice(self.A)
		return action


	def toStateRepresentation(self, state):
		temp =[]
		for i in range(3):
		    temp1 = []
		    a =0
		    for k in state[i]:
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
			self.q_table[state_t]= [0.0,0.0,0.0,0.0,0.0,0.0]
		return state_t

	def setState(self, state):
		self.state = state

	def setEpsilon(self, epsilon):
		self.epsilon = epsilon

	def setLearningRate(self, learningRate):
		self.learningRate = learningRate

	def computeHyperparameters(self, numTakenActions, episodeNumber):
		lr_s = 0.5
		lr_e = 0.05
		lr_d = 3e5
		eps_s = 1
		eps_e = 0.01
		eps_d =3e6 ## 3E5
		self.learningRate = lr_e + (lr_s - lr_e) * np.exp(-1 * numTakenActions / lr_d)
		self.epsilon = eps_e + (eps_s - eps_e)* np.exp(-1 * numTakenActions / eps_d)
		self.epislon = max(math.exp(-1*6/2000*episodeNumber),0.1)#0.1
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
    numOpponents = 1
    numAgents = 2
    MARLEnv = DiscreteMARLEnvironment(numOpponents=numOpponents, numAgents=numAgents)
    agents = []
    for i in range(numAgents):
        agent = IndependentQLearningAgent(learningRate=0.1, discountFactor=0.98, epsilon=1.0)
        agents.append(agent)

    numEpisodes = 50000
    numTakenActions = 0
    numTakenActionCKPT = 0

    status_lst = []
    n_episode = []
    data = {'status': status_lst, 'steps_in_episode': n_episode}

    for episode in range(numEpisodes):
        status = ["IN_GAME","IN_GAME","IN_GAME"]
        observation = MARLEnv.reset()
        totalReward = 0.0
        timeSteps = 0

        while status[0] == "IN_GAME":
            for agent in agents:
                learningRate, epsilon = agent.computeHyperparameters(numTakenActions, episode)
                agent.setEpsilon(epsilon)
                agent.setLearningRate(learningRate)
            actions = []
            stateCopies, nextStateCopies = [], []
            for agentIdx in range(numAgents):
                obsCopy = deepcopy(observation[agentIdx])
                stateCopies.append(obsCopy)
                agents[agentIdx].setState(agents[agentIdx].toStateRepresentation(obsCopy))
                actions.append(agents[agentIdx].act())
            numTakenActions += 1
            nextObservation, reward, done, status = MARLEnv.step(actions)

            for agentIdx in range(numAgents):
                agents[agentIdx].setExperience(agent.toStateRepresentation(stateCopies[agentIdx]), actions[agentIdx], reward[agentIdx],
                    status[agentIdx], agent.toStateRepresentation(nextObservation[agentIdx]))
                agents[agentIdx].learn()

            observation = nextObservation

        status_lst.append(
            status[0])
        n_episode.append(numTakenActions - numTakenActionCKPT)
        numTakenActionCKPT = numTakenActions

        if episode % 100 == 0:
            print('action number %d, episode numer %d' % (numTakenActions, episode))

    describe(data)
