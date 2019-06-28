#!/usr/bin/env python3
# encoding utf-8

import random
import argparse
from DiscreteMARLUtils.Environment import DiscreteMARLEnvironment
from DiscreteMARLUtils.Agent import Agent
from copy import deepcopy
import numpy as np
		
class WolfPHCAgent(Agent):
	def __init__(self, learningRate, discountFactor, winDelta=0.01, loseDelta=0.1, initVals=0.0):
		super(WolfPHCAgent, self).__init__()
		self.learningRate = learningRate
		self.discountFactor =discountFactor
		self.winDelta = winDelta
		self.loseDelta =loseDelta
		self.initVals = initVals
		self.A = ['MOVE_UP','MOVE_DOWN','MOVE_LEFT','MOVE_RIGHT','KICK','NO_OP']
		self.reward = 0.0
		self.status = 0
		self.state =[[[0,0],[0,0]],[[0,0]],[[0,0]]]
		self.nextState =[[[0,0],[0,0]],[[0,0]],[[0,0]]]
		self.action = ""
		self.q_table = {}
		self.a_table = {}
		self.avg_table ={}
		self.action = ""
		self.epsilon = 0.9
		self.c = {}
		
		
	def setExperience(self, state, action, reward, status, nextState):
		self.state = state
		self.action = action
		self.reward = reward
		self.status = status
		self.nextState = nextState

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

	def calculateAveragePolicyUpdate(self):
		self.c[self.state] += 1.0
		temp =[(x-y)/self.c[self.state] for x,y in zip(self.a_table[self.state],self.avg_table[self.state])]
		self.avg_table[self.state] = [a+b for a,b in zip(temp,self.avg_table[self.state])]
		return self.avg_table[self.state]

	def calculatePolicyUpdate(self):
		Q_max = np.max(self.q_table[self.state])
		a_suboptimal = []
		a_not_suboptimal = [0,1,2,3,4,5]
		for i in range(len(self.A)):
			if self.q_table[self.state][i] != Q_max:
				a_suboptimal.append(i)
				a_not_suboptimal.remove(i)
		summ = sum(x*y for x,y in zip(self.a_table[self.state],self.q_table[self.state]))
		summ_avg = sum(a*b for a,b in zip(self.avg_table[self.state], self.q_table[self.state]))
		if summ >= summ_avg:
			lr = self.winDelta
		else:
			lr = self.loseDelta
		p_moved = 0
		for i in a_suboptimal:
			p_moved += min(lr/len(a_suboptimal),self.a_table[self.state][i])
			self.a_table[self.state][i] -= min(lr/len(a_suboptimal),self.a_table[self.state][i])
		for j in a_not_suboptimal:
			self.a_table[self.state][j] += p_moved/(len(self.A)-len(a_suboptimal))



		return self.a_table[self.state]

	
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
			self.a_table[state_t]= [1.0/6.0,1.0/6.0,1.0/6.0,1.0/6.0,1.0/6.0,1.0/6.0]
			self.avg_table[state_t]= [1.0/6.0,1.0/6.0,1.0/6.0,1.0/6.0,1.0/6.0,1.0/6.0]
			self.c[state_t]= 0.0
		return state_t

	def setState(self, state):
		self.state = state

	def setLearningRate(self,lr):
		self.learningRate = lr
		
	def setWinDelta(self, winDelta):
		self.winDelta = winDelta
		
	def setLoseDelta(self, loseDelta):
		self.loseDelta = loseDelta
	
	def computeHyperparameters(self, numTakenActions, episodeNumber):
		lr_s = 0.75
		lr_e = 0.0075
		lr_d = 2e5
		eps_s = 1e-3
		eps_e= 1e-4
		eps_d =2e5
		self.learningRate = lr_e + (lr_s - lr_e) * np.exp(-1 * numTakenActions / lr_d)
		self.loseDelta = eps_e + (eps_s - eps_e)* np.exp(-1 * numTakenActions / eps_d)
		self.winDelta = self.loseDelta
		return self.loseDelta,self.winDelta,self.learningRate

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
	parser.add_argument('--numEpisodes', type=int, default=100000)

	args=parser.parse_args()

	numOpponents = args.numOpponents
	numAgents = args.numAgents
	MARLEnv = DiscreteMARLEnvironment(numOpponents = numOpponents, numAgents = numAgents)

	agents = []
	for i in range(args.numAgents):
		agent = WolfPHCAgent(learningRate = 0.2, discountFactor = 0.99, winDelta=0.01, loseDelta=0.1)
		agents.append(agent)
	numEpisodes = 50000
	numTakenActions = 0
	numTakenActionCKPT = 0
	status_lst = []
	n_episode = []
	data = {'status':status_lst,'steps_in_episode':n_episode}

	for episode in range(numEpisodes):	
		status = ["IN_GAME","IN_GAME","IN_GAME"]
		observation = MARLEnv.reset()
		
		while status[0]=="IN_GAME":
			for agent in agents:
				loseDelta, winDelta, learningRate = agent.computeHyperparameters(numTakenActions, episode)
				agent.setLoseDelta(loseDelta)
				agent.setWinDelta(winDelta)
				agent.setLearningRate(learningRate)
			actions = []
			perAgentObs = []
			agentIdx = 0
			for agent in agents:
				obsCopy = deepcopy(observation[agentIdx])
				perAgentObs.append(obsCopy)
				agent.setState(agent.toStateRepresentation(obsCopy))
				actions.append(agent.act())
				agentIdx += 1
			nextObservation, reward, done, status = MARLEnv.step(actions)
			numTakenActions += 1

			agentIdx = 0
			for agent in agents:
				agent.setExperience(agent.toStateRepresentation(perAgentObs[agentIdx]), actions[agentIdx], reward[agentIdx], 
					status[agentIdx], agent.toStateRepresentation(nextObservation[agentIdx]))
				agent.learn()
				agent.calculateAveragePolicyUpdate()
				agent.calculatePolicyUpdate()
				agentIdx += 1
			
			observation = nextObservation
		status_lst.append(status[0])
		n_episode.append(numTakenActions - numTakenActionCKPT)
		numTakenActionCKPT = numTakenActions
		if episode % 100 == 0:
			print('action number %d, episode numer %d' % (numTakenActions, episode))
	describe(data)
