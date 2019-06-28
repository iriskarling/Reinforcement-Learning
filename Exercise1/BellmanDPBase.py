from MDP import MDP
import numpy as np


class BellmanDPSolver(object):
	def __init__(self, discountRate):
		self.MDP = MDP()
		self.dr = discountRate
		self.S = [(x,y) for x in range(5) for y in range(5)]
		self.S.append("GOAL")
		self.S.append("OUT")
		self.A = ["DRIBBLE_UP","DRIBBLE_DOWN","DRIBBLE_LEFT","DRIBBLE_RIGHT","SHOOT"]
		self.oppositions = [(2,2), (4,2)]
		self.goalProbs = [[0.00,0.00,0.0,0.00,0.00],[0.0, 0.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.0,0.0],[0.0,0.3,0.5,0.3,0.0],[0.0,0.8,0.9,0.8,0.0]]
		self.vs ={}
		self.act = {}


	def initVs(self):
		for s in self.S:
			self.vs[s] = 0.0

	def BellmanUpdate(self):
		for s in self.S:
			temp = -100000000
			action = " "
			value_max= []
			action_max = []
			for a in self.A:
				nextState = self.MDP.probNextStates(s,a)
				value_a = 0.0
				for s_next,prob in nextState.items():
					reward = self.MDP.getRewards(s,a,s_next)
					value_next = self.vs[s_next]
					value_a += prob*(reward + self.dr*value_next)
				value_max.append(value_a)
			self.vs[s]= np.max(value_max)
			for i in range(len(self.A)):
				if value_max[i] == np.max(value_max):
					action_max.append(self.A[i])			
			self.act[s] = action_max
		return self.vs,self.act
		
if __name__ == '__main__':
	solution = BellmanDPSolver(0.9)
	solution.initVs()
	for i in range(20000):
		values, policy = solution.BellmanUpdate()
	print("Values : ", values)
	print("Policy : ", policy)

