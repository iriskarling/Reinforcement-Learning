import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from Networks import ValueNetwork
from torch.autograd import Variable
from Environment import HFOEnv
from SharedAdam import SharedAdam
import random
import numpy as np

def train(idx, args, value_network, target_value_network, optimizer, lock, counter):
	port = 3020+idx*10
	seed = 12+idx*10
	counter_v = 0
	hfoEnv = HFOEnv(numTeammates = 0, numOpponents =1, port = port, seed = seed)
	hfoEnv.connectToServer()
	loss = nn.MSELoss()
	for episodeNumber in range(1,args.epochs+1):
		observation = hfoEnv.reset()
		counter.value += 1
		counter_v += 1

		for timestep in range(args.timesteps):
			#observation_t = torch.Tensor(observation)
			#action = greedy_action(observation_t,value_network,args)
			#print('!!!!!!!!!!!!!!',action)
			#act = hfoEnv.possibleActions[action]
			action = random.randint(0,3)
			act = hfoEnv.possibleActions[action]
			newObservation, reward, done, status, info = hfoEnv.step(act)
			#print(newObservation,reward,done,status,info)                                
			optimizer.zero_grad()
			lock.acquire()

			observation_t = torch.Tensor(observation)
			newObservation_t = torch.Tensor(newObservation)
			output = computePrediction(observation_t,action,value_network)
			target = computeTargets(reward,newObservation_t,args.discountFactor,done,target_value_network)
			lock.release()
			out = loss(output,target)
			out.backward()
			if counter_v% args.target_interval == 0:
				target_value_network.load_state_dict(value_network.state_dict())
				
			if counter.value % args.predict_interval == 0 or done:
				optimizer.step()
				optimizer.zero_grad()
				strDirectory = "value_network"+str(idx)+".pth"
				saveModelNetwork(value_network,strDirectory)
				strDirectory_targert = "targetNetwork"+str(idx)+".pth"
				saveModelNetwork(target_value_network,strDirectory_targert)
			observation = newObservation
			if done:
				break
			if counter.value >= args.tmax:
				break


def computeTargets(reward, nextObservation, discountFactor, done, targetNetwork):

	value = targetNetwork(nextObservation)
	if done == False:
		q_target = reward + discountFactor* torch.max(value)
	else:
		q_target = torch.FloatTensor([reward])
	
	return q_target

def computePrediction(state, action, valueNetwork):
	value =  valueNetwork(state)
	return value[0][action]
	
# Function to save parameters of a neural network in pytorch.
def saveModelNetwork(model, strDirectory):
	torch.save(model.state_dict(), strDirectory)
	


def greedy_action(state,valueNetwork,args):

	if np.random.uniform()< 0.9:
		return random.randint(0,3)
	else:
		value = valueNetwork(state)
		action_num = torch.argmax(value[0])
		
		return action_num
		


