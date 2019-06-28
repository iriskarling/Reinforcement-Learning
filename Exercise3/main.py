#!/usr/bin/env python3
# encoding utf-8
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
from Networks import ValueNetwork
from SharedAdam import SharedAdam
from Worker import train

# Use this script to handle arguments and
# initialize important components of your experiment.
# These might include important parameters for your experiment, and initialization of
# your models, torch's multiprocessing methods, etc.
if __name__ == "__main__" :

	# Example on how to initialize global locks for processes
	# and counters.

	#counter = mp.Value('i', 0)
	#lock = mp.Lock()

	# Example code to initialize torch multiprocessing.
	#for idx in range(0, args.num_processes):
	#	trainingArgs = (idx, args, value_network, target_value_network, optimizer, lock, counter)
	#	p = mp.Process(target=train, args=())
	#	p.start()
	#	processes.append(p)
	#for p in processes:
	#	p.join()
	parser = argparse.ArgumentParser()
	parser.add_argument('--epochs', type=int, default=10000000, metavar='N',
                    help='number of epochs to train (default: 10)')
	parser.add_argument('--lr', type=float, default=0.0001, metavar='LR',
                    help='learning rate (default: 0.01)')
	parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
	parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
	parser.add_argument('--num-processes', type=int, default=8)
	parser.add_argument('--discountFactor', type=float, default=0.9)
	parser.add_argument('--target-interval', type=int, default=2000)
	parser.add_argument('--predict-interval', type=int, default=250)
	parser.add_argument('--timesteps', type=int, default=10000000)
	parser.add_argument('--tmax', type=float, default=32e6)
	parser.add_argument('--epsilon', type=float, default=0.9)
	parser.add_argument('--save-interval', type=int, default=100000)




	args=parser.parse_args()
	mp.set_start_method('spawn')
	value_network= ValueNetwork(68,[15,15],4)
	target_value_network= ValueNetwork(68,[15,15],4)
	target_value_network.load_state_dict(value_network.state_dict())
	value_network.share_memory()
	target_value_network.share_memory()
	counter = mp.Value('i', 0)
	lock = mp.Lock()
	optimizer = SharedAdam(value_network.parameters())
	#optimizer.share_memory()
	processes = []
	rank = 0
	for idx in range(0, args.num_processes):
		trainingArgs = (idx, args, value_network, target_value_network, optimizer, lock, counter)
		p = mp.Process(target=train, args=(idx, args, value_network, target_value_network, optimizer, lock, counter))
		rank += 1
		p.start()
		processes.append(p)
	for p in processes:
		p.join()
