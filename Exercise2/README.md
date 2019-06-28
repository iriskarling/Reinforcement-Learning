# Exercise 2- SARSA Methods

You are required to implement the SARSA algorithm using the codes provided in `SARSABase.py`. Before you proceed,install the HFO Domain first by following the necessary steps outlined in `Exercise2/README.md`. Your task is to extend `SARSABase.py` by implementing functions that have yet been implemented in `SARSABase.py`.

## Specifications
### Automarking requirements
To ensure that your codes can be used by the marking script, ensure that all the necessary functions have been implemented. To check whether these implementations are correct, use the code snippet given in `__main__` to test your implementation. This code snippet gives an outline on how your implemented functions will interact to train a SARSA agent. Additionally, a similar sequence of commands is used in the automarking tools. In general, If you just add some code to modify number of episodes, you can use this code to train a SARSA agent.

Additionally, **although the state-action values can be initialized using any value, you are to initialize the values as zeros for every state-action pair for automarking.** 

### Implemented Functions
#### `__init__(self, learningRate, discountFactor, epsilon)`
This init function should initialize all the necessary parameters for training a SARSA Agent. This includes the learning rate, discount factor, and the epsilon value (if you use an epsilon greedy exploration strategy). This function will only be called once at the very beginning when you initialize agents for training.

#### `learn()` - Used in Automarking
This is the most important function you need to implement in this task. This function has no input parameters. On the other hand, this method **must** return a single scalar that specifies the change **(value after update subtracted by value before update)** in updated state-action value after you've trained your agents using SARSA's update. This function will be used in the automarker to compare the correctness of your implementation against the solution. **This function has the same functionality as line 9 in the books' SARSA pseudocode**


#### `act()`
This function will be used to choose the actions that your agents will use when faced with a state. It should only return the action that should be taken by the agent at the current state. **This function has the same functionality as line 5 in the books' SARSA pseudocode**
#### `setState(state)`
This function will be used to provide the agents you're controlling with the current state information. It will receive the state representation from the environment as an input. On the other hand, this does not need to output anything.

#### `setExperience(state, action, reward, status, nextState)`
Once an agent executes an action, it will receive the rewards, status, and next states resulting from that action. Use this method to set these data to prepare your agent to learn using the SARSA update. **This function has the same functionality as line 10 in the books' SARSA pseudocode**

#### `toStateRepresentation(state)`
You might want to use a different representation compared to the ones provided by the environment. This will provide a problem to the automarker. Therefore, you should implement a function that maps the raw state representation into the the state representation that you are using in your implementation. This function will receive a state and outputs it's value under the representations that you are using in your implementations.

#### `reset()`
You might want to reset some states of an agent at the beginning of each episode. Use this function to do that. This function does not require any inputs. Additionally, it also does not provide any outputs.

#### `setLearningRate(learningRate)` and `setEpsilon(epsilon)`
This function should be used to set the learning rate and the epsilon (if you are using epsilon greedy) that you use during training. 

#### `computeHyperparameters(numTakenActions, episodeNumber)`

This function should return a tuple indicating the learning rate and epsilon used at a certain timestep. This allows you to schedule the values of your hyperparameters and change it midway of training.

### Training process
To see how your implemented function interact with each other to train the agent, check the `__main__` function inside `SARSABase.py`. Make sure that you can successfully train your agent using the codes inside `__main__` to ensure that your implementations are correct. A similar sequence of commands with those provded in`__main__` is also going to be used in the marking process.

## Marking details
### Performance marking
Using similar codes as what you've seen in `__main__`, we are going to run your agent on a randomly sampled MDP and compare it's performance to our solution. For details on the experiment, refer to the **Marking** section in Exercise 2's README.

### Unit test marking
We compare the results of updates from `learn()`. This function should return the difference between the value of the updated state-action pair after and before the update. E.g, (Q(s_t,a_t)(t+1) - Q(s_t,a_t)(t)) 

As an example, let's say that an agent is exposed to the following sequence of experience:
```
Timestep Number, State, Action, Reward, Next State
1, ((1,1),(2,1)), MOVE_RIGHT, -0.4, ((2,1),(2,1))
2, ((2,1),(2,1)), MOVE_LEFT, 0.0, ((1,1),(2,1))
3, ((1,1),(2,1)), KICK, 1.0, GOAL
```

Assuming an initial value of 0 for each state-action pair, a learning rate of 0.1 and a discount rate of 1, these should be the outputs of the learn functions at the end of each timestep :

```
<<<<<<< HEAD
Timestep Number, learn Output
1, N/A (Because learn is not called in the first timestep)
2, -0.04 (Calculate the difference in value of < ((1,1),(2,1)), MOVE_RIGHT > before and after update )
3, 0.0 (Calculate the difference in value of < ((2,1),(2,1)), MOVE_LEFT > before and after update )
4 (The final Update), 0.1 (Calculate the difference in value of < ((1,1),(2,1)), KICK > before and after update)
```
=======
# Move to the example directory in your HFO folder and clone the latest code base
cd HFO/example
git clone https://github.com/raharrasy/RL2019-BaseCodes.git
# Move to the directory that contains the code base for the algorithm you'd like to implement
cd RL2019-BaseCodes/Exercise2/<Algorithm Code Base Directory>
cp * ..
cd ..
./<Caller for desired algorithm>.sh
```  

## Marking
Marking will be based on the correctness of your implementations and the performance of your agents. 

To examine the correctness of the implementations, we will require you to implement functions that output specific values related to the algorithm being implemented. To find these functions and what they are supposed to output, refer to the README files inside each specific algorithm that you are supposed to implement. Additionally, we've also provided a small section of code in the **main** functions in each python files to provide information on how the functions are supposed to interact. You **must implement your agents such that the sequence of commands in the main function can train your agents**.

On the **performance marking**, we will do several experiments under the same MDP where we **run the agents for 5000 episodes** using commands that are similar to what has been provided in the **main functions**. **In these experiments, we guarantee that the size of the grid will be 6x5, there will only be a single defender, and the location of the defender does not change across the 5000 episodes**. Additionally, we also guarantee that **in each experiment we will store the performance of your agents in episodes that are divisible by 500 and average this value across experiments. We will then make a plot of the agent performance and compare it with our solutions. In 5000 episodes, given good hyperparameter settings, your agents should be able to reach performance that is close to optimal.**

## Additional Information

### Implemented Files (**Contains functions to be implemented**)
1. `QLearning/QLearningBase.py`
2. `MonteCarlo/MonteCarloBase.py`
3. `SARSA/SARSABase.py`

### Environment Files (**Should not be modified**)
1. `DiscreteHFO/HFOAttackingPlayer.py`
   - File to establish connections with HFO and preprocess state representations gathered from the HFO domain.
2. `DiscreteHFO/HFODefendingPlayer.py`
   - File to control defending player inside the HFO environment. 
3. `DiscreteHFO/HFOGoalkeepingPlayer.py`
   - File to control Goalkeeper inside the HFO environment. HFO environment cannot run without a goalkeeper. 
4. `DiscreteHFO/DiscretizedDefendingPlayer.py`
   - File to initialize the defending player.
5. `DiscreteHFO/Goalkeeper.py`
   - File to initialize the Goalkeeper.
   
### Caller Files (**Can be modified, adapt to your existing directories if necessary**)
1. `QLearning/QLearningAgent.sh`
   - This file runs all the necessary files to initialize a discrete HFO domain and run a Q-Learning agent.
2. `SARSA/SARSAAgent.sh`
   - This file runs all the necessary files to initialize a discrete HFO domain and run a SARSA agent.
3. `MonteCarlo/MonteCarlo.sh`
   - This file runs all the necessary files to initialize a discrete HFO domain and run a Monte Carlo agent.

## Environment Details
   
### State Space
The environment is modelled as a 6x5 grid. The grid cell with `(0,0)` coordinate is located in the top left part of the field. At each timestep, the agent will be given a state representation, in the form of a list, which has information on the defending players' location and the agent's own location on the grid. The first item in the list is the agent's location and the rest are the location of the defending players. 

The location of the goal is not modelled inside the grid. Therefore, agents cannot dribble into the goal and must rely on the `KICK` action to score goals. 

### Action Spaces
Agents are equipped with a set of discrete actions. To move to adjacent grids, agents can use the `DRIBBLE_UP`,`DRIBBLE_DOWN`,`DRIBBLE_LEFT`, and `DRIBBLE_RIGHT` actions. Additionally, the `KICK` action enables the agents to shoot the ball toward the goal. 

### Reward Functions
Agents only receive non-zero rewards at the completion of each episode. In this case, a goal will result in a reward of **+1**. However, occupying the same grid as defending players will result in a penalty.

### Environment Dynamics
Environment transitions resulting from the actions are stochastic. For the dribbling actions, there will be a small probability for agents to end up dribbling into an adjacent (but wrong) grid. There is also the possibility of agent's kicks going wayward from the goal after executing the `KICK` action. This probability of kicking the ball depends on the location in the grid that the agent executes the `KICK` action from.

## Status
Status are integers that denote certain terminal events in the game. It will always be 0 when the agents are in the middle of a game. Other numbers might denote different events like a goal successfully scored, ball kicked out of bounds, or episodes running out of time. Full information of the possible status values for the HFO environment can be seen at `HFO/bin/HFO` script inside the HFO codes given in the original HFO repository.
>>>>>>> 9b22e4e0f6802c09845651b79cac602f675c6942

