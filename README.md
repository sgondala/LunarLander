# Lunar Lander 

## Instructions to run - 

Dependencies - gym, numpy, tensorflow, matplotlib

I have already trained the model and saved it under weights/.
To test the code, run 'python3 reinforce_tf.py' 

When you execute it, the program starts in test mode by default. 
It loads the trained weights into the NN and runs 100 trails using this trained NN and prints the score values for those 100 runs.

If you want to train it, open the file, uncomment the last line and comment out 3 lines above it.

In this project, I created an agent and trained it to successfully play the Lunar Lander game from OpenAI Gym.

### A bit about the env:

This game consists of a rocket trying to land on the moon. Input is a continuous 8 dimensional space. There are 4 discrete actions available. Each action incurs a rewards. The final goal is to land the rocket right and get a total reward of at least 200. 

### Solving Methodology:

There are several ways to solve this problem. 

**Discretization** - We discretize continuous state space and we would then end up with a finite number of discrete states. We can apply Q Learning on the discrete states to solve this problem. However, there are multiple issues here 
Say we divide first 6 attributes of a state into discrete components. First 2 are range values - x,y. Say we divide them into 200 parts. Next 2 are velocity components. Let’s divide them into 20 parts each. Next two are angle values - Say we divide them into 360 parts each, each part for each angle. The total number of states we have here is 200 * 200 * 20 * 20 * 360 * 360 * 2 * 2 ~ 10^13 states. It is not computationally feasible to apply Q Learning
Information can also be lost in discretization and could result in mishandling of edge cases


**DQN and DDQN** - DQN, DDQN are value based RL approaches. I’ve coded it out using Keras and I was able to successfully converge a CartPole problem. However, when it came to Lunar Landing, I had issues with convergence. I tried to dig deeper into it and did a lot of optimizations but was unable to converge in short time. 


**REINFORCE** - Monte-Carlo Policy Gradient, also known as REINFORCE, is a Policy Based RL algorithm. I finally chose this over DQN because policy based approaches have better convergence properties.

### Training method:

- I created a 2 hidden layer neural network which takes input as state/observation vector and output as a 4 column vector, where each attribute correspond to each action. Size of each hidden layer is 16 neurons. I initially assigned it with random weights.
- Softmax of output gives us the probability with which a particular action should be chosen.
- For each episode, we start with an observation, feed the observation into NN and take an action in accordance with the output probabilities. We then store input state, action taken and reward observed for training. We keep taking actions until the episode is finished
- Once the episode is finished, we train the NN using the (observation, action, reward) values stored from the episode. As we’ve given out custom loss function, NN applies gradient descent in order to minimize our loss function 
- We keep iterating this process of playing, training until we get a series of consecutive 100 episodes, whose mean reward value is greater than 200. In this case, it took me ~3500 episodes.
- There are only 2 hyper parameters here - Gamma used to calculate the G Value and alpha, which is learning rate of the neural network.

## Analysis

### Training Graphs

All these training graphs have the same hyper parameters -  Gamma = 0.99, Alpha = 0.005

**Scores per episode** - This graph denotes the score obtained per each episode. We can see that score value obtained in the each episode started from around -600 and as the model kept learning, it reached upto 200 and 250 (250 is clear from the next graph)

![alt text](https://github.com/sgondala/LunarLander/blob/master/images/1.PNG)

**Total score per episode for last 100 episodes**  - This graph is same as above, but zoomed in to reveal only the total scores of last 100 episodes before the algorithm converged. As we can see, the total score in each episode went upto 250. 

![alt text](https://github.com/sgondala/LunarLander/blob/master/images/2.PNG)

**Mean score over the training process** - In this graph, f(x) denote the mean total scores for 100 episodes in the range [x*100-99, x*100]

![alt text](https://github.com/sgondala/LunarLander/blob/master/images/3.PNG)

### Test Graph 

Using the trained model, I ran a set of 100 trials and computed the score value for each of 100 episodes. This is the plot for the same

The mean score value for 100 trails is 187.34

![alt text](https://github.com/sgondala/LunarLander/blob/master/images/4.PNG)


**Effect of varying Alpha with optimal gamma = 0.99**

We can see that Alpha = 0.1 and 0.3 perform really bad. This is the range of Alphas where learning rate is higher than optimal and does not converge at all.  While alpha = 0.005 performs optimally, we can see that alpha = 0.001 is also on the path of convergence and just requires bit more learning

![alt text](https://github.com/sgondala/LunarLander/blob/master/images/5.PNG)

**Effect of varying Gamma with optimal Alpha = 0.005**

From the graph we can see that gamma around 0.99 is the ideal range. When gamma = 1, it suffers from having to look ahead too much into the future and gamma = 0.95 also is not optimal because it is not looking “enough” hence learning less than what it could actually learn

![alt text](https://github.com/sgondala/LunarLander/blob/master/images/6.PNG)
