import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.distributions as distributions

import matplotlib.pyplot as plt
import numpy as np
import gymnasium as gym

# as mentioned in the report
# this file not only defines the
# network and associated functions
# but also does the network training
# which was not smart
# but it doesn't seem to be worth fixing

start = time.time()

# generate cartpole environments
train_env = gym.make('CartPole-v1')
test_env = gym.make('CartPole-v1')

SEED = 1234
np.random.seed(SEED);
torch.manual_seed(SEED);

# define neural network class
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout = 0.5):
        super().__init__()

        self.fc_1 = nn.Linear(input_dim, hidden_dim)
        self.fc_2 = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc_1(x)
        x = self.dropout(x)
        x = F.relu(x)
        x = self.fc_2(x)
        return x

# set neural network dimensions
INPUT_DIM = train_env.observation_space.shape[0]
HIDDEN_DIM = 128
OUTPUT_DIM = train_env.action_space.n

# create an instance of the neural network
policy = MLP(INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM)

# define weight initialization function
def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_normal_(m.weight)
        m.bias.data.fill_(0)

# initialize network weights
policy.apply(init_weights)

LEARNING_RATE = 0.01

# select optimizer
optimizer = optim.Adam(policy.parameters(), lr = LEARNING_RATE)

# define neural network training function
# following REINFORCE policy gradient method
def train(env, policy, optimizer, discount_factor):
    
    policy.train()
    log_prob_actions = []
    rewards = []
    done = False
    episode_reward = 0

    state = env.reset()[0]

    while not done:
        # get state from environment
        # then get action
        state = torch.from_numpy(state)
        action_pred = policy(state)
        action_prob = F.softmax(action_pred, dim = -1)
        dist = distributions.Categorical(action_prob)
        action = dist.sample()
        log_prob_action = dist.log_prob(action)

        # then get new state and reward
        state, reward, done, _, _ = env.step(action.item())

        log_prob_actions.append(log_prob_action)

        rewards.append(reward)

        episode_reward += reward
    log_prob_actions = torch.stack(log_prob_actions)

    # calculate returns from trajectory
    # do backpropagation and take gradient descent step
    returns = calculate_returns(rewards, discount_factor)
    loss = update_policy(returns, log_prob_actions, optimizer)

    return loss, episode_reward

# function for calculating discounted sum
# of trajectory returns
def calculate_returns(rewards, discount_factor, normalize = True):
    returns = []
    R = 0
    
    for r in reversed(rewards):
        R = r + R * discount_factor
        returns.insert(0, R)
        
    returns = torch.tensor(returns)

    if normalize:
        returns = (returns - returns.mean()) / returns.std()
        
    return returns

# function for gradient descent step
def update_policy(returns, log_prob_actions, optimizer):
    
    returns = returns.detach()
    loss = - (returns * log_prob_actions).sum()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()

# function to test policy
def evaluate(env, policy):
    
    policy.eval()
    done = False
    episode_reward = 0
    state = env.reset()[0]

    while not done:
        
        state = torch.from_numpy(state)
        with torch.no_grad():
            action_pred = policy(state)
            action_prob = F.softmax(action_pred, dim = -1)
        action = torch.argmax(action_prob, dim = -1)
        state, reward, done, _, _ = env.step(action.item())
        episode_reward += reward
        
    return episode_reward

# parameters for training and testing
MAX_EPISODES = 500
DISCOUNT_FACTOR = 0.99
N_TRIALS = 25
REWARD_THRESHOLD = 475
PRINT_EVERY = 10

train_rewards = []
test_rewards = []

# loop for training and testing
for episode in range(1, MAX_EPISODES+1):
    
    loss, train_reward = train(train_env, policy, optimizer, DISCOUNT_FACTOR)
    
    test_reward = evaluate(test_env, policy)
    
    train_rewards.append(train_reward)
    test_rewards.append(test_reward)
    
    mean_train_rewards = np.mean(train_rewards[-N_TRIALS:])
    mean_test_rewards = np.mean(test_rewards[-N_TRIALS:])
    
    if episode % PRINT_EVERY == 0:
    
        print(f'| Episode: {episode:3} | Mean Train Rewards: {mean_train_rewards:5.1f} | Mean Test Rewards: {mean_test_rewards:5.1f} |')
    
    if mean_test_rewards >= REWARD_THRESHOLD:
        
        print(f'Reached reward threshold in {episode} episodes')
        
        break

# plotting output
# (not included in report)
plt.figure(figsize=(12,8))
plt.plot(test_rewards, label='Test Reward')
plt.plot(train_rewards, label='Train Reward')
plt.xlabel('Episode', fontsize=20)
plt.ylabel('Reward', fontsize=20)
plt.hlines(REWARD_THRESHOLD, 0, len(test_rewards), color='r')
plt.legend(loc='lower right')
plt.grid()
plt.savefig(r"C:\Users\jesse\Downloads\STAT 548\code_and_output\policy_gradient.png")

# save trained policy
torch.save(policy.state_dict(), r"C:\Users\jesse\Downloads\STAT 548\code_and_output\policy.pth")

end = time.time()
print("Policy gradient done in", end - start, "seconds.")