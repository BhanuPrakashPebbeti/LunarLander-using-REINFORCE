import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
import matplotlib.pyplot as plt
import numpy as np
import gym
import os

ROOT_DIR = os.path.dirname(__file__)
VALIDATING = True
PRETRAINED = True

class Agent:
    def __init__(self, ALPHA=0.1, GAMMA=0.99, n_actions=4, layer1_size=16,
                 layer2_size=16, input_dims=8, name='reinforce.pt'):
        self.lr = ALPHA
        self.gamma = GAMMA
        self.G = 0
        self.n_actions = n_actions
        self.state_memory = []
        self.action_memory = []
        self.reward_memory = []
        self.log_prob_memory = []
        self.policy = Policy_Network(self.lr, input_dims, n_actions, layer1_size, layer2_size)
        self.action_space = [i for i in range(n_actions)]
        self.model_file = name

    def choose_action(self, obs):
        obs = torch.tensor(obs).float()
        probs = self.policy(obs)
        action = np.random.choice(self.n_actions, p = np.squeeze(probs.cpu().detach().numpy()))
        log_prob = torch.log(probs.squeeze(0)[action])
        return action, log_prob

    def store_transition(self, observation, action, reward, log_prob):
        self.state_memory.append(observation)
        self.reward_memory.append(reward)
        self.action_memory.append(action)
        self.log_prob_memory.append(log_prob)

    def learn(self):
        policy_gradients = []
        discounted_rewards = []
        for t in range(len(self.reward_memory)):
            Gt = 0
            discount = 1
            for i in range(t, len(self.reward_memory)):
                Gt += discount * self.reward_memory[i]
                discount *= self.gamma
            discounted_rewards.append(Gt)
        discounted_rewards = torch.tensor(discounted_rewards)
        discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std() + 1e-9)
        for prob, Gt in zip(self.log_prob_memory, discounted_rewards):
            policy_gradients.append(-prob * Gt)
        self.policy.optimizer.zero_grad()
        policy_gradients = torch.stack(policy_gradients).sum()
        policy_gradients.backward()
        self.policy.optimizer.step()
        self.state_memory.clear()
        self.action_memory.clear()
        self.reward_memory.clear()
        self.log_prob_memory.clear()

    def save(self):
        torch.save(self.policy.state_dict(), os.path.realpath(os.path.join(ROOT_DIR, self.model_file)))

    def load(self):
        self.policy.load_state_dict(torch.load(os.path.realpath(os.path.join(ROOT_DIR, self.model_file))))


class Policy_Network(nn.Module):
    def __init__(self, lr, input_dims, n_actions, fc1=256, fc2=128):
        super(Policy_Network, self).__init__()
        self.input_dims = input_dims
        self.n_actions = n_actions

        self.fc1 = fc1
        self.fc2 = fc2

        self.fc1_l = nn.Linear(self.input_dims, self.fc1)
        self.fc2_l = nn.Linear(self.fc1, self.fc2)
        self.final = nn.Linear(self.fc2, self.n_actions)

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.optimizer = optim.Adam(self.parameters(), lr = lr)
        self.to(self.device)

    def forward(self, X):
        X = X.float().to(self.device)
        X = F.relu(self.fc1_l(X))
        X = F.relu(self.fc2_l(X))
        X = F.softmax(self.final(X), dim = -1)
        return X


agent = Agent(ALPHA = 0.0001, GAMMA = 0.99, n_actions = 4, layer1_size = 64,
              layer2_size = 64, input_dims = 8, name = 'reinforce.pt')

if VALIDATING or PRETRAINED:
    agent.load()
    print("Pre-Trained Weights Loaded")

env = gym.make('LunarLander-v2')
scores = []
EPISODES = 20000
for i in range(10000,EPISODES):
    done = False
    score = 0
    state = env.reset()
    while not done:
        action, log_prob = agent.choose_action(state)
        state_, reward, done, info = env.step(action)
        if not VALIDATING:
            if i % 250 == 0:
                env.render()
            agent.store_transition(state, action, reward, log_prob)
        else:
            env.render()
        state = state_
        score += reward
    if not VALIDATING:
        scores.append(score)
        agent.learn()
        if i % 100 == 0:
            agent.save()
        print("Episode :", i, "Score :", score, "Average Score :", np.mean(scores[-100:]))
env.close()

if not VALIDATING:
    plt.plot([i for i in range(EPISODES)], scores, label = "Rewards")
    plt.legend(loc = 4)
    plt.savefig(os.path.realpath(os.path.join(ROOT_DIR, "Rewards.png")))
    plt.show()
