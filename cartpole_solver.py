import gym
from tensorboardX import SummaryWriter

import torch
from torch import nn, optim
import numpy as np

BATCH_SIZE = 16

# Neural network for agent
class Agent(nn.Module):
    def __init__(self, obs_size, n_act) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_size, 128),
            nn.ReLU(),
            nn.Linear(128, n_act),
        )
    
    def forward(self, x):
        return self.net(x)


# Generate Training Data
def create_batch(env, agent, batch_size):
    '''
    Keeps on generating training batches all of size batch_size.
    call next(create_batch) to get next batch
    '''

    softmax = nn.Softmax()
    batch = []
    is_done = False
    while True:
        episode = []
        episode_reward = 0.0
        obs = env.reset()
        while not is_done:
            obs_tens = torch.FloatTensor([obs])
            # get action probabilities from agent
            with torch.no_grad():
                act_prob_tens = softmax(agent(obs_tens))
            act_prob = act_prob_tens.data.numpy()[0]
            # choose action using probabilities
            act = np.random.choice(len(act_prob), p=act_prob)
            # print(act)
            # act using action
            next_obs, reward, is_done, _ = env.step(act)
            episode_reward += reward
            # train using observation and chosen action
            episode.append((obs, act))
            obs = next_obs

        batch.append((episode_reward, episode))
        is_done = False
        if (len(batch) == batch_size):
            yield batch
            batch = []


# keep top episodes in each each batch
def filter_batch(batch, percentile=70):
    '''discard bottom 'percentile' episodes in batch'''
    rewards = [r for (r,e) in batch]
    # print(rewards)
    reward_thres = np.percentile(rewards, percentile)
    reward_mean = np.mean(rewards)

    obs = []
    act = []
    for ( r,e ) in batch:
        if r >= reward_thres:
            obs.extend([o for (o, _) in e])
            act.extend([a for (_, a) in e])
            
    obs_tens = torch.FloatTensor(obs)
    act_tens = torch.LongTensor(act)

    return obs_tens, act_tens, reward_thres, reward_mean

# Training 
def train(env, agent, batch_size):
    loss_func = nn.CrossEntropyLoss()
    optimizer = optim.Adam(agent.parameters(), lr=1e-2)

    writer = SummaryWriter(comment="cartpole")
    for i, batch in enumerate(create_batch(env, agent, batch_size)):
        optimizer.zero_grad()
        obs, targ_act, reward_thres, reward_mean = filter_batch(batch)
        pred_act = agent(obs)
        # print(pred_act)
        # print(targ_act)
        loss = loss_func(pred_act, targ_act)
        loss.backward()
        optimizer.step()

        print(f"epoch:{i}, reward threshold:{reward_thres}, mean reward in batch:{reward_mean}")
        writer.add_scalar("loss", loss.item(), i)
        writer.add_scalar("mean reward", reward_mean, i)
        
        if (reward_mean > 199):
            print("solved!")
            torch.save(agent.state_dict(), "./cartpole_solver.pt")
            break
        writer.close()

if __name__ == "__main__":

    env = gym.make("CartPole-v0")
    obs_size = env.observation_space.shape[0]
    n_act = env.action_space.n

    agent = Agent(obs_size, n_act)
    train(env, agent, BATCH_SIZE)