import argparse
import gym
import torch
from torch import nn, optim
import torch.functional as F
import numpy as np
from tensorboardX import SummaryWriter

ENV = "LunarLander-v2"
K = 4
GAMMA = 0.99

# Neural network for agent
class Agent():

    def __init__(self, env : gym.Env, obs_size : int, n_act : int, dev="cpu") -> None:
        self.net = self.Create_net(obs_size, n_act).to(dev)
        self.dev = dev
        self.softmax = nn.Softmax()
        self.env = env
        self.n_act = n_act
        self.optimizer = optim.Adam(self.net.parameters())

    def Create_net(self, obs_size, n_act):
        return nn.Sequential(
                nn.Linear(obs_size, 128),
                nn.ReLU(),
                nn.Linear(128, n_act),
            )
    
    @torch.no_grad()
    def get_action(self, state):
        state_v = torch.FloatTensor(state).unsqueeze(dim=0).to(self.dev)
        logits = self.net(state_v).squeeze().cpu()
        probs = np.around(self.softmax(logits).numpy(), decimals=10)
        action = np.random.choice(n_act, p=probs)
        return action


def train(env : gym.Env, agent: Agent, dev="cpu"):
    writer = SummaryWriter(comment=f"{ENV}_reinforce")
    best_reward = None
    mean_reward = 0
    ep_no = 0
    ep_rewards = []
    epoch = 0
    while True:
        epoch += 1
        # gather k episodes
        states = []
        actions = []
        next_states = []
        rewards = []

        for _ in range(K):
            time_factor = 1
            state = env.reset()
            ep_reward = 1
            while True:
                time_factor += 0.1
                print(time_factor)
                a = agent.get_action(state)
                new_state, r, is_done, _ = env.step(a)  
                if r > 0 :
                    r /= time_factor
                states.append(state)
                actions.append(a)
                next_states.append(new_state)
                rewards.append(r)
                ep_reward += r
                state = new_state
                if is_done:
                    ep_no += 1
                    # log episode info
                    ep_rewards.append(ep_reward)
                    writer.add_scalar("episode reward", ep_reward, ep_no)
                    break
            print(time_factor)


        mean_reward = float(np.mean(ep_rewards[-5:]))
        if mean_reward is not None and (best_reward is None or mean_reward > best_reward):
            torch.save(agent.net.state_dict(), f"model_{ENV}_reinforce.pt")
            best_reward = mean_reward
            print(f"best reward updated: {best_reward}")
            

        q_values = []
        sum_r = 0
        for r in reversed(rewards):
            sum_r *= GAMMA
            sum_r += r

            q_values.append(sum_r)

        q_values.reverse()

        # train
        agent.optimizer.zero_grad()

        q_values_v = torch.FloatTensor(q_values).to(dev)
        states_v = torch.FloatTensor(states).to(dev)
        actions_v = torch.LongTensor(actions).to(dev)
        logits = agent.net(states_v)
        log_prob_v = nn.functional.log_softmax(logits, dim=1) 

        Loss = -q_values_v * log_prob_v[range(len(states_v)), actions_v]
        Loss = Loss.mean()
        Loss.backward()
        agent.optimizer.step()

        # add loss in plot 
        writer.add_scalar("loss", Loss.item(), epoch)
        states.clear()
        next_states.clear()
        rewards.clear()
        actions.clear()
        

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--m', choices=['train', 'run'], help='run to run the trained the model, train to train the model')
    parser.add_argument("--p", help="path to saved model in current directory")
    dev = "cpu"
    args = parser.parse_args()
    env = gym.make(ENV)
    obs_size = env.observation_space.shape[0]
    n_act = env.action_space.n
    agent = Agent(env, obs_size, n_act, dev)

    if args.m == 'train':
        train(env, agent, dev)
    elif args.m == 'run':
        agent.net.load_state_dict(torch.load(args.p, map_location=dev))
        is_done = False
        total_reward = 0.0
        with torch.no_grad():
            obs = env.reset()
            sm = nn.Softmax()
            while not is_done:
                obs_tens = torch.FloatTensor(obs)
                act_probs = sm(agent.net(obs_tens))
                act = int(torch.argmax(act_probs))
                env.render()
                obs, reward, is_done, _ = env.step(act)
                
    env.close()
