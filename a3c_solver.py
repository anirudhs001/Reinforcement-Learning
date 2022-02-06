import argparse
import gym
import collections
import torch
from torch import nn, optim
import torch.nn.functional as F
import torch.multiprocessing as mp
import numpy as np
from tensorboardX import SummaryWriter

ENV_NAME = "PongNoFrameskip-v4"
# ENV_NAME="CartPole-v0"
PROC_COUNT = 1
NUM_ENVS_PER_PROC = 1
BATCH_SIZE = 128

ENTROPY_BETA = 1e-2
DEV = "cuda" if torch.cuda.is_available() else "cpu"
print(DEV)
GAMMA = 0.99
REPEAT_STEPS = 4


class Agent(nn.Module):

    def __init__(self, obs_size, n_act: int, dev=DEV) -> None:
        super().__init__()
        self.dev = dev
        self.gamma = GAMMA
        self.rep_steps = REPEAT_STEPS
        self.conv = nn.Sequential(
            nn.Conv2d(obs_size[2], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
        ).to(dev)
        _conv_out_size = self.get_out_size((obs_size))
        self.policy_net = nn.Sequential(
            nn.Linear(_conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, n_act)
        ).to(dev)
        self.value_net = nn.Sequential(
            nn.Linear(_conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        ).to(dev)

    def forward(self, x):
        x = self.conv(x.permute(0,3,1,2))
        x = x.view(x.shape[0], -1)
        return self.policy_net(x), self.value_net(x)

    @torch.no_grad()
    def get_out_size(self, shape):
        out = self.conv(torch.zeros(size=(1, shape[2], shape[0], shape[1])).to(self.dev))
        return int(np.prod(out.shape))

    def get_action(self, s):
        s_v = torch.FloatTensor(s).to(self.dev).unsqueeze(0)
        a = self(s_v)[0].squeeze().argmax().item()
        return a

    def save_net(self, epoch, path="."):
        model_state = {
            "epoch": epoch,
            "model": self.qmodel_main.state_dict(),
            "optim": self.optim.state_dict(),
        }
        path = f"{path}/model_{epoch}.pt"
        torch.save(model_state, path)

    def load_net(self, fname):
        model_state = torch.load(fname, map_location=self.dev)
        self.qmodel_main.load_state_dict(model_state["model"])
        self.optim.load_state_dict(model_state["optim"])
        return model_state["epoch"]


Reward_container = collections.namedtuple("Episode_Reward", 'reward')


class Experience_source():
    def __init__(self, agent: Agent, num_envs=NUM_ENVS_PER_PROC):
        self.envs = [gym.make(ENV_NAME) for _ in range(num_envs)]
        self.n_envs = num_envs
        self.ep_rewards = [0. for _ in range(self.n_envs)]
        self.curr_states = [env.reset() for env in self.envs]
        self.agent = agent
        self.rep_steps = REPEAT_STEPS

    def step_envs(self):
        exps = []
        total_rewards = []
        for i, (s, env) in enumerate(zip(self.curr_states, self.envs)):
            a = self.agent.get_action(s)
            r = 0
            new_s = s
            for _ in range(self.rep_steps):
                new_s, _r, is_done, _ = env.step(self.agent.get_action(new_s))
                r += _r
                self.ep_rewards[i] += _r
                if is_done:
                    env.reset()
                    total_rewards.append(Reward_container(self.ep_rewards[i]))
                    self.ep_rewards[i] = 0.
                    break
            exps.append((s, new_s, a, r, is_done))

        return total_rewards, exps


def data_func(agent: Agent, shared_queue: mp.Queue):
    '''Func run on diff processes to collect data'''
    print(f"Child [{mp.current_process().name}] started")
    exp_src = Experience_source(agent=agent, num_envs=NUM_ENVS_PER_PROC)
    while True:
        ep_rewards, exps = exp_src.step_envs()
        if len(ep_rewards) > 0:
            shared_queue.put(*ep_rewards)
            print(ep_rewards)
        shared_queue.put(*exps)


def create_batch(agent, batch, dev=DEV):
    states, actions, next_states, rewards, is_dones = [], [], [], [], []
    for s, n_s, a, r, isdn in batch:
        states.append(s)
        actions.append(a)
        next_states.append(n_s)
        rewards.append(r)
        is_dones.append(isdn)

    state_v = torch.FloatTensor(np.array(states)).to(dev)
    next_state_v = torch.FloatTensor(np.array(next_states)).to(dev)
    # print(next_state_v.shape)
    action_v = torch.LongTensor(actions).to(dev)
    reward_v = torch.FloatTensor(rewards).to(dev).unsqueeze(1)
    is_done_v = torch.BoolTensor(is_dones).to(dev)
    _, val_pred_next_v = agent(next_state_v)
    val_pred_next_v[is_done_v] = 0.
    traj_return_v = reward_v + ((agent.gamma ** agent.rep_steps) * val_pred_next_v)

    return action_v, state_v, traj_return_v
    

if __name__ == "__main__":
    try:
        mp.set_start_method('spawn')
    except:
        pass
    shared_queue = mp.Queue(maxsize=NUM_ENVS_PER_PROC)
    dummy_env = gym.make(ENV_NAME)
    agent = Agent(dummy_env.observation_space.shape,
                  dummy_env.action_space.n)
    del dummy_env
    optimizer = optim.Adam(agent.parameters())
    agent.share_memory()
    ep_rewards = collections.deque([0. for _ in range(100)], maxlen=100)
    r_mean = 0.
    writer = SummaryWriter(comment=f"{ENV_NAME}_a3c")

    best_r = None

    data_proc_list = []
    for _ in range(PROC_COUNT):
        data_proc = mp.Process(target=data_func, args=(agent, shared_queue))
        data_proc.start()
        data_proc_list.append(data_proc)

    print(f"Main Process [{mp.current_process().name}]: training started")

    batch = []
    while True:
        obj = shared_queue.get()
        if isinstance(obj, Reward_container):
            r_mean += (obj.reward - ep_rewards[0]) / len(ep_rewards)
            ep_rewards.append(obj.reward)
            writer.add_scalar("episode_reward", obj.reward)
            writer.add_scalar("episode_reward_mean", r_mean)

            if best_r is None or obj.reward > best_r:
                agent.save_net()
                best_r = obj.reward
                print(f"best model updated. score = {best_r}")

            continue

        batch.append(obj)
        if len(batch) >= BATCH_SIZE:
            optimizer.zero_grad()
            actions_v, states_v, traj_return_v = create_batch(agent, batch)

            logits_v, state_values_v = agent(states_v)

            prob_v = F.softmax(logits_v, dim=1)
            log_prob_v = F.log_softmax(logits_v, dim=1)
            log_prob_v = log_prob_v[actions_v]
            adv_v = traj_return_v - state_values_v
            loss_pol = -(log_prob_v*adv_v).mean()

            loss_val = F.mse_loss(state_values_v, traj_return_v)

            loss_entropy = ENTROPY_BETA * \
                (prob_v * log_prob_v).sum(dim=1).mean()

            loss = loss_entropy + loss_val + loss_pol

            loss.backward()
            optimizer.step()
            batch.clear()

            print(f"Total_loss = {loss.item():.4f}")
            writer.add_scalar("total_loss", loss.item())
            writer.add_scalar("policy_loss", loss_pol.item())
            writer.add_scalar("value_loss", loss_val.item())
            writer.add_scalar("entropy_loss", loss_entropy.item())