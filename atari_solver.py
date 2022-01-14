from os import replace
import gym
from numpy.core.fromnumeric import repeat
import tensorboardX
import time
import numpy as np
import random
from matplotlib import pyplot as plt
import matplotlib.animation as animation
import torch
from torch import nn, optim
import argparse
import collections

ENV_NAME = "PongNoFrameskip-v4"
GAMMA = 0.99
BATCH_SIZE = 32
LR = 1e-4
BUFFER_MAXLEN = 10 ** 4
REPLAY_START_SIZE = 10 ** 4
REPEAT_ACTIONS_COUNT = 2
TARGET_NET_UPDATE_RATE = 1000
MAX_EPSILON = 1.0
MIN_EPSILON = 2e-2
EPSILON_DECAY_LAST_ITR = 1e5
THRESHOLD_SCORE = 18.0


class Agent:
    def __init__(self, n_actions, dev="cpu") -> None:
        self.buffer_maxlen = BUFFER_MAXLEN
        self.replay_buffer_initial = collections.deque(maxlen=BUFFER_MAXLEN//2)
        self.replay_buffer = collections.deque(maxlen=BUFFER_MAXLEN - BUFFER_MAXLEN//2)
        self.qmodel_main = self.Create_q_model(n_actions).to(dev)
        self.qmodel_targ = self.Create_q_model(n_actions).to(dev)
        self.color = np.array([144, 72, 17]).mean() # this color is replaced with 0 in image to improve contrast
        self.epsilon = MAX_EPSILON
        self.optim = optim.Adam(self.qmodel_main.parameters(), lr=LR)
        self.batch_size = BATCH_SIZE
        self.ep_score = 0.0
        self.objective = nn.MSELoss()
        self.dev = dev
        self.gamma = GAMMA

    def Create_q_model(self, n_actions):
        """
        Neural net to estimate Qvalues for all states and actions.
        takes in a state, outputs Qvalues for actions.
        """
        return nn.Sequential(
            nn.Conv2d(1, 32, 8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3136, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions),
        )

    def mod_img(self, s):
        s = s[34:194:2, ::2]
        s = np.mean(s, axis=2)
        s[s == self.color] = 0
        s = (s - 127.5) / 127.5
        s = torch.FloatTensor(s).unsqueeze(0)
        return s

    def add_transn(self, s, next_s, r, a, is_done):
        s_mod = self.mod_img(s)
        next_s_mod = self.mod_img(next_s)
        if self.replay_buffer_initial.maxlen > len(self.replay_buffer_initial):
            self.replay_buffer_initial.append((s_mod, next_s_mod, r, a, is_done))
        else:
            self.replay_buffer.append((s_mod, next_s_mod, r, a, is_done))

    @torch.no_grad()
    def get_action(self, s, env : gym.Env):
        a = None
        if np.random.random() < self.epsilon:
#             a = env.action_space.sample()
            a = np.random.randint(low = 0, high = 2)
        else:
            s_mod = self.mod_img(s).unsqueeze(0).to(self.dev)
            q_m = self.qmodel_main(s_mod).squeeze()
            a = torch.argmax(q_m).item()
        return a + 2

    @torch.no_grad()
    def play_k_steps(self, s, env : gym.Env, repeat_steps=1, save_transn=False):
        a = self.get_action(s, env)
        ep_score = 0
        r = 0.
        for _ in range(repeat_steps):
            next_s, _r, is_done, _ = env.step(a)
            r += _r
            if is_done: 
                break

        self.ep_score += r
        if save_transn:
            self.add_transn(s, next_s, r, a, is_done)
        ep_score = self.ep_score
        if is_done: 
            self.ep_score = 0
            next_s = env.reset()
        return next_s, is_done, ep_score

    def get_batch(self, batch_size=32):
        def get_batch_from_buff(buff, batch_size, s, next_s, r, a, is_done):
            minibatch = random.sample(buff, batch_size)
            for sample in minibatch:
                s.append(sample[0])
                next_s.append(sample[1])
                r.append(sample[2])
                a.append(sample[3])
                is_done.append(sample[4])
            return s, next_s, r, a, is_done
        s, next_s, r, a, is_done = [], [], [], [], []
        get_batch_from_buff(self.replay_buffer_initial, batch_size//2, s, next_s, r, a, is_done) 
        get_batch_from_buff(self.replay_buffer, batch_size - batch_size//2, s, next_s, r, a, is_done) 
        s = torch.stack(s, dim=0).to(self.dev)
        next_s = torch.stack(next_s, dim=0).to(self.dev)
        r = torch.tensor(r).to(self.dev)
        a = torch.tensor(a).to(self.dev)
        is_done = torch.BoolTensor(is_done).to(self.dev)
        return s, next_s, r, a, is_done

    def train(self, repeat_steps=1):
        if len(self.replay_buffer_initial) + len(self.replay_buffer) < REPLAY_START_SIZE:
            return 0
        self.optim.zero_grad()
        s, next_s, r, a, is_done = self.get_batch(self.batch_size)
        q_ms = self.qmodel_main(s)
        q_m = torch.gather(q_ms, 1, (a - 2).unsqueeze(-1)).squeeze()
        with torch.no_grad():
            best_a_next = torch.argmax(self.qmodel_main(next_s), dim=1) 
            q_ts = self.qmodel_targ(next_s)
            q_t = torch.gather(q_ts, 1, best_a_next.unsqueeze(-1)).squeeze()
            q_t[is_done] = 0.0
            q_t = q_t.detach()
        y = r + (self.gamma ** repeat_steps * q_t)
        loss = self.objective(q_m, y)
        loss.backward()
        self.optim.step()

        return loss.item()

    def save_net(self, path, epoch):
        model_state = {
            "epoch": epoch,
            "model": self.qmodel_main.state_dict(),
            "optim": self.optim.state_dict(),
        }
        path = f"{path}/model_{epoch}.pt"
        torch.save(model_state, path)

    def load_net(self, path):
        model_state = torch.load(path, map_location=self.dev) 
        self.qmodel_main.load_state_dict(model_state["model"])
        self.optim.load_state_dict(model_state["optim"])
        return model_state["epoch"]

    def update_targ_net(self):
        self.qmodel_targ.load_state_dict(self.qmodel_main.state_dict())


def train(env: gym.Env, agent: Agent):
    writer = tensorboardX.SummaryWriter(comment="-pacman_doubleQlearning")
    state = env.reset()

    print("Starting training")
    episode_num = 0
    time_step = 0
    reward_history = []
    best_reward = None
    while episode_num < 1000:
        # 1. run agent for k iteration and update replay buffer
        state, ep_done, ep_score = agent.play_k_steps(state, env, REPEAT_ACTIONS_COUNT, save_transn=True)
        if ep_done:
            reward_history.append(ep_score)
            print(f"reward for episode [{episode_num:.2f}] : mean reward = {np.mean(reward_history[-20:]):.2f}, r = {ep_score}, epsilon = {agent.epsilon:.2f}")
            writer.add_scalar("episode_reward", ep_score, episode_num)
            
            if not best_reward or best_reward < ep_score:
                agent.save_net(".", episode_num)
                best_reward = ep_score
                print(f"New best Score:{best_reward}")
                if best_reward > THRESHOLD_SCORE:
                    print("Stopping trainig")
                    break
            episode_num += 1
        # 2. train agent
        loss = agent.train(REPEAT_ACTIONS_COUNT)
        writer.add_scalar("train_loss", loss, time_step)
        agent.epsilon = max(
            MIN_EPSILON, MAX_EPSILON - time_step / EPSILON_DECAY_LAST_ITR
        )
        writer.add_scalar("epsilon", agent.epsilon, time_step)
        time_step += 1
        # 3. make target network = main network
        if time_step % TARGET_NET_UPDATE_RATE == 0:
            agent.update_targ_net()
            # print("updated target net")
        # exit if avg score > some threshold
    writer.close()
    # run agent on environment and show output


def run(env, agent: Agent, m):
    state = None
    mod_state = None
    ## Utility function to update output window with new state
    def show_game_screen(*args):
        im = args[1]
        im.set_array(state)
        return (im,)

    state = env.reset()
    mod_state = agent.mod_img(state).permute(1,2,0)
    fig, ax = plt.subplots()
    im = plt.imshow(mod_state, animated=True)
    plt.ion()
    fargs = (im,)
    ani = animation.FuncAnimation(
        fig, show_game_screen, fargs=fargs, interval=100, blit=True
    )
    plt.show()
    if m == "run-agent":
        agent.epsilon = 0
    elif m == "run-random":
        agent.epsilon = 1
    agent.load_net("model_209.pt")
    while True:
        state, is_done, score = agent.play_k_steps(state, env, repeat_steps=REPEAT_ACTIONS_COUNT, save_transn=False)
        mod_state = agent.mod_img(state).permute(1,2,0)
        plt.pause(0.001)


if __name__ == "__main__":
    # Main loop here
    print(f"Selected Env : {ENV_NAME}")
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--m",
        default="train",
        choices=["run-agent", "run-random", "train"],
        help="""
            run-random : play randomly,
            run-agent : play using trained model,
            train : train to train the model
            """,
    )
    env = gym.make(ENV_NAME)
    agent = Agent(2, dev="cpu")
    args = parser.parse_args()

    if args.m == "train":
        train(env, agent)
    else:
        run(env, agent, args.m)
