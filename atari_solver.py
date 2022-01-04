import collections
import gym
import tensorboardX
import time
import numpy as np
import random
from matplotlib import pyplot as plt
import matplotlib.animation as animation
import torch
from torch import nn, optim
from torch.nn.modules import padding

ENV_NAME = "MsPacman-v0"
BUFFER_MAXLEN = 5000
N1 = 100
EPISODE_LENGTH = 500


class Agent:
    def __init__(self, n_actions) -> None:
        self.replay_buffer = collections.deque(maxlen=BUFFER_MAXLEN)
        self.qmodel_main = self.QModel(n_actions)
        self.qmodel_target = self.QModel(n_actions)
        self.qmodel_target.eval()
        self.color = np.array(
            [210, 164, 74]
        ).mean()  # constant color used by mod_image().
        # this color is replaced with 0 in image to improve contrast
        self.epsilon = 0.2
        self.optim = optim.Adam(self.qmodel_main.parameters(), lr=1e-2)
        self.batch_size = 16
        self.total_reward = 0.0

    def QModel(n_actions, in_h=88, in_w=80, in_c=1):
        """
        Neural net to estimate Qvalues for all states and actions.
        takes in a state, outputs Qvalues for actions.
        """
        net = nn.Sequential(
            nn.Conv2d(in_c, 16, 8, stride=4),
            nn.LeakyReLU(),
            nn.Conv2d(16, 32, 4, stride=2),
            nn.LeakyReLU(),
            nn.Flatten(),
            nn.Linear(2304, 1600),
            nn.LeakyReLU(),
            nn.Linear(1600, n_actions),
        )
        return net

    def mod_image(self, state):
        """
        Function to transform raw screen to smaller image before passing to the neural net.
        output = 88x80x1 image
        """
        image = state[1:176:2, ::2]
        image = image.mean(axis=2)
        image[image == self.color] = 0
        # normalize image
        image = (image - 128) / 128 - 1
        # print(image.shape)
        # reshape: add channel dimension
        image = torch.tensor(np.expand_dims(image.reshape(88, 80, 1), axis=0)).view(
            2, 1, 0
        )
        return image

    def update_target_network(self):
        self.qmodel_target.load_state_dict(self.qmodel_main.state_dict())

    def play_episode(self, env, s):
        a = self.get_action(s, env)
        new_s, r, is_done, _ = env.step(a)
        self.add_transn(s, a, new_s, r, is_done)
        if is_done:
            s = env.reset()
            self.total_reward = 0
        else:
            s = new_s
            self.total_reward += r
        return s

    def add_transn(self, s, a, new_s, r, done):
        self.replay_buffer.append(
            (self.mod_image(s), a, self.mod_image(new_s), r, done)
        )

    def get_action(self, s, env):
        """
        Take an action acc to epsilon greedy policy
        """

        # 1. select action with epsilon greedy policy
        if np.random() < self.epsilon:
            return env.action_space.sample()
        else:
            # 2. find a which maximizes Q_model(s, a)
            best_a = None
            s_mod = self.mod_image(s).unsqueeze(0)
            with torch.no_grad():
                q_m = self.qmodel_main(s_mod).squeeze(0)
            best_a = torch.argmax(q_m)
            return best_a

    def train(self):
        class criterion(nn.Module):
            def __init__(self) -> None:
                super().__init__()

            def forward(self, actions, rewards, q_ms, dones):
                loss = 0.0
                for i in range(self.batch_size):
                    yi = rewards[i]
                    if not dones[i]:
                        yi += q_ts[i][best_actions_next[i]]
                    loss += yi - q_ms[i][actions[i]]
                loss = loss ** 2
                loss /= self.batch_size
                return loss

        if len(self.replay_buffer) < self.batch_size:
            return
        states, actions, next_states, rewards, dones = self.get_batch()
        self.optim.zero_grad()
        q_ms = self.qmodel_main(states)
        with torch.no_grad():
            q_ms_next = self.qmodel_main(next_states)
            best_actions_next = torch.argmax(q_ms_next, dim=1)
            q_ts = self.qmodel_target(next_states)

        loss = criterion(actions, rewards, q_ms, dones)
        loss.backward()
        optim.step()

    def get_batch(self):
        minibatch = random.sample(self.replay_buffer, self.batch_size)
        return minibatch


## Utility function to update output window with new state
def show_game_screen(*args):
    env = args[1]
    im = args[2]
    state, _, _, _ = env.step(env.action_space.sample())
    im.set_array(state)
    return (im,)


if __name__ == "__main__":
    pass
    # Main loop here
    env = gym.make(ENV_NAME)
    agent = Agent(env.action_space.n)
    writer = tensorboardX.SummaryWriter(comment="-pacman_doubleQlearning")

    state = env.reset()
    print("Starting training")
    episode_num = 0
    while True:
        # 1. loop N times
        while _ in range(EPISODE_LENGTH):
            # 1. run agent for 1 iteration and update replay buffer
            state, total_reward, is_done = agent.play_episode(env)
            if is_done:
                # print(f"reward for episode = {total_reward}")
                writer.add_scalar("episode_reward", total_reward, episode_num)
                episode_num += 1
            # 2. train agent
            agent.train()
        # 2. make target network = main network
        agent.update_target_network()
        # exit if avg score > some threshold
    writer.close()
    # run agent on environment and show output

    # fig, ax = plt.subplots()
    # im = plt.imshow(state, animated=True)
    # fargs = (env, im)
    # ani = animation.FuncAnimation(fig, show_game_screen, fargs=fargs, interval = 10, blit=True)
    # plt.show()
