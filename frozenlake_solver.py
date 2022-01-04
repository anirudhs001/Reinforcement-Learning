
import collections
import gym
import tensorboardX
import os
import time

ENV_NAME = "FrozenLake-v1"
GAMMA = 0.9
TEST_EPISODES = 10
IS_SLIPPERY=True

class Agent():
    def __init__(self, gamma=GAMMA) -> None:
        self.Rewards = collections.defaultdict(float)         
        self.env = gym.make(ENV_NAME, is_slippery=IS_SLIPPERY)
        self.state = self.env.reset() # reset returns the observation.
        # for frozenlake, observation = current position in environment = current state
        self.Transns = collections.defaultdict(collections.Counter)
        self.Values = collections.defaultdict(float)
        self.gamma = gamma
    
    def play_n_random_steps(self, n):
        for _ in range(n):
            a = self.env.action_space.sample()
            new_s, r, is_done, _ = self.env.step(a)
            self.Rewards[(self.state, a, new_s)] = r
            self.Transns[(self.state, a)][new_s] += 1
            self.state = self.env.reset() if is_done else new_s
    
    def get_action_values(self, state, action):
        action_value = 0.0
        reachable_states_and_counts = self.Transns[(state, action)]
        count_of_reachable_states = sum(reachable_states_and_counts.values())
        for tgt_state, count in reachable_states_and_counts.items():
            reward = self.Rewards[(state, action, tgt_state)]
            action_value += (count / count_of_reachable_states) * \
                            (reward + self.gamma * self.Values[tgt_state])
        return action_value

    def select_best_action(self, state):
        '''Find best action. best action = action with highest action value for given input state'''
        best_a, best_q = None, None
        for a in range(self.env.action_space.n):
            q = self.get_action_values(state, a)
            if best_q is None or best_q < q:
                best_q = q
                best_a = a

        return best_a
    
    def update_state_values(self):
        for state in range(self.env.observation_space.n):
            action_values = [self.get_action_values(state, a) for a in range(self.env.action_space.n)]
            self.Values[state] = max(action_values)
    
    def play_episode(self, env, render=False):
        total_reward = 0.0
        is_done = False
        state = env.reset()
        while not is_done:
            a = self.select_best_action(state)
            state, r, is_done, _ = env.step(a)
            total_reward += r
            if render:
                os.system("cls" if os.name == 'nt' else 'clear')
                env.render()
                time.sleep(0.5)
        return total_reward
    
    def show_matrices(self):
        print(f"State Values:\n{self.Values}")
        print(f"Transition Dynamics:\n{self.Transns}")
        print(f"Rewards:\n{self.Rewards}")

if __name__ == "__main__":
    env = gym.make(ENV_NAME, is_slippery = IS_SLIPPERY)
    env.render()
    agent = Agent()
    writer = tensorboardX.SummaryWriter(comment="-V-learning")

    best_reward = 0.0
    epoch = 0
    print("Starting Training")
    while True:
        # print("play randomly for 100 episodes")
        agent.play_n_random_steps(100)
        agent.update_state_values()
        # agent.show_matrices()
        reward = 0.0 # reward for one episode
        for i in range(TEST_EPISODES):
            reward += agent.play_episode(env)
        avg_reward = reward / TEST_EPISODES
        # print(avg_reward)
        writer.add_scalar("Avg Reward", avg_reward, epoch)
        if avg_reward > best_reward:
            best_reward = avg_reward
            print(f"Best reward updated : {best_reward}")
        if avg_reward > 0.8:
            print(f"Environment solved in {epoch}")
            break
        epoch += 1
    
    # Run the trained agent on environment
    agent.play_episode(env, render=True)

    writer.close()

    
    