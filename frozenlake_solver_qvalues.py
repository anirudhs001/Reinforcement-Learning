
import collections
import gym
import tensorboardX
import os
import time

ENV_NAME = "FrozenLake-v1"
GAMMA = 0.9
TEST_EPISODES = 10
IS_SLIPPERY=True
REWARD_THRES = 0.9

class Agent():
    def __init__(self, gamma=GAMMA) -> None:
        self.Rewards = collections.defaultdict(float)         
        self.env = gym.make(ENV_NAME, is_slippery=IS_SLIPPERY)
        self.state = self.env.reset() # reset returns the observation.
        # for frozenlake, observation = current position in environment = current state
        self.Transns = collections.defaultdict(collections.Counter)
        self.Values = collections.defaultdict(float)
        self.Action_values = collections.defaultdict(float)
        self.gamma = gamma
    
    def play_n_random_steps(self, n):
        for _ in range(n):
            a = self.env.action_space.sample()
            new_s, r, is_done, _ = self.env.step(a)
            self.Rewards[(self.state, a, new_s)] = r
            self.Transns[(self.state, a)][new_s] += 1
            self.state = self.env.reset() if is_done else new_s
    

    def get_prob_of_reaching_state_on_action(self, start_state, action, end_state):
        state_counts = self.Transns[(start_state, action)]
        total_transns = sum(state_counts.values())
        return state_counts[end_state] / total_transns

    def update_value_tables(self, env=None):
        if env is None:
            env = self.env
        for s in range(env.observation_space.n):
            for a in range(env.action_space.n):
                action_value = 0.0
                for reachable_state in self.Transns[(s, a)].keys():
                    p = self.get_prob_of_reaching_state_on_action(s, a, reachable_state)
                    action_value += p * (self.Rewards[(s, a, reachable_state)] + \
                                    self.gamma * self.Values[reachable_state])
                self.Action_values[(s, a)] = action_value
        
        for s in range(env.observation_space.n):
            self.Values[s] = max([self.Action_values[(s, a_)] for a_ in range(env.action_space.n)])
                
    def select_best_action(self, state):
        '''Find best action. best action = action with highest action value for given input state'''
        best_a, best_q = None, None
        for a in range(self.env.action_space.n):
            q = self.Action_values[(state, a)]
            if best_q is None or best_q < q:
                best_q = q
                best_a = a
        return best_a

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
    
    
    
    
        print(f"Rewards:\n{self.Rewards}")

if __name__ == "__main__":
    env = gym.make(ENV_NAME, is_slippery = IS_SLIPPERY)
    env.render()
    agent = Agent()
    writer = tensorboardX.SummaryWriter(comment="-Q-learning")

    best_reward = 0.0
    epoch = 0
    print("Starting Training")
    while True:
        # print("play randomly for 100 episodes")
        agent.play_n_random_steps(100)
        agent.update_value_tables()
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
        if avg_reward > REWARD_THRES:
            print(f"Environment solved in {epoch}")
            break
        epoch += 1
    
    # Run the trained agent on environment
    agent.play_episode(env, render=True)

    writer.close()

    
    