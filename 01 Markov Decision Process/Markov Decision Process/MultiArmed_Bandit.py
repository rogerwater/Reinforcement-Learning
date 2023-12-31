# 多臂赌博机代码实现
import numpy as np
from matplotlib import pyplot as plt


class KB_Game:
    def __init__(self, *args, **kwargs):
        self.q = np.array([0.0, 0.0, 0.0])
        self.action_counts = np.array([0, 0, 0])
        self.current_cumulative_rewards = 0.0
        self.actions = [1, 2, 3]
        self.counts = 0
        self.counts_history = []
        self.current_cumulative_rewards_history = []
        self.a = 1
        self.r = 0

    def step(self, a):
        r = 0
        if a == 1:
            r = np.random.normal(1, 1)
        if a == 2:
            r = np.random.normal(2, 1)
        if a == 3:
            r = np.random.normal(1.5, 1)
        return r

    def choose_action(self, policy, **kwargs):
        action = 0
        if policy == "e_greedy":
            if np.random.random() < kwargs["epsilon"]:
                action = np.random.randint(1, 4)
            else:
                action = np.argmax(self.q) + 1

        if policy == "ucb":
            c_ratio = kwargs["c_ratio"]
            if 0 in self.action_counts:
                action = np.where(self.action_counts == 0)[0][0] + 1

            else:
                value = self.q + c_ratio*np.sqrt(np.log(self.counts) / self.action_counts)
                action = np.argmax(value) + 1

        if policy == "boltzmann":
            tau = kwargs["temperature"]
            p = np.exp(self.q / tau) / (np.sum(np.exp(self.q /tau)))
            action = np.random.choice([1, 2, 3], p = p.ravel())

        return action

    def train(self, play_total, policy, **kwargs):

        for i in range(play_total):
            action = 0
            if policy == "e_greedy":
                action = self.choose_action(policy, epsilon=kwargs["epsilon"])

            if policy == "ucb":
                action = self.choose_action(policy, c_ratio=kwargs["c_ratio"])

            if policy == "boltzmann":
                action = self.choose_action(policy, temperature=kwargs["temperature"])

            # 与环境交互一次
            self.a = action
            self.r = self.step(self.a)
            self.counts += 1

            # 更新值函数
            self.q[self.a-1] = (self.q[self.a-1]*self.action_counts[self.a-1]+self.r) / (self.action_counts[self.a-1] + 1)
            self.action_counts[self.a-1] += 1
            self.current_cumulative_rewards += self.r
            self.current_cumulative_rewards_history.append(self.current_cumulative_rewards)
            self.counts_history.append(i)

    def reset(self):
        self.q = np.array([0.0, 0.0, 0.0])
        self.action_counts = np.array([0, 0, 0])
        self.current_cumulative_rewards = 0.0
        self.counts = 0
        self.counts_history = []
        self.current_cumulative_rewards_history = []
        self.a = 1
        self.r = 0

    def plot(self, colors, policy):
        plt.figure(1)
        plt.plot(self.counts_history, self.current_cumulative_rewards_history, colors, label=policy)
        plt.legend()
        plt.xlabel('n', fontsize=18)
        plt.ylabel('total rewards', fontsize=18)

if __name__ == '__main__':
    np.random.seed(0)
    k_gamble = KB_Game()
    total = 2000

    k_gamble.train(play_total=total, policy='e_greedy', epsilon=0.05)
    k_gamble.plot(colors='r', policy='e_greedy')
    k_gamble.reset()

    k_gamble.train(play_total=total, policy='boltzmann', temperature=1)
    k_gamble.plot(colors='b', policy='boltzmann')
    k_gamble.reset()

    k_gamble.train(play_total=total, policy='ucb',c_ratio=0.5)
    k_gamble.plot(colors='g', policy='ucb')
    k_gamble.reset()

    plt.show()


