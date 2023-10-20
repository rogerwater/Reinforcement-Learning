import random
import time
from yuanyang_env import YuanYangEnv


class DP_Policy_Iter:
    def __init__(self, yuanyang):
        self.states = yuanyang.states
        self.actions = yuanyang.actions
        self.v = [0.0 for i in range(len(self.states) + 1)]
        self.pi = dict()
        self.yuanyang = yuanyang
        self.gamma = yuanyang.gamma

        # 初始化策略
        for state in self.states:
            flag1 = 0
            flag2 = 0
            flag1 = yuanyang.collide(yuanyang.state_to_position(state))
            flag2 = yuanyang.find(yuanyang.state_to_position(state))
            if flag1 == 1 or flag2 == 1: continue
            # 利用随机函数初始化策略
            self.pi[state] = self.actions[int(random.random() * len(self.actions))]

    # 策略评估子函数
    def policy_evaluate(self):
        # 计算值函数
        for i in range(100):
            delta = 0.0
            for state in self.states:
                flag1 = 0
                flag2 = 0
                flag1 = self.yuanyang.collide(self.yuanyang.state_to_position(state))
                flag2 = self.yuanyang.find(self.yuanyang.state_to_position(state))
                if flag1 == 1 or flag2 == 1: continue

                action = self.pi[state]
                s, r, t = self.yuanyang.transform(state, action)

                # 更新值
                new_v = r + self.gamma * self.v[s]
                delta += abs(self.v[state] - new_v)
                self.v[state] = new_v
            if delta < 1e-6:
                print("策略评估迭代次数：", i)
                break

    # 策略改善子函数
    def policy_improve(self):
        # 利用更新后的值函数进行策略改善
        for state in self.states:
            flag1 = 0
            flag2 = 0
            flag1 = self.yuanyang.collide(self.yuanyang.state_to_position(state))
            flag2 = self.yuanyang.find(self.yuanyang.state_to_position(state))
            if flag1 == 1 or flag2 == 1: continue

            a1 = self.actions[0]
            s, r, t = self.yuanyang.transform(state, a1)
            v1 = r + self.gamma * self.v[s]

            for action in self.actions:
                s, r, t = self.yuanyang.transform(state, action)
                if v1 < r + self.gamma * self.v[s]:
                    a1 = action
                    v1 = r + self.gamma * self.v[s]

            # 贪婪策略进行更新
            self.pi[state] = a1

    # 策略迭代子函数
    def policy_iteration(self):
        for i in range(100):
            # 策略评估
            self.policy_evaluate()
            # 策略改善
            pi_old = self.pi.copy()
            self.policy_improve()
            if pi_old == self.pi:
                print("策略改善次数：", i)
                break


if __name__ == "__main__":
    yuanyang = YuanYangEnv()
    policy_iter = DP_Policy_Iter(yuanyang)
    policy_iter.policy_iteration()

    # 对学到的策略进行测试，初始状态为0，当前的路径还不存在
    # 将策略迭代中学到的值函数给到游戏中，渲染出来
    flag = 1
    s = 0

    # 将v值打印出来
    for state in range(100):
        i = int(state/10)
        j = state % 10
        yuanyang.value[i, j] = policy_iter.v[state]

    step_num = 0

    # 下面是agent利用学到的策略pi与游戏环境进行交互
    # 并将交互结果渲染出来，在雄鸟移动过程中，我们将移动的状态和动作都打印出来
    # 将最优路径打印出来
    while flag:
        # 渲染路径点
        yuanyang.path.append(s)
        a = policy_iter.pi[s]
        print('%d -> %s \t'%(s, a))
        yuanyang.bird_male_position = yuanyang.state_to_position(s)
        yuanyang.render()
        time.sleep(0.2)
        step_num += 1
        s_, r, t = yuanyang.transform(s, a)
        if t == True or step_num > 200:
            flag = 0
        s = s_

    # 渲染最后的路径点
    yuanyang.bird_male_position = yuanyang.state_to_position(s)
    yuanyang.path.append(s)
    yuanyang.render()
    while True:
        yuanyang.render()
