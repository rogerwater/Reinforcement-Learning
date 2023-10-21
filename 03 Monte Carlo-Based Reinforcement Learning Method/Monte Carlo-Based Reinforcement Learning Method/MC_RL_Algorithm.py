import pygame
import time
import random
import numpy as np
import matplotlib.pyplot as plt
from yuanyang_env import YuanYangEnv


class MC_RL:
    def __init__(self, yuanyang):
        # 初始化行为值函数
        self.qvalue = np.zeros((len(yuanyang.states), len(yuanyang.actions))) * 0.1
        # n用来表示状态行为对被访问的次数 q(s, a) = G(s, a) / n(s, a)
        self.n = 0.001 * np.ones((len(yuanyang.states), len(yuanyang.actions)))
        self.actions = yuanyang.actions
        self.yuanyang = yuanyang
        self.gamma = yuanyang.gamma

    def greedy_policy(self, qfun, state):
        amax = qfun[state, :].argmax()
        return self.actions[amax]

    def epsilon_greedy_policy(self, qfun, state, epsilon):
        amax = qfun[state, :].argmax()
        # 概率部分
        if np.random.uniform() < 1 - epsilon:
            return self.actions[amax]
        else:
            return self.actions[int(random.random() * len(self.actions))]

    # 找到动作对应的序号
    def find_anum(self, a):
        for i in range(len(self.actions)):
            if a == self.actions[i]:
                return i

    # 探索性初始化蒙特卡洛方法
    def mc_rl_ei(self, num_iter):
        self.qvalue = np.zeros((len(self.yuanyang.states), len(self.yuanyang.actions)))
        self.n = 0.001 * np.ones((len(self.yuanyang.states), len(self.yuanyang.actions)))
        # 学习num_iter次
        for iter1 in range(num_iter):
            # 采集状态样本
            s_sample = []
            # 采集动作样本
            a_sample = []
            # 采集回报样本
            r_sample = []
            # 随机初始化状态
            s = self.yuanyang.reset()
            a = self.actions[int(random.random() * len(self.actions))]
            done = False
            step_num = 0

            # 调用mc_test函数，该函数用来测试经过学习后策略是否找到了目标；如果找到，返回1，否则返回0
            if self.mc_test() == 1:
                print("探索初始化第一次完成任务需要的次数：", iter1)
                break

            # 采集数据，用当前贪婪策略进行一次试验，并保存相应的数据到s_sample，r_sample，a_sample
            while False == done and step_num < 30:
                # 与环境交互
                s_next, r, done = self.yuanyang.transform(s, a)
                a_num = self.find_anum(a)
                # 往回走给予惩罚
                if s_next in s_sample:
                    r = -2
                # 存储数据，采样数据
                s_sample.append(s)
                r_sample.append(r)
                a_sample.append(a_num)
                # 转移到下一个状态，继续试验
                s = s_next
                a = self.greedy_policy(self.qvalue, s)

                # 根据回报值计算折扣累积回报，先得到下一个状态处的值函数，然后逆向求解前面状态行为对的折扣累积回报
                a = self.greedy_policy(self.qvalue, s)
                g = self.qvalue[s, self.find_anum(a)]
                for i in range(len(s_sample) -1, -1, -1):
                    g *= self.gamma
                    g += r_sample[i]

                for i in range(len(s_sample)):
                    # 计算状态行为对的次数
                    self.n[s_sample[i], a_sample[i]] += 1.0
                    # 利用增量式方法更新值函数
                    self.qvalue[s_sample[i], a_sample[i]] = (self.qvalue[s_sample[i], a_sample[i]] * (self.n[s_sample[i], a_sample[i]] - 1) + g) / self.n[s_sample[i], a_sample[i]]
                    g -= r_sample[i]
                    g /= self.gamma
        return self.qvalue

    def mc_rl_on_policy(self, num_iter, epsilon):
        self.qvalue = np.zeros((len(self.yuanyang.states), len(self.yuanyang.actions)))
        self.n = 0.001 * np.ones((len(self.yuanyang.states), len(self.yuanyang.actions)))
        # 学习num_iter次
        for iter1 in range(num_iter):
            # 采集状态样本
            s_sample = []
            # 采集动作样本
            a_sample = []
            # 采集回报样本
            r_sample = []
            #固定初始状态
            s = 0
            done = False
            step_num = 0
            epsilon = epsilon * np.exp(-iter1 / 1000)

            # 设置好基本变量，进入与环境交互程序，根据当前策略不断采集数据
            # 采集数据
            while False == done and step_num < 30:
                a = self.epsilon_greedy_policy(self.qvalue, s, epsilon)
                # 与环境交互
                s_next, r, done = self.yuanyang.transform(s, a)
                a_num = self.find_anum(a)
                # 往回走给予惩罚
                if s_next in s_sample:
                    r = -2
                # 存储数据，采样数据
                s_sample.append(s)
                r_sample.append(r)
                a_sample.append(a_num)
                step_num += 1
                # 转移到下一个状态，继续试验
                s = s_next

            if s == 90:
                print("同策略第一次完成任务需要的次数：", iter1)
                break

            # 根据回报值计算折扣累积回报，先得到下一个状态处的值函数，然后逆向求解前面状态行为对的折扣累积回报
            a = self.epsilon_greedy_policy(self.qvalue, s, epsilon)
            g = self.qvalue[s, self.find_anum(a)]
            for i in range(len(s_sample) - 1, -1, -1):
                g *= self.gamma
                g += r_sample[i]

            for i in range(len(s_sample)):
                self.n[s_sample[i], a_sample[i]] += 1.0
                self.qvalue[s_sample[i], a_sample[i]] = (self.qvalue[s_sample[i], a_sample[i]] * (self.n[s_sample[i], a_sample[i]] - 1) + g) / self.n[s_sample[i], a_sample[i]]
                g -= r_sample[i]
                g /= self.gamma
        return self.qvalue

    # 测试子函数，初始状态为0,与环境交互
    def mc_test(self):
        s = 0
        s_sample = []
        done = False
        flag = 0
        step_num = 0
        while False == done and step_num < 30:
            a = self.greedy_policy(self.qvalue, s)
            # 与环境交互
            s_next, r, done = self.yuanyang.transform(s, a)
            s_sample.append(s)
            s = s_next
            step_num += 1
        if s == 90:
            flag = 1
        return flag


if __name__ == "__main__":
    yuanyang = YuanYangEnv()
    mc_rl = MC_RL(yuanyang)
    # 探索性初始化方法
    qvalue1 = mc_rl.mc_rl_ei(num_iter=10000)
    # 同策略方法
    qvalue2 = mc_rl.mc_rl_on_policy(num_iter=10000, epsilon=0.2)
    # 将行为值函数渲染出来
    yuanyang.action_value = qvalue2
    # 测试学习到的策略
    flag = 1
    s = 0
    step_num = 0
    while flag:
        yuanyang.path.append(s)
        a = mc_rl.greedy_policy(qvalue2, s)
        print('%d -> %s\t'%(s,a), qvalue2[s,0],qvalue2[s,1],qvalue2[s,2],qvalue2[s,3])
        yuanyang.bird_male_position = yuanyang.state_to_position(s)
        yuanyang.render()
        time.sleep(0.25)
        step_num += 1
        s_, r, t = yuanyang.transform(s, a)
        if t == True or step_num > 30:
            flag = 0
        s = s_
    yuanyang.bird_male_position = yuanyang.state_to_position(s)
    yuanyang.path.append(s)
    yuanyang.render()
    while True:
        yuanyang.render()
