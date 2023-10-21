import numpy as np
import random
import os
import pygame
import time
import matplotlib.pyplot as plt
from yuanyang_env import YuanYangEnv


class TD_RL:
    def __init__(self, yuanyang):
        self.gamma = yuanyang.gamma
        self.yuanyang = yuanyang
        # 值函数的初始值
        self.qvalue = np.zeros((len(self.yuanyang.states), len(self.yuanyang.actions)))

    def greedy_policy(self, qfun, state):
        amax = qfun[state, :].argmax()
        return self.yuanyang.actions[amax]

    def epsilon_greedy_policy(self, qfun, state, epsilon):
        amax = qfun[state, :].argmax()
        if np.random.uniform() < 1 - epsilon:
            return self.yuanyang.actions[amax]
        else:
            return self.yuanyang.actions[int(random.random() * len(self.yuanyang.actions))]

    def find_anum(self, a):
        for i in range(len(self.yuanyang.actions)):
            if a == self.yuanyang.actions[i]:
                return i

    def sarsa(self, num_iter, alpha, epsilon):
        iter_num = []
        self.qvalue = np.zeros((len(self.yuanyang.states), len(self.yuanyang.actions)))
        # 第一个大循环，产生了多少次实验
        for iter in range(num_iter):
            # 随机初始化状态
            epsilon = epsilon * 0.99
            s_sample = []
            s = 0
            flag = self.greedy_test()
            if flag == 1:
                iter_num.append(iter)
                if len(iter_num) < 2:
                    print("Sarsa第一次完成任务需要的迭代次数为：", iter_num[0])
            if flag == 2:
                print("Sarsa第一次实现最短路径需要的迭代次数为：", iter)
                break

            # 利用epsilon-greedy策略选择初始动作
            a = self.epsilon_greedy_policy(self.qvalue, s, epsilon)
            t = False
            count = 0
            while t == False and count < 30:
                # 与环境交互得到下一个状态
                s_next, r, t = self.yuanyang.transform(s, a)
                a_num = self.find_anum(a)

                if s_next in s_sample:
                    r = -2
                s_sample.append(s)

                # 判断是否是终止状态
                if t == True:
                    q_target = r
                else:
                    # 下一个状态处的最大动作，此处体现同策略
                    a1 = self.epsilon_greedy_policy(self.qvalue, s_next, epsilon)
                    a1_num = self.find_anum(a1)
                    # Q-learning 更新公式
                    q_target = r + self.gamma * self.qvalue[s_next, a1_num]
                # 利用TD方法更新动作值函数
                self.qvalue[s, a_num] = self.qvalue[s, a_num] = alpha * (q_target - self.qvalue[s, a_num])
                # 转到下一个状态
                s = s_next
                # 行为策略
                a = self.epsilon_greedy_policy(self.qvalue, s, epsilon)
                count += 1
        return self.qvalue

    def greedy_test(self):
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
        if s == 90 and step_num < 21:
            flag = 2
        return flag

    def q_learning(self, num_iter, alpha, epsilon):
        iter_num = []
        self.qvalue = np.zeros((len(self.yuanyang.states), len(self.yuanyang.actions)))
        for iter in range(num_iter):
            s = 0
            flag = self.greedy_test()
            if flag == 1:
                iter_num.append(iter)
                if len(iter_num) < 2:
                    print("Q-learning第一次完成任务需要的迭代次数为：", iter_num[0])
            if flag == 2:
                print("Q-learning第一次实现最短路径需要的迭代次数为：", iter)
                break

            s_sample = []
            a = self.epsilon_greedy_policy(self.qvalue, s, epsilon)
            t = False
            count = 0
            while False == t and count < 30:
                # 与环境交互得到下一个状态
                s_next, r, t = self.yuanyang.transform(s, a)
                a_num = self.find_anum(a)
                if s_next in s_sample:
                    r = -2
                s_sample.append(s)
                if t == True:
                    q_target = r
                else:
                    a1 = self.greedy_policy(self.qvalue, s_next)
                    a1_num = self.find_anum(a1)
                    q_target = r + self.gamma * self.qvalue[s_next, a1_num]
                self.qvalue[s, a_num] = self.qvalue[s, a_num] + alpha * (q_target - self.qvalue[s, a_num])
                s = s_next
                a = self.epsilon_greedy_policy(self.qvalue, s, epsilon)
                count += 1
        return self.qvalue


if __name__ == "__main__":
    yuanyang = YuanYangEnv()
    td_rl = TD_RL(yuanyang)
    qvalue1 = td_rl.sarsa(num_iter=5000, alpha=0.1, epsilon=0.8)
    qvalue2 = td_rl.q_learning(num_iter=5000, alpha=0.1, epsilon=0.1)
    yuanyang.action_value = qvalue2
    flag = 1
    s = 0
    step_num = 0
    while flag:
        yuanyang.path.append(s)
        a = td_rl.greedy_policy(qvalue2, s)
        print("%d -> %s\t"%(s, a))
        yuanyang.bird_male_position = yuanyang.state_to_position(s)
        yuanyang.render()
        time.sleep(0.25)
        step_num += 1
        s_, r, t = yuanyang.transform(s, a)
        if t == True or step_num > 30 :
            flag = 0
        s = s_
    yuanyang.bird_male_position = yuanyang.state_to_position(s)
    yuanyang.path.append(s)
    yuanyang.render()
    while True:
        yuanyang.render()