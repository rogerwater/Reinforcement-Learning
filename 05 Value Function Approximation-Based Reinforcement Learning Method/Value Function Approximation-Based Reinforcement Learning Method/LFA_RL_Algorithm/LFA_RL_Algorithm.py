import time

from yuanyang_env import YuanYangEnv
from yuanyang_env import *

class LFA_RL:
    def __init__(self, yuanyang):
        self.gamma = yuanyang.gamma
        self.yuanyang = yuanyang
        self.theta_tr = np.zeros((400, 1)) * 0.1
        self.theta_fsr = np.zeros((80, 1)) * 0.1

    # 找到动作对应的序号
    def find_anum(self, a):
        for i in range(len(self.yuanyang.actions)):
            if a == self.yuanyang.actions[i]:
                return i

    # 表格特征表示
    def feature_tr(self, s, a):
        phi_s_a = np.zeros((1, 400))
        phi_s_a[0, 100 * a + s] = 1
        return phi_s_a

    # 定义基于表格特征的贪婪策略
    def greedy_policy_tr(self, state):
        qfun = np.array([0, 0, 0, 0]) * 0.1
        # 计算行为值函数Q(s,a) = phi(s,a) * theta
        for i in range(4):
            qfun[i] = np.dot(self.feature_tr(state, i), self.theta_tr)
        amax = qfun.argmax()
        return self.yuanyang.actions[amax]

    # 定义基于表格特征epsilon贪婪策略
    def epsilon_greedy_policy_tr(self, state, epsilon):
        qfun = np.array([0, 0, 0, 0]) * 0.1
        # 计算行为值函数Q(s,a) = phi(s,a) * theta
        for i in range(4):
            qfun[i] = np.dot(self.feature_tr(state, i), self.theta_tr)
        amax = qfun.argmax()
        # 概率部分
        if np.random.uniform() < 1 - epsilon:
            # 最优动作
            return self.yuanyang.actions[amax]
        else:
            return self.yuanyang.actions[int(random.random() * len(self.yuanyang.actions))]

    # 定义基于表格特征的贪婪策略测试函数
    def greedy_test_tr(self):
        s = 0
        s_sample = []
        done = False
        flag = 0
        step_num = 0
        while False == done and step_num < 30:
            a = self.greedy_policy_tr(s)
            # 与环境交互
            s_next, r, done = self.yuanyang.transform(s, a)
            s = s_next
            step_num += 1
        if s == 90:
            flag = 1
        if s == 90 and step_num < 21:
            flag = 2
        return flag

    def q_learning_lfa_tr(self, num_iter, alpha, epsilon):
        iter_num = []
        self.theta_tr = np.zeros((400, 1)) * 0.1
        for iter in range(num_iter):
            s = 0
            flag = self.greedy_test_tr()
            if flag == 1:
                iter_num.append(iter)
                if len(iter_num) < 2:
                    print("q-learning_lfa_tr第一次完成任务需要的迭代次数为：", iter_num[0])
            if flag == 2:
                print("q-learning_lfa_tr第一次实现最短路径需要的迭代次数为：", iter)
                break

            s_sample = []
            a = self.epsilon_greedy_policy_tr(s, epsilon)
            t = False
            count = 0
            while False == t and count < 30:
                # 与环境交互
                s_next, r, t = self.yuanyang.transform(s,a)
                a_num = self.find_anum(a)
                if s_next in s_sample:
                    r = -2
                s_sample.append(s)
                if t == True:
                    q_target = r
                else:
                    # 下一个状态处的最大动作，a1用greedy_policy
                    a1 = self.greedy_policy_tr(s_next)
                    a1_num = self.find_anum(a1)
                    q_target = r + self.gamma * np.dot(self.feature_tr(s_next, a1_num), self.theta_tr)
                self.theta_tr = self.theta_tr + alpha * (q_target - np.dot(self.feature_tr(s, a_num), self.theta_tr))[0,0] * np.transpose(self.feature_tr(s, a_num))
                s = s_next

                # 行为策略
                a = self.epsilon_greedy_policy_tr(s, epsilon)
                count += 1
        return self.theta_tr

    ################# 固定稀疏表示 ###################
    def feature_fsr(self, s, a):
        phi_s_a = np.zeros((1, 80))
        y = int(s / 10)
        x = s - 10 * y
        phi_s_a[0, 20 * a + x] = 1
        phi_s_a[0, 20 * a + 10 + y] = 1
        return phi_s_a

    def greedy_policy_fsr(self, state):
        qfun = np.array([0, 0, 0, 0]) * 0.1
        # 计算行为值函数Q(s,a) = phi(s,a) * theta
        for i in range(4):
            qfun[i] = np.dot(self.feature_fsr(state, i), self.theta_fsr)
        amax = qfun.argmax()
        return self.yuanyang.actions[amax]

    def epsilon_greedy_policy_fsr(self, state, epsilon):
        qfun = np.array([0, 0, 0, 0]) * 0.1
        # 计算行为值函数Q(s,a) = phi(s,a) * theta
        for i in range(4):
            qfun[i] = np.dot(self.feature_fsr(state, i), self.theta_fsr)
        amax = qfun.argmax()
        # 概率部分
        if np.random.uniform() < 1 - epsilon:
            # 最优动作
            return self.yuanyang.actions[amax]
        else:
            return self.yuanyang.actions[int(random.random() * len(self.yuanyang.actions))]

    def greedy_test_fsr(self):
        s = 0
        s_sample = []
        done = False
        flag = 0
        step_num = 0
        while False == done and step_num < 30:
            a = self.greedy_policy_fsr(s)
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

    def q_learning_lfa_fsr(self, num_iter, alpha, epsilon):
        iter_num = []
        self.theta_fsr = np.zeros((80, 1)) * 0.1
        for iter in range(num_iter):
            s = 0
            flag = self.greedy_test_fsr()
            if flag == 1:
                iter_num.append(iter)
                if len(iter_num) < 2:
                    print("q-learning_lfa_fsr第一次完成任务需要的迭代次数为：", iter_num[0])
            if flag == 2:
                print("q-learning_lfa_fsr第一次实现最短路径需要的迭代次数为：", iter)
                break
            s_sample = []
            a = self.epsilon_greedy_policy_fsr(s, epsilon)
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
                    # 下一个状态处的最大动作，a1用greedy_policy
                    a1 = self.greedy_policy_fsr(s_next)
                    a1_num = self.find_anum(a1)
                    q_target = r + self.gamma * np.dot(self.feature_fsr(s_next, a1_num), self.theta_fsr)
                self.theta_fsr = self.theta_fsr + alpha * \
                                 (q_target - np.dot(self.feature_fsr(s, a_num), self.theta_fsr))[
                                     0, 0] * np.transpose(self.feature_fsr(s, a_num))
                s = s_next
                # 行为策略
                a = self.epsilon_greedy_policy_fsr(s, epsilon)
                count += 1

        return self.theta_fsr


if __name__ == "__main__":
    yuanyang = YuanYangEnv()
    lfa_rl = LFA_RL(yuanyang)
    lfa_rl.q_learning_lfa_fsr(num_iter=5000, alpha=0.1, epsilon=0.1)
    lfa_rl.q_learning_lfa_tr(num_iter=5000, alpha=0.1, epsilon=0.1)

    qvalue2 = np.zeros((100, 4))
    qvalue1 = np.zeros((100, 4))
    for i in range(400):
        y = int(i / 100)
        x = i - 100 * y
        qvalue2[x, y] = np.dot(lfa_rl.feature_tr(x, y), lfa_rl.theta_tr)
        qvalue1[x, y] = np.dot(lfa_rl.feature_fsr(x, y), lfa_rl.theta_fsr)

    yuanyang.action_value = qvalue1

    # 测试学到的策略
    flag = 1
    s = 0
    step_num = 0
    while flag:
        yuanyang.path.append(s)
        a = lfa_rl.greedy_policy_fsr(s)
        print("%d -> %s\t"%(s, a))
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
