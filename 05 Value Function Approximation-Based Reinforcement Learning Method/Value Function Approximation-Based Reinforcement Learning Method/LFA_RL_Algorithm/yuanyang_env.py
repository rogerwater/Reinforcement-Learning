# 问题描述：雄性鸳鸯绕过障碍物找到雌性鸳鸯
# 将鸳鸯系统的整个状态空间离散为只包含100个状态的状态空间，雄性鸳鸯在每个状态处都有4个可行的动作，雄性鸳鸯需要做一系列的最优动作才能找到雌性鸳鸯——序贯决策问题
# 优势函数：A(s,a) = q_pi(s,a) - v_pi(s)

import pygame
import random
import numpy as np

from pygame import QUIT


class YuanYangEnv:
    def __init__(self):
        # 定义MDP中的相关元素
        self.states = []
        for i in range(0, 100):
            self.states.append(i)
        self.actions = ['e', 's', 'w', 'n']
        self.gamma = 0.95
        self.value = np.zeros((10, 10))
        self.action_value = np.zeros((100,4))

        # 设置渲染属性
        self.viewer = None
        self.FPSCLOCK = pygame.time.Clock()
        # 屏幕大小
        self.screen_size = (1200, 900)
        self.bird_position = (0, 0)
        self.limit_distance_x = 120
        self.limit_distance_y = 90
        self.obstacle_size = [120, 90]
        self.obstacle1_x = []
        self.obstacle1_y = []
        self.obstacle2_x = []
        self.obstacle2_y = []
        for i in range(8):
            # 第一个障碍物
            self.obstacle1_x.append(360)
            if i <= 3:
                self.obstacle1_y.append(90 * i)
            else:
                self.obstacle1_y.append(90 * (i + 2))
            # 第二个障碍物
            self.obstacle2_x.append(720)
            if i <= 4:
                self.obstacle2_y.append(90 * i)
            else:
                self.obstacle2_y.append(90 * (i + 2))
        self.bird_male_init_position = [0.0, 0.0]
        self.bird_male_position = [0, 0]
        self.bird_female_init_position = [1080, 0]

        self.path = []

    def collide(self, state_position):
        # flag, flag1, flag2 分别表示是否与障碍物、障碍物墙1、障碍物墙2发生碰撞
        flag = 1
        flag1 = 1
        flag2 = 1

        # 判断第一个障碍物
        dx = []
        dy = []
        for i in range(8):
            dx1 = abs(self.obstacle1_x[i] - state_position[0])
            dx.append(dx1)
            dy1 = abs(self.obstacle1_y[i] - state_position[1])
            dy.append(dy1)
        mindx = min(dx)
        mindy = min(dy)
        if mindx >= self.limit_distance_x or mindy >= self.limit_distance_y:
            flag1 = 0

        # 判断第二个障碍物
        second_dx = []
        second_dy = []
        for i in range(8):
            dx2 = abs(self.obstacle2_x[i] - state_position[0])
            second_dx.append(dx2)
            dy2 = abs(self.obstacle2_y[i] - state_position[1])
            second_dy.append(dy2)
        mindx = min(second_dx)
        mindy = min(second_dy)
        if mindx >= self.limit_distance_x or mindy >= self.limit_distance_y:
            flag2 = 0

        if flag1 == 0 and flag2 == 0:
            flag = 0

        # 判断是否和边界碰撞
        if state_position[0] > 1080 or state_position[0] < 0 or state_position[1] > 810 or state_position[1] < 0:
            flag = 1

        return flag

    def find(self, state_position):
        flag = 0
        if abs(state_position[0] - self.bird_female_init_position[0]) < self.limit_distance_x and abs(state_position[1] - self.bird_female_init_position[1]) < self.limit_distance_y:
            flag = 1
        return flag

    def state_to_position(self, state):
        i = int(state / 10)
        j = state % 10
        position = [0, 0]
        position[0] = 120 * i
        position[1] = 90 * j
        return position

    def position_to_state(self, position):
        i = position[0] / 120
        j = position[1] / 90
        return int(i * 10 + j)

    def reset(self):
        # 随机产生初始状态
        flag1 = 1
        flag2 = 2
        while flag1 or flag2 ==1:
            state = self.states[int(random.random() * len(self.states))]
            state_position = self.state_to_position(state)
            flag1 = self.collide(state_position)
            flag2 = self.find(state_position)

        return state

    # 定义状态转移概率模型P和回报函数
    def transform(self, state, action):
        # 将当前状态转化为坐标
        current_position = self.state_to_position(state)
        next_position = [0, 0]
        flag_collide = 0
        flag_find = 0

        # 判断当前坐标是否与障碍物碰撞
        flag_collide = self.collide(current_position)

        # 判断状态是否是终点
        flag_find = self.find(current_position)

        if flag_collide == 1:
            return state, -10, True
        if flag_find == 1:
            return state, 10, True

        # 状态转移
        if action == 'e':
            next_position[0] = current_position[0] + 120
            next_position[1] = current_position[1]
        if action == 's':
            next_position[0] = current_position[0]
            next_position[1] = current_position[1] + 90
        if action == 'w':
            next_position[0] = current_position[0] - 120
            next_position[1] = current_position[1]
        if action == 'n':
            next_position[0] = current_position[0]
            next_position[1] = current_position[1] - 90

        # 判断next_state是否与障碍物碰撞
        flag_collide = self.collide(next_position)
        # 如果碰撞，那么回报为-10,并结束
        if flag_collide == 1:
            return self.position_to_state(current_position), -10, True

        # 判断是否是终点
        flag_find = self.find(next_position)
        if flag_find == 1:
            return self.position_to_state(next_position), 10, True
        return self.position_to_state(next_position), -0.1, False

    # 判断是否要结束游戏
    def gameover(self):
        for event in pygame.event.get():
            if event.type == QUIT:
                exit()

    # 渲染游戏
    def render(self):
        if self.viewer is None:
            pygame.init()
            # 画一个窗口
            self.viewer = pygame.display.set_mode(self.screen_size, 0, 32)
            pygame.display.set_caption("YuanYang")
            # 下载图片
            self.bird_male = pygame.image.load('./LFA_RL_Algorithm/bird.png').convert_alpha()
            self.bird_female = pygame.image.load('./LFA_RL_Algorithm/bird.png').convert_alpha()
            self.background = pygame.image.load('./LFA_RL_Algorithm/background.png').convert_alpha()
            self.obstacle = pygame.image.load('./LFA_RL_Algorithm/obstacle.png').convert_alpha()
            self.viewer.blit(self.bird_male, self.bird_male_init_position)
            self.viewer.blit(self.bird_female, self.bird_female_init_position)
            self.viewer.blit(self.background, (0, 0))
            self.font = pygame.font.SysFont('times', 15)

        self.viewer.blit(self.background, (0, 0))
        # 画直线
        for i in range(11):
            pygame.draw.lines(self.viewer, (255, 255, 255), True, ((120 * i, 0), (120 * i, 900)), 1)
            pygame.draw.lines(self.viewer, (255, 255, 255), True, ((0, 90 * i), (1200, 90 * i)), 1)
        # 画障碍物
        for i in range(8):
            self.viewer.blit(self.obstacle, (self.obstacle1_x[i], self.obstacle1_y[i]))
            self.viewer.blit(self.obstacle, (self.obstacle2_x[i], self.obstacle2_y[i]))
        # 画小鸟
        self.viewer.blit(self.bird_female, self.bird_female_init_position)
        self.viewer.blit(self.bird_male, self.bird_male_position)
        # 画值函数
        # for i in range(10):
        #    for j in range(10):
        #        surface = self.font.render(str(round(float(self.value[i,j]), 3)), True, (255, 0, 0))
        #        self.viewer.blit(surface, (120 * i + 5, 90 * j + 70))
        # 画动作值函数
        for i in range(100):
            y = int(i / 10)
            x = i % 10
            # 往东的值函数
            surface = self.font.render(str(round(float(self.action_value[i,0]), 2)), True, (0, 0, 0))
            self.viewer.blit(surface, (120 * x + 80, 90 * y + 45))
            # 往南的值函数
            surface = self.font.render(str(round(float(self.action_value[i, 1]), 2)), True, (0, 0, 0))
            self.viewer.blit(surface, (120 * x + 50, 90 * y + 70))
            # 往西的值函数
            surface = self.font.render(str(round(float(self.action_value[i, 2]), 2)), True, (0, 0, 0))
            self.viewer.blit(surface, (120 * x + 10, 90 * y + 45))
            # 往北的值函数
            surface = self.font.render(str(round(float(self.action_value[i, 3]), 2)), True, (0, 0, 0))
            self.viewer.blit(surface, (120 * x + 50, 90 * y + 10))

        # 画路径点
        for i in range(len(self.path)):
            rec_position = self.state_to_position(self.path[i])
            pygame.draw.rect(self.viewer, [255, 0, 0], [rec_position[0], rec_position[1], 120, 90], 3)
            surface = self.font.render(str(i), True, (255, 0, 0))
            self.viewer.blit(surface, (rec_position[0] + 5, rec_position[1] + 5))

        pygame.display.update()
        self.gameover()
        self.FPSCLOCK.tick(30)


if __name__ == "__main__":
    yy = YuanYangEnv()
    yy.render()
    while True:
        for event in pygame.event.get():
            if event.type == QUIT:
                exit()



