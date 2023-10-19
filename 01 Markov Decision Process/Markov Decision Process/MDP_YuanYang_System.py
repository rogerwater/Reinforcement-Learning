# 问题描述：雄性鸳鸯绕过障碍物找到雌性鸳鸯
# 将鸳鸯系统的整个状态空间离散为只包含100个状态的状态空间，雄性鸳鸯在每个状态处都有4个可行的动作，雄性鸳鸯需要做一系列的最优动作才能找到雌性鸳鸯——序贯决策问题
# 优势函数：A(s,a) = q_pi(s,a) - v_pi(s)

import pygame
from load import *
import random
import numpy as np
