
import numpy as np
import sys
import random
import pygame
import flappy_bird_utils
import pygame.surfarray as surfarray
from pygame.locals import *
from itertools import cycle

FPS = 30
SCREENWIDTH  = 288
SCREENHEIGHT = 512

pygame.init() # 初始化 pygame 各模块
FPSCLOCK = pygame.time.Clock() # 控制每秒帧数
SCREEN = pygame.display.set_mode((SCREENWIDTH, SCREENHEIGHT))
pygame.display.set_caption('Flappy Bird')

# 载入图片/音效/碰撞掩码
IMAGES, SOUNDS, HITMASKS = flappy_bird_utils.load()

PIPEGAPSIZE = 100   # 上下水管之间的缺口高度
BASEY = SCREENHEIGHT * 0.79   # 地面所在的 y 坐标

# 取出小鸟和水管的宽高（后面碰撞检测要用）
PLAYER_WIDTH  = IMAGES['player'][0].get_width()
PLAYER_HEIGHT = IMAGES['player'][0].get_height()
PIPE_WIDTH    = IMAGES['pipe'][0].get_width()
PIPE_HEIGHT   = IMAGES['pipe'][0].get_height()
BACKGROUND_WIDTH = IMAGES['background'].get_width()

# 让小鸟翅膀动画循环播放：0→1→2→1→0→...
PLAYER_INDEX_GEN = cycle([0, 1, 2, 1])

# GameState：核心环境类
#   外部通过 __init__ 重置环境
#   通过 frame_step(action) 推进一帧并返回（图像，奖励，是否结束）
class GameState:
    def __init__(self):
        # 初始化各种计数/位置/速度
        self.score = self.playerIndex = self.loopIter = 0
        self.playerx = int(SCREENWIDTH * 0.2)          # 小鸟固定 x
        self.playery = int((SCREENHEIGHT - PLAYER_HEIGHT) / 2)
        self.basex = 0
        self.baseShift = IMAGES['base'].get_width() - BACKGROUND_WIDTH  # 地面移动用

        # 先放两根水管，一根在屏幕右侧，另一根再右移半个屏
        newPipe1 = getRandomPipe()
        newPipe2 = getRandomPipe()
        self.upperPipes = [
            {'x': SCREENWIDTH,                    'y': newPipe1[0]['y']},
            {'x': SCREENWIDTH + (SCREENWIDTH / 2), 'y': newPipe2[0]['y']},
        ]
        self.lowerPipes = [
            {'x': SCREENWIDTH,                    'y': newPipe1[1]['y']},
            {'x': SCREENWIDTH + (SCREENWIDTH / 2), 'y': newPipe2[1]['y']},
        ]

        # 小鸟运动参数
        self.pipeVelX      =  -4   # 水管左移速度
        self.playerVelY    =   0   # 小鸟当前 y 速度
        self.playerMaxVelY =  10   # 最大下落速度
        self.playerMinVelY =  -8   # 最大上升速度
        self.playerAccY    =   1   # 重力加速度（每帧）
        self.playerFlapAcc =  -9   # 拍翅膀时获得的向上速度
        self.playerFlapped = False # 标记本帧是否刚拍过翅膀

    # frame_step：外部调用此函数即可“走一步”
    # input_actions 长度=2  one-hot 编码：
    #   [1,0] 什么都不做
    #   [0,1] 拍翅膀
    # 返回：
    #   image_data : 当前屏幕 RGB 数组（用于 CNN）
    #   reward     : 即时奖励  +0.1 存活 / +1 得分 / -1 撞了
    #   terminal   : 是否游戏结束
    def frame_step(self, input_actions):
        pygame.event.pump()   # 防止窗口假死

        reward = 0.1
        terminal = False

        # 合法性检查：必须且只能有一个动作
        if sum(input_actions) != 1:
            raise ValueError('Multiple input actions!')

        # 如果动作是“拍翅膀”，且小鸟还没飞出顶部，则给向上速度
        if input_actions[1] == 1:
            if self.playery > -2 * PLAYER_HEIGHT:
                self.playerVelY = self.playerFlapAcc
                self.playerFlapped = True
                # SOUNDS['wing'].play()   # 先静音，方便训练

        # 得分检测
        playerMidPos = self.playerx + PLAYER_WIDTH / 2
        for pipe in self.upperPipes:
            pipeMidPos = pipe['x'] + PIPE_WIDTH / 2
            # 小鸟刚越过水管中心线时加 1 分
            if pipeMidPos <= playerMidPos < pipeMidPos + 4:
                self.score += 1
                # SOUNDS['point'].play()
                reward = 1

        # 动画更新
        # 每 3 帧换一张翅膀图
        if (self.loopIter + 1) % 3 == 0:
            self.playerIndex = next(PLAYER_INDEX_GEN)
        self.loopIter = (self.loopIter + 1) % 30
        # 地面循环左移
        self.basex = -((-self.basex + 100) % self.baseShift)

        # 小鸟物理
        # 没拍翅膀时受重力加速
        if self.playerVelY < self.playerMaxVelY and not self.playerFlapped:
            self.playerVelY += self.playerAccY
        if self.playerFlapped:
            self.playerFlapped = False
        # 更新 y 坐标（不能穿过地面）
        self.playery += min(self.playerVelY,
                            BASEY - self.playery - PLAYER_HEIGHT)
        if self.playery < 0:
            self.playery = 0

        #  水管移动
        for uPipe, lPipe in zip(self.upperPipes, self.lowerPipes):
            uPipe['x'] += self.pipeVelX
            lPipe['x'] += self.pipeVelX
        # 当第一根水管快离开左侧时，在右侧再生成一根新水管
        if 0 < self.upperPipes[0]['x'] < 5:
            newPipe = getRandomPipe()
            self.upperPipes.append(newPipe[0])
            self.lowerPipes.append(newPipe[1])
        # 把完全离开屏幕的水管删掉
        if self.upperPipes[0]['x'] < -PIPE_WIDTH:
            self.upperPipes.pop(0)
            self.lowerPipes.pop(0)

        # 碰撞检测
        isCrash = checkCrash({'x': self.playerx, 'y': self.playery,
                              'index': self.playerIndex},
                             self.upperPipes, self.lowerPipes)
        if isCrash:
            # SOUNDS['hit'].play()
            # SOUNDS['die'].play()
            terminal = True
            self.__init__()   # 立即重置环境
            reward = -1

        # 绘制屏幕
        SCREEN.blit(IMAGES['background'], (0, 0))
        for uPipe, lPipe in zip(self.upperPipes, self.lowerPipes):
            SCREEN.blit(IMAGES['pipe'][0], (uPipe['x'], uPipe['y']))
            SCREEN.blit(IMAGES['pipe'][1], (lPipe['x'], lPipe['y']))
        SCREEN.blit(IMAGES['base'], (self.basex, BASEY))
        SCREEN.blit(IMAGES['player'][self.playerIndex],
                    (self.playerx, self.playery))

        # 把当前屏幕转成 RGB 数组供外部算法使用
        image_data = pygame.surfarray.array3d(pygame.display.get_surface())
        pygame.display.update()
        FPSCLOCK.tick(FPS)
        return image_data, reward, terminal


# 工具函数：随机生成一对水管（上/下）的 y 坐标
def getRandomPipe():
    """returns a randomly generated pipe"""
    # 缺口中心 y 的可选范围（像素）
    gapYs = [20, 30, 40, 50, 60, 70, 80, 90]
    index = random.randint(0, len(gapYs)-1)
    gapY = gapYs[index]

    gapY += int(BASEY * 0.2)      # 整体下移一点
    pipeX = SCREENWIDTH + 10      # 从右侧外进入

    return [
        {'x': pipeX, 'y': gapY - PIPE_HEIGHT},  # 上水管底部 y
        {'x': pipeX, 'y': gapY + PIPEGAPSIZE},  # 下水管顶部 y
    ]

# 碰撞检测总入口：检测小鸟是否撞地面或水管
def checkCrash(player, upperPipes, lowerPipes):
    """returns True if player collders with base or pipes."""
    pi = player['index']
    player['w'] = IMAGES['player'][0].get_width()
    player['h'] = IMAGES['player'][0].get_height()

    # 撞地面
    if player['y'] + player['h'] >= BASEY - 1:
        return True
    else:
        playerRect = pygame.Rect(player['x'], player['y'],
                                 player['w'], player['h'])
        for uPipe, lPipe in zip(upperPipes, lowerPipes):
            uPipeRect = pygame.Rect(uPipe['x'], uPipe['y'], PIPE_WIDTH, PIPE_HEIGHT)
            lPipeRect = pygame.Rect(lPipe['x'], lPipe['y'], PIPE_WIDTH, PIPE_HEIGHT)

            pHitMask = HITMASKS['player'][pi]
            uHitmask = HITMASKS['pipe'][0]
            lHitmask = HITMASKS['pipe'][1]

            # 像素级碰撞检测
            uCollide = pixelCollision(playerRect, uPipeRect, pHitMask, uHitmask)
            lCollide = pixelCollision(playerRect, lPipeRect, pHitMask, lHitmask)

            if uCollide or lCollide:
                return True
    return False


# 像素级碰撞：先矩形相交，再对比 alpha 掩码
def pixelCollision(rect1, rect2, hitmask1, hitmask2):
    """Checks if two objects collide and not just their rects"""
    rect = rect1.clip(rect2)          # 相交矩形
    if rect.width == 0 or rect.height == 0:
        return False

    # 把相交区域换算到各自局部坐标
    x1, y1 = rect.x - rect1.x, rect.y - rect1.y
    x2, y2 = rect.x - rect2.x, rect.y - rect2.y

    # 逐像素查掩码
    for x in range(rect.width):
        for y in range(rect.height):
            if hitmask1[x1+x][y1+y] and hitmask2[x2+x][y2+y]:
                return True
    return False
