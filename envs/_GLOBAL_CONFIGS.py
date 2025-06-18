# global configs
import os

ROOT_PATH = os.path.abspath(__file__)
ROOT_PATH = ROOT_PATH[:ROOT_PATH.rfind('/')]
ROOT_PATH = ROOT_PATH[:ROOT_PATH.rfind('/')+1]

ASSETS_PATH = os.path.join(ROOT_PATH, 'assets/')
EMBODIMENTS_PATH = os.path.join(ASSETS_PATH, 'embodiments/')
CONFIGS_PATH = os.path.join(ROOT_PATH, 'task_config/')


# 世界坐标euler角
# t3d.euler.quat2euler(quat) = theta_x, theta_y, theta_z
# theta_y 控制俯仰角，theta_z控制垂直桌面平面上的旋转
GRASP_DIRECTION_DIC = {
    'left':         [0,      0,   0,    -1],
    'front_left':   [-0.383, 0,   0,    -0.924],
    'front' :       [-0.707, 0,   0,    -0.707],
    'front_right':  [-0.924, 0,   0,    -0.383],
    'right':        [-1,     0,   0,    0],
    'top_down':     [-0.5,   0.5, -0.5, -0.5],
}

WORLD_DIRECTION_DIC = {
    'left':         [0.5,  0.5,  0.5,  0.5],
    'front_left':   [0.65334811, 0.27043713, 0.65334811, 0.27043713],
    'front' :       [0.707, 0,    0.707, 0],
    'front_right':  [0.65334811, -0.27043713,  0.65334811, -0.27043713],
    'right':        [0.5,    -0.5, 0.5,  0.5],
    'top_down':     [0,      0,    1,    0],
}