import numpy as np
import copy
import time
from config import CONFIG
from collections import deque   # 这个队列用来判断长将或长捉
import random

num2array = dict({
    1: np.array([1, 0, 0, 0, 0, 0, 0]),
    2: np.array([0, 1, 0, 0, 0, 0, 0]),
    3: np.array([0, 0, 1, 0, 0, 0, 0]),
    4: np.array([0, 0, 0, 1, 0, 0, 0]),
    5: np.array([0, 0, 0, 0, 1, 0, 0]),
    6: np.array([0, 0, 0, 0, 0, 1, 0]),
    7: np.array([0, 0, 0, 0, 0, 0, 1]),
    8: np.array([-1, 0, 0, 0, 0, 0, 0]),
    9: np.array([0, -1, 0, 0, 0, 0, 0]),
    10: np.array([0, 0, -1, 0, 0, 0, 0]),
    11: np.array([0, 0, 0, -1, 0, 0, 0]),
    12: np.array([0, 0, 0, 0, -1, 0, 0]),
    13: np.array([0, 0, 0, 0, 0, -1, 0]),
    14: np.array([0, 0, 0, 0, 0, 0, -1]),
    15: np.array([0, 0, 0, 0, 0, 0, 0]),
    16: np.array([1, 1, 1, 1, 1, 1, 1])
})

def array2num(array):
    return list(filter(lambda string: (num2array[string] == array).all(), num2array))[0]

# 压缩存储
def state_list2state_num_array(state_list):
    _state_array = np.zeros([4, 8, 7])
    for i in range(4):
        for j in range(8):
            _state_array[i][j] = num2array[state_list[i][j]]
    return _state_array


# (state, mcts_prob, winner)
# 現在暗棋: ((62,4,8),384,1)
def zip_state_mcts_prob(tuple):
    state, mcts_prob, winner = tuple
    state = state.reshape((8, -1))         # flatten 為 (8, ?)
    mcts_prob = mcts_prob.reshape((2, -1)) # (2, 192)
    state = zip_array(state)
    mcts_prob = zip_array(mcts_prob)
    return state, mcts_prob, winner


def recovery_state_mcts_prob(tuple):
    state, mcts_prob, winner = tuple
    state = recovery_array(state)
    mcts_prob = recovery_array(mcts_prob)
    state = state.reshape((-1, 4, 8))
    mcts_prob = mcts_prob.reshape(384)
    return state, mcts_prob, winner


def zip_array(array, data=0.):  # 压缩成稀疏数组
    zip_res = [[array.shape[0], array.shape[1], 0.0]]
    for i in range(len(array)):
        for j in range(len(array[0])):
            if array[i][j] != data:
                zip_res.append([i, j, array[i][j]])
    return np.array(zip_res)


def recovery_array(array, data=0.):  # 恢复数组
    rows = int(array[0][0])
    cols = int(array[0][1])
    recovery_res = [[data for _ in range(cols)] for _ in range(rows)]
    for i in range(1, len(array)):
        recovery_res[int(array[i][0])][int(array[i][1])] = array[i][2]
    return np.array(recovery_res)