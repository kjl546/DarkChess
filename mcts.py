"""蒙特卡洛树搜索"""


import numpy as np
import copy
import random

from matplotlib import pyplot as plt

from config import CONFIG
from game import get_all_legal_moves_darkchess
from game import Game
move_id2move_action, move_action2move_id = get_all_legal_moves_darkchess()

def softmax(x):
    probs = np.exp(x - np.max(x))
    probs /= np.sum(probs)
    return probs


# 定义叶子节点
class TreeNode(object):
    """
    mcts树中的节点，树的子节点字典中，键为动作，值为TreeNode。记录当前节点选择的动作，以及选择该动作后会跳转到的下一个子节点。
    每个节点跟踪其自身的Q，先验概率P及其访问次数调整的u
    """

    def __init__(self, parent, prior_p):
        """
        :param parent: 当前节点的父节点
        :param prior_p:  当前节点被选择的先验概率
        """
        self._parent = parent
        self._children = {} # 从动作到TreeNode的映射
        self._n_visits = 0  # 当前当前节点的访问次数
        self._Q = 0         # 当前节点对应动作的平均动作价值
        self._u = 0         # 当前节点的置信上限         # PUCT算法
        self._P = prior_p

    def expand(self, action_priors):    # 这里把不合法的动作概率全部设置为0
        """通过创建新子节点来展开树"""
        for action, prob in action_priors:
            if action not in self._children:
                self._children[action] =  TreeNode(self, prob)

    def select(self, c_puct, legal_moves):
            """
            [ISMCTS 核心改動]：過濾非法動作
            在子节点中，只考慮「當前決定化盤面」所允許的合法动作。
            """
            # 利用字典推導式，只保留當下合法的分支
            valid_children = {act: node for act, node in self._children.items() if act in legal_moves}
            
            # 如果這個分支是第一次遇到這種隱藏狀態（例如前幾次模擬都翻出車，這次翻出卒），
            # valid_children 會是空的，我們回傳 None 觸發中斷與重新展開。
            if not valid_children:
                return None, None

            return max(valid_children.items(),
                    key=lambda act_node: act_node[1].get_value(c_puct))

    def get_value(self, c_puct):
        """
        计算并返回此节点的值，它是节点评估Q和此节点的先验的组合
        c_puct: 控制相对影响（0， inf）
        """
        self._u = (c_puct * self._P *
                   np.sqrt(self._parent._n_visits) / (1 + self._n_visits))
        return self._Q + self._u

    def update(self, leaf_value):
        """
        从叶节点评估中更新节点值
        leaf_value: 这个子节点的评估值来自当前玩家的视角
        """
        # 统计访问次数
        self._n_visits += 1
        # 更新Q值，取决于所有访问次数的平均树，使用增量式更新方式
        self._Q += 1.0 * (leaf_value - self._Q) / self._n_visits

    # 使用递归的方法对所有节点（当前节点对应的支线）进行一次更新
    def update_recursive(self, leaf_value):
        """就像调用update()一样，但是对所有直系节点进行更新"""
        # 如果它不是根节点，则应首先更新此节点的父节点
        if self._parent:
            self._parent.update_recursive(-leaf_value)
        self.update(leaf_value)

    def is_leaf(self):
        """检查是否是叶节点，即没有被扩展的节点"""
        return self._children == {}

    def is_root(self):
        return self._parent is None


# 蒙特卡洛搜索树
class MCTS(object):

    def __init__(self, policy_value_fn, c_puct=5, n_playout=2000):
        """policy_value_fn: 接收board的盘面状态，返回落子概率和盘面评估得分"""
        self._root = TreeNode(None, 1.0)
        self._policy = policy_value_fn
        self._c_puct = c_puct
        self._n_playout = n_playout

    def _playout(self, state):
            """
            进行一次搜索，根据叶节点的评估值进行反向更新树节点的参数
            """
            node = self._root
            while True:
                if node.is_leaf():
                    break
                
                # [ISMCTS 核心改動]：取得當前盤面的合法動作，傳給 select 進行過濾
                legal_moves = state.availables
                action, next_node = node.select(self._c_puct, legal_moves)
                
                # 如果走到未知的平行時空分支，直接視為葉節點中斷
                if action is None:
                    break
                    
                node = next_node
                state.do_move(action, is_simulate=True)

            # 使用网络评估叶子节点... (以下維持原程式碼不變)
            action_probs, leaf_value = self._policy(state)
            # 查看游戏是否结束
            end, winner = state.game_end()
            if not end:
                node.expand(action_probs)
            else:
                if winner == -1:    # Tie
                    leaf_value = 0.0
                else:
                    leaf_value = (
                        1.0 if winner == state.get_current_player_id() else -1.0
                    )
            node.update_recursive(-leaf_value)

    def get_move_probs(self, state, temp=1e-3, plot_probs=True):
        """按顺序运行所有搜索并返回可用的动作及其相应的概率"""
        for n in range(self._n_playout):
            state_copy = copy.deepcopy(state)
            
            # [ISMCTS 核心改動]：盤面決定化 (Determinization)
            if hasattr(state_copy, 'remain_pieces') and len(state_copy.remain_pieces) > 0:
                # 處理第一步的特殊規則防呆
                if getattr(state_copy, 'first_move', False):
                    # 必須一直洗牌，直到抽出與當前玩家顏色相同的棋子
                    while state_copy.current_player_color not in state_copy.remain_pieces[0]:
                        random.shuffle(state_copy.remain_pieces)
                else:
                    # 非第一步，正常隨機洗牌
                    random.shuffle(state_copy.remain_pieces)
                
            self._playout(state_copy)

            # 跟据根节点处的访问计数来计算移动概率... (以下維持原程式碼不變)
        act_visits= [(act, node._n_visits)
                    for act, node in self._root._children.items()]
        acts, visits = zip(*act_visits)
        act_probs = softmax(1.0 / temp * np.log(np.array(visits) + 1e-10))
        return acts, act_probs

    def update_with_move(self, last_move):
        """
        在当前的树上向前一步，保持我们已经直到的关于子树的一切
        """
        if last_move in self._root._children:
            self._root = self._root._children[last_move]
            self._root._parent = None
        else:
            self._root = TreeNode(None, 1.0)

    def __str__(self):
        return 'MCTS'

# 基于MCTS的AI玩家
class MCTSPlayer(object):


    def __init__(self, policy_value_function, c_puct=3, n_playout=2000, is_selfplay=0):
        self.mcts = MCTS(policy_value_function, c_puct, n_playout)
        self._is_selfplay = is_selfplay
        self.agent = "AI"
        self.playout_count = 0

    def set_player_ind(self, p):
        self.player = p

    # 重置搜索树
    def reset_player(self):
        self.mcts.update_with_move(-1)
        self.playout_count = 0

    def __str__(self):
        return 'MCTS {}'.format(self.player)

    # 得到行动
    def get_action(self, board, temp=1e-3, return_prob=0):
        # 像alphaGo_Zero论文一样使用MCTS算法返回的pi向量
        move_probs = np.zeros(384)

        acts, probs = self.mcts.get_move_probs(board, temp)
        eat_ids = []
        for a in acts:
            y1, x1, y2, x2 = map(int, move_id2move_action[a])
            start = board.state_deque[-1][y1][x1]
            target = board.state_deque[-1][y2][x2]
            if target not in ('一一', '暗棋') and board.current_player_color not in target:
                #print(start,target,sep=" ")
                eat_ids.append(a)
        if eat_ids:
            # 啟發式增益：若當前有可吃子行動，人為放大其機率（AlphaZero 原始算法不含此步，此為項目優化）
            # 這樣可以讓 AI 在訓練初期更具攻擊性
            eat_index = [i for i, a in enumerate(acts) if a in eat_ids]
            probs = probs.copy()
            probs[eat_index] *= 40
            probs /= probs.sum()
        move_probs[list(acts)] = probs
        #if eat_ids:
           #plot_move_probs(acts, probs, eat_ids, "Move Probability (After Amplification)")
        if self._is_selfplay:
            # 前 30 步用高溫度（探索更多），之後降溫
            self.playout_count += 1
            current_temp = 1.0 if self.playout_count <= 30 else 1e-3

            move = np.random.choice(
                acts,
                p=0.8 * probs + 0.2 * np.random.dirichlet(CONFIG['dirichlet'] * np.ones(len(probs)))
            ) if current_temp == 1.0 else np.random.choice(acts, p=probs)

            self.mcts.update_with_move(move)
        else:
            move = np.random.choice(acts, p=probs)
            self.mcts.update_with_move(-1)

        if return_prob:
            return move, move_probs
        else:
            return move
