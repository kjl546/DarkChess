"""使用收集到数据进行训练"""
import glob
import os
import random
import re
from collections import defaultdict, deque

import numpy as np
import pickle
import time

import zip_array
from config import CONFIG
from game import Game, Board
from mcts import MCTSPlayer
from mcts_pure import MCTS_Pure
if CONFIG['use_redis']:
    import my_redis, redis
    import zip_array

if CONFIG['use_frame'] == 'paddle':
    from paddle_net import PolicyValueNet
elif CONFIG['use_frame'] == 'pytorch':
    from pytorch_net import PolicyValueNet
else:
    print('暂不支持您选择的框架')

# 根據 no_dark_mode 設定不同檔案位置
if CONFIG.get('no_dark_mode', True):
    TRAIN_DATA_PATH = CONFIG.get('train_data_buffer_path_no_dark', 'train_data_buffer_no_dark.pth')
    BEST_POLICY_PATH = 'best_policy_no_dark.pth'
    CURRENT_POLICY_PATH = 'current_policy_no_dark.pth'
    MODELS_DIR = 'models_no_dark'
    REDIS_KEY = 'train_data_buffer_no_dark'
    REDIS_ITERS_KEY = 'iters_no_dark'
else:
    TRAIN_DATA_PATH = CONFIG.get('new_data_buffer_path', 'new_data_buffer.pth')
    BEST_POLICY_PATH = 'new_best_policy.pth'
    CURRENT_POLICY_PATH = 'new_current_policy.pth'
    MODELS_DIR = 'new_models'
    REDIS_KEY = 'new_train_data_buffer'
    REDIS_ITERS_KEY = 'new_iters'

class TrainPipeline:
    def __init__(self, init_model=None):
        # 模型只載入一次；載入失敗時停止，不改用隨機權重。
        os.makedirs(MODELS_DIR, exist_ok=True)
        model_file = self._select_model(init_model)
        self.loaded_model_path = model_file
        self.start_iter = self._checkpoint_step(model_file)
        if model_file is None:
            self.policy_value_net = PolicyValueNet()
            print("沒有既有模型，從第 0 批建立新模型")
        else:
            self.policy_value_net = PolicyValueNet(model_file=model_file)
            print(f"載入權重: {model_file}；已完成批次: {self.start_iter}")
            print("目前是權重續訓；optimizer 與學習率調整狀態重新初始化")

        self.board = Board()
        self.game = Game(self.board)
        self.n_playout = CONFIG['play_out']
        self.c_puct = CONFIG['c_puct_for_fight']
        self.learn_rate = 1e-3
        self.lr_multiplier = 1
        self.temp = 1.0
        self.batch_size = CONFIG['batch_size']
        self.epochs = CONFIG['epochs']
        self.kl_targ = CONFIG['kl_targ']
        self.check_freq = 100
        self.game_batch_num = CONFIG['game_batch_num']
        self.best_win_ratio = 0.0
        self.pure_mcts_playout_num = 50  # 用於 pure MCTS baseline
        if CONFIG['use_redis']:
            self.redis_cli = my_redis.get_redis_cli()
        self.buffer_size = CONFIG['buffer_size']
        self.data_buffer = deque(maxlen=self.buffer_size)

        self.best_policy_net = None
        self.iters = None
        self.data_poll_interval = float(CONFIG.get(
            'train_data_poll_interval', 5 if CONFIG['use_redis'] else 30))
        if self.batch_size <= 0 or self.buffer_size < self.batch_size:
            raise ValueError('buffer_size must be at least a positive batch_size')
        if not np.isfinite(self.data_poll_interval) or self.data_poll_interval <= 0:
            raise ValueError('train_data_poll_interval must be finite and positive')

    @staticmethod
    def _checkpoint_step(path):
        """批次只取自實際載入的 checkpoint；一般權重檔無批次資訊。"""
        if path is None:
            return 0
        match = re.fullmatch(r'new_current_policy_batch(\d+)\.pth', os.path.basename(path))
        return int(match.group(1)) if match else 0

    @classmethod
    def _select_model(cls, init_model):
        """指定檔 > 最大批次 checkpoint > current 權重 > 新模型。"""
        if init_model is not None:
            if not os.path.isfile(init_model):
                raise FileNotFoundError(f'指定的模型不存在: {init_model}')
            return os.fspath(init_model)
        checkpoints = [path for path in glob.glob(
            os.path.join(MODELS_DIR, 'new_current_policy_batch*.pth'))
            if re.fullmatch(r'new_current_policy_batch\d+\.pth', os.path.basename(path))
            and os.path.isfile(path)]
        if checkpoints:
            return max(checkpoints, key=cls._checkpoint_step)
        if os.path.isfile(CURRENT_POLICY_PATH):
            return CURRENT_POLICY_PATH
        return None

    def _read_training_data(self):
        """每次等待都重新讀取來源，不依賴會過期的記憶體長度。"""
        if CONFIG['use_redis']:
            before = self.redis_cli.get(REDIS_ITERS_KEY)
            data = my_redis.get_list_range(self.redis_cli, REDIS_KEY, 0, -1)
            generation = self.redis_cli.get(REDIS_ITERS_KEY)
            if before != generation:
                raise ValueError('讀取期間 Redis 資料版本改變，稍後重試')
        else:
            with open(TRAIN_DATA_PATH, 'rb') as data_file:
                snapshot = pickle.load(data_file)
            data = snapshot['data_buffer']
            generation = snapshot['iters']
        if generation is None:
            raise ValueError('資料尚無 iters 版本，稍後重試')
        return deque(data, maxlen=self.buffer_size), generation

    def _wait_for_training_data(self, last_iters):
        """版本未訓練過且樣本至少 batch_size 時才返回。

        資料不足不消耗該版本；等候下一輪會重新讀檔或讀 Redis。
        KeyboardInterrupt 不會被攔截，使用者可隨時停止等待。
        """
        while True:
            try:
                data, generation = self._read_training_data()
            except Exception as exc:
                print(f'資料暫不可讀，稍後重新載入: {exc}')
            else:
                self.data_buffer = data
                if len(data) < self.batch_size:
                    print(f'樣本不足 ({len(data)}/{self.batch_size})，稍後重新載入')
                elif generation == last_iters:
                    print(f'資料版本 {generation} 已訓練，等待新資料')
                else:
                    self.iters = generation
                    return generation
            time.sleep(self.data_poll_interval)

    def _fight(self, player1, player2, n_games=30):
        """雙方對戰，返回勝率、詳細勝負"""
        win_cnt = defaultdict(int)
        show = 0
        for i in range(n_games):
            if i % 2 == 0:
                winner = self.game.start_play(player1, player2, start_player=1, is_shown=show)
                if winner == 1:
                    win_cnt["p1"] += 1
                elif winner == 2:
                    win_cnt["p2"] += 1
                else:
                    win_cnt["tie"] += 1
            else:
                winner = self.game.start_play(player2, player1, start_player=1, is_shown=show)
                if winner == 1:
                    win_cnt["p2"] += 1
                elif winner == 2:
                    win_cnt["p1"] += 1
                else:
                    win_cnt["tie"] += 1
            print("第",i+1,"場結束",sep="")
        win_ratio = (win_cnt["p1"] + 0.5 * win_cnt["tie"]) / n_games
        return win_ratio, win_cnt

    def multi_evaluate(self, n_games=30):
        """同時測試 current vs best, current vs pureMCTS, best vs pureMCTS"""
        from players import RandomPlayer

        print("開始測試")
        # 印出檔案時間
        for path, name in [(CURRENT_POLICY_PATH, "current_policy"), (BEST_POLICY_PATH, "best_policy")]:
            if os.path.exists(path):
                print(f"[檔案時間] {name} 最後修改: {time.ctime(os.path.getmtime(path))}")

        current_player = MCTSPlayer(self.policy_value_net.policy_value_fn,
                                    c_puct=self.c_puct,
                                    n_playout=100)

        # baseline: pure MCTS
        random_player = RandomPlayer()

        # best_policy
        best_player = None
        if os.path.exists(BEST_POLICY_PATH):
            try:
                best_net = PolicyValueNet(model_file=BEST_POLICY_PATH)
                best_player = MCTSPlayer(best_net.policy_value_fn,
                                         c_puct=self.c_puct,
                                         n_playout=100)
            except Exception as e:
                print("[EVAL] 載入 best_policy 失敗:", e)

        results = {}

        # 1. current vs best
        if best_player:
            print("存在best_policy 開始evaluate",BEST_POLICY_PATH)
            r, cnt = self._fight(current_player, best_player, n_games=n_games)
            results["current_vs_best"] = (r, cnt)
            print(f"[EVAL] current vs best → {cnt}, 勝率={r:.3f}")
            results["current_vs_random"] = (0, 0)
        else:
            print("目前不存在best_policy")
            r, cnt = self._fight(current_player, random_player, n_games=10)
            results["current_vs_random"] = (r, cnt)
            print(f"[EVAL] current vs random → {cnt}, 勝率={r:.3f}")
        # 2. current vs pureMCTS


        return results

    def policy_evaluate(self, n_games=30):
        """
        Evaluate current policy against best policy if exists,
        otherwise against a pure MCTS baseline.
        返回胜率 (float)
        """
        # 重新嘗試載入 disk 上的 best model（若被外部程序更新）
        if os.path.exists(BEST_POLICY_PATH):
            try:
                self.best_policy_net = PolicyValueNet(model_file=BEST_POLICY_PATH)
                print(f"[EVAL] 使用已存在的 best model: {BEST_POLICY_PATH}")
            except Exception as e:
                print(f"[EVAL] 載入 best model 失敗，改用 pure MCTS baseline: {e}")
                self.best_policy_net = None
        else:
            print("[EVAL] 無 best model 檔案，改用 pure MCTS baseline")

        # 構造對手
        current_mcts_player = MCTSPlayer(self.policy_value_net.policy_value_fn,
                                         c_puct=self.c_puct,
                                         n_playout=max(50, self.n_playout // 4))  # 評估可以用較少 playout
        if self.best_policy_net is not None:
            best_mcts_player = MCTSPlayer(self.best_policy_net.policy_value_fn,
                                          c_puct=self.c_puct,
                                          n_playout=max(50, self.n_playout // 4))
            baseline_is_pure_mcts = False
            print("[EVAL] baseline = best_policy_net")
        else:
            # 用 pure MCTS 作 baseline（不使用神經網路）
            best_mcts_player = MCTS_Pure(n_playout=self.pure_mcts_playout_num)
            baseline_is_pure_mcts = True
            print(f"[EVAL] baseline = pure MCTS ({self.pure_mcts_playout_num} playouts)")

        win_cnt = defaultdict(int)
        for i in range(n_games):
            # 交換先手/後手
            if i % 2 == 0:
                winner = self.game.start_play(current_mcts_player, best_mcts_player, start_player=1, is_shown=0)
                if winner == 1:
                    win_cnt["current"] += 1
                elif winner == 2:
                    win_cnt["best"] += 1
                else:
                    win_cnt["tie"] += 1
            else:
                winner = self.game.start_play(best_mcts_player, current_mcts_player, start_player=1, is_shown=0)
                # 注意：當 baseline 為 pure MCTS，best_mcts_player 並非 "best" 但計數方式一致
                if winner == 1:
                    # winner 1 對於傳入的 (best, current) 代表 best 贏
                    win_cnt["best"] += 1
                elif winner == 2:
                    win_cnt["current"] += 1
                else:
                    win_cnt["tie"] += 1

        win_ratio = (win_cnt["current"] + 0.5 * win_cnt["tie"]) / n_games
        print(f"[模型對戰] new vs baseline → win: {win_cnt['current']}, lose: {win_cnt['best']}, tie: {win_cnt['tie']}")
        print(f"[EVAL] 勝率（current vs baseline）: {win_ratio:.3f}")
        return win_ratio

    def policy_updata(self):
        mini_batch = random.sample(self.data_buffer, self.batch_size)
        mini_batch = [zip_array.recovery_state_mcts_prob(data) for data in mini_batch]
        state_batch = np.array([data[0] for data in mini_batch], dtype='float32')
        mcts_probs_batch = np.array([data[1] for data in mini_batch], dtype='float32')
        winner_batch = np.array([data[2] for data in mini_batch], dtype='float32')

        old_probs, old_v = self.policy_value_net.policy_value(state_batch)
        kl = 0.0
        loss = None
        entropy = None
        for i in range(self.epochs):
            loss, entropy = self.policy_value_net.train_step(state_batch, mcts_probs_batch, winner_batch,
                                                             self.learn_rate * self.lr_multiplier)
            new_probs, new_v = self.policy_value_net.policy_value(state_batch)
            kl = np.mean(np.sum(old_probs * (
                np.log(old_probs + 1e-10) - np.log(new_probs + 1e-10)), axis=1))
            if kl > self.kl_targ * 4:
                break

        if kl > self.kl_targ * 2 and self.lr_multiplier > 0.1:
            self.lr_multiplier /= 1.5
        elif kl < self.kl_targ / 2 and self.lr_multiplier < 10:
            self.lr_multiplier *= 1.5

        # 保險：避免除以零
        explained_var_old = 1.0
        explained_var_new = 1.0
        try:
            explained_var_old = 1 - np.var(winner_batch - old_v.flatten()) / np.var(winner_batch)
            explained_var_new = 1 - np.var(winner_batch - new_v.flatten()) / np.var(winner_batch)
        except:
            pass

        print(f"kl:{kl:.5f}, lr_multiplier:{self.lr_multiplier:.3f}, loss:{loss}, entropy:{entropy}, "
              f"explained_var_old:{explained_var_old:.9f}, explained_var_new:{explained_var_new:.9f}")
        return loss, entropy

    def run(self):
        try:
            last_iters = None

            for i in range(self.start_iter, self.game_batch_num + self.start_iter):
                generation = self._wait_for_training_data(last_iters)
                print(f"訓練批次 {i + 1}，累積樣本數：{len(self.data_buffer)}")
                loss, entropy = self.policy_updata()
                # 真正完成更新才記錄版本，資料不足或讀取失敗不會消耗批次。
                last_iters = generation
                time.sleep(CONFIG['train_update_interval'])

                # 定期評估
                if (i + 1) % self.check_freq == 0:
                    # 重新載入 best model（如果檔案存在），避免 state mismatch
                    if os.path.exists(BEST_POLICY_PATH):
                        try:
                            self.best_policy_net = PolicyValueNet(model_file=BEST_POLICY_PATH)
                        except Exception as e:
                            print("重新載入 best model 失敗:", e)
                            self.best_policy_net = None
                    else:
                        print("不存在best model")
                    results = self.multi_evaluate()
                    # 取 current vs best 的勝率來決定是否更新 best_policy
                    win_ratio = results.get("current_vs_best", results["current_vs_random"])[0]
                    print(f"第 {i + 1} 批自對弈訓練，勝率：{win_ratio:.3f}")

                    # 儲存條件：勝率要超過閾值且比歷史最佳好
                    if win_ratio > 0.55 :
                        print(f"🎯 新最佳策略發現！勝率 {win_ratio * 100:.2f}")
                        self.best_win_ratio = win_ratio
                        self.policy_value_net.save_model(CURRENT_POLICY_PATH)
                        self.policy_value_net.save_model(BEST_POLICY_PATH)
                    else:
                        print("勝率不足 55% 或沒有超越歷史最佳，跳過保存 best_policy")

                    # checkpoint
                    self.policy_value_net.save_model(f'{MODELS_DIR}/new_current_policy_batch{i + 1}.pth')
        except KeyboardInterrupt:
            print('\n⛔️ 手動終止訓練')


if __name__ == '__main__':
    if CONFIG['use_frame'] in ('paddle', 'pytorch'):
        # 不強制傳入 current 檔，才能啟用最新 checkpoint 的自動選擇。
        training_pipeline = TrainPipeline()
        training_pipeline.run()
    else:
        raise ValueError('不支援的訓練框架')
