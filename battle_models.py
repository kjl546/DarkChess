"""
模型對戰腳本：放在組員的 DarkChess 資料夾裡執行（用他的 game.py / mcts.py 規則）。

用法（PowerShell，在 DarkChess 資料夾內）：
  python battle_models.py 模型A.pth 模型B.pth            # A vs B
  python battle_models.py 模型A.pth greedy               # A vs Greedy
  python battle_models.py 模型A.pth random               # A vs Random
可選參數： --games 40 --playout 200 --log battle_models_log.csv

天眼(oracle)模型：
  程式會依 .pth 的輸入 channel 自動判斷 62 / 76。76 channel 模型預設「開天眼」
  (讀真實暗棋排列、搜尋也直接取真實翻棋結果)，量的是「有天眼時的上限」。
  檔名後加 :blind 則把天眼平面清成 0 且不開天眼，量的是「沒有天眼時的實力」：
    python battle_models.py oracle_B.pth:blind new_best_policy.pth
"""
import argparse
import csv
import inspect
import os
import time

import torch

from game import Board, Game
from mcts import MCTSPlayer
from players import GreedyPlayer, RandomPlayer
from pytorch_net import PolicyValueNet


def make_player(spec, n_playout):
    if spec == 'greedy':
        return GreedyPlayer(), 'greedy'
    if spec == 'random':
        return RandomPlayer(), 'random'
    path, _, mode = spec.partition(':')
    kwargs = {}
    label = os.path.basename(path)
    if 'input_channels' in inspect.signature(PolicyValueNet.__init__).parameters:
        # 新版 pytorch_net：明確指定 channel，避免 62 channel 模型被預設補成 76
        channels = torch.load(path, map_location='cpu')['conv_block.weight'].shape[1]
        kwargs['input_channels'] = channels
        if channels > 62:
            kwargs['blind'] = (mode == 'blind')
            label += ':blind' if mode == 'blind' else ':oracle'
    net = PolicyValueNet(model_file=path, use_gpu=False, **kwargs)
    return MCTSPlayer(net.policy_value_fn, c_puct=1, n_playout=n_playout), label


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('a')
    ap.add_argument('b')
    ap.add_argument('--games', type=int, default=40)
    ap.add_argument('--playout', type=int, default=200)
    ap.add_argument('--log', default='battle_models_log.csv')   # 多個視窗平行跑時，各用各的檔名
    args = ap.parse_args()

    pa, na = make_player(args.a, args.playout)
    pb, nb = make_player(args.b, args.playout)
    game = Game(Board())

    win = lose = tie = 0
    t0 = time.time()
    for i in range(args.games):
        # 偶數場 A 當 player1，奇數場 B 當 player1（輪流，避免位置/先手偏差）
        if i % 2 == 0:
            w = game.start_play(pa, pb, start_player=1, is_shown=0)
            a_id = 1
        else:
            w = game.start_play(pb, pa, start_player=1, is_shown=0)
            a_id = 2
        if w == -1:
            tie += 1
        elif w == a_id:
            win += 1
        else:
            lose += 1
        for p in (pa, pb):
            if hasattr(p, 'reset_player'):
                p.reset_player()
        print(f'[{i+1}/{args.games}] A勝{win} A負{lose} 和{tie}  '
              f'已耗時 {(time.time()-t0)/60:.1f} 分', flush=True)

    rate = (win + 0.5 * tie) / args.games
    print(f'\n結果：{na} vs {nb}  →  {win}勝 {lose}負 {tie}和，A 勝率={rate:.3f}')

    new = not os.path.exists(args.log)
    with open(args.log, 'a', newline='', encoding='utf-8-sig') as f:
        w = csv.writer(f)
        if new:
            w.writerow(['A', 'B', 'playout', 'games', 'win', 'lose', 'tie', 'A_rate'])
        w.writerow([na, nb, args.playout, args.games, win, lose, tie, f'{rate:.3f}'])


if __name__ == '__main__':
    main()
