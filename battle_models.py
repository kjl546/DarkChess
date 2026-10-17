"""
模型對戰腳本：放在組員的 DarkChess 資料夾裡執行（用他的 game.py / mcts.py 規則）。

用法（PowerShell，在 DarkChess 資料夾內）：
  python battle_models.py 模型A.pth 模型B.pth            # A vs B
  python battle_models.py 模型A.pth greedy               # A vs Greedy
  python battle_models.py 模型A.pth random               # A vs Random
可選參數： --games 40 --playout 200
"""
import argparse
import csv
import os
import sys
import time

from game import Board, Game
from mcts import MCTSPlayer
from players import GreedyPlayer, RandomPlayer
from pytorch_net import PolicyValueNet


def make_player(spec, n_playout):
    if spec == 'greedy':
        return GreedyPlayer(), 'greedy'
    if spec == 'random':
        return RandomPlayer(), 'random'
    net = PolicyValueNet(model_file=spec, use_gpu=False)
    return MCTSPlayer(net.policy_value_fn, c_puct=1, n_playout=n_playout), os.path.basename(spec)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('a')
    ap.add_argument('b')
    ap.add_argument('--games', type=int, default=40)
    ap.add_argument('--playout', type=int, default=200)
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

    new = not os.path.exists('battle_models_log.csv')
    with open('battle_models_log.csv', 'a', newline='', encoding='utf-8-sig') as f:
        w = csv.writer(f)
        if new:
            w.writerow(['A', 'B', 'playout', 'games', 'win', 'lose', 'tie', 'A_rate'])
        w.writerow([na, nb, args.playout, args.games, win, lose, tie, f'{rate:.3f}'])


if __name__ == '__main__':
    main()
