r"""
離線補訓練：用收集程式已存好的 new_data_buffer.pth 重新訓練，不需要收集程式在跑。
規則與 train.py 的 policy_updata 完全相同（batch / epochs / lr / kl 調整），
只是不等新資料、不做內部評估，並且每隔 --save-every 次更新就存一個檔。

用法（在 DarkChess_A 或 DarkChess_B 資料夾內，A、B 參數要一樣）：
  python train_offline.py --init base.pth --updates 150 --save-every 50
輸出：offline_models\offline_batch50.pth、offline_batch100.pth、offline_batch150.pth
"""
import argparse
import os
import pickle
import random
import time

import numpy as np
import torch

from config import CONFIG
import zip_array
from pytorch_net import PolicyValueNet


def update_once(net, data_buffer, state):
    batch_size = CONFIG['batch_size']
    epochs = CONFIG['epochs']
    kl_targ = CONFIG['kl_targ']

    mini_batch = random.sample(data_buffer, batch_size)
    mini_batch = [zip_array.recovery_state_mcts_prob(d) for d in mini_batch]
    state_batch = np.array([d[0] for d in mini_batch], dtype='float32')
    probs_batch = np.array([d[1] for d in mini_batch], dtype='float32')
    winner_batch = np.array([d[2] for d in mini_batch], dtype='float32')

    old_probs, old_v = net.policy_value(state_batch)
    kl = 0.0
    loss = entropy = None
    for _ in range(epochs):
        loss, entropy = net.train_step(state_batch, probs_batch, winner_batch,
                                       state['lr'] * state['mult'])
        new_probs, new_v = net.policy_value(state_batch)
        kl = np.mean(np.sum(old_probs * (np.log(old_probs + 1e-10) - np.log(new_probs + 1e-10)), axis=1))
        if kl > kl_targ * 4:
            break

    if kl > kl_targ * 2 and state['mult'] > 0.1:
        state['mult'] /= 1.5
    elif kl < kl_targ / 2 and state['mult'] < 10:
        state['mult'] *= 1.5
    return kl, loss, entropy


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--init', default='new_current_policy.pth')
    ap.add_argument('--buffer', default='new_data_buffer.pth')
    ap.add_argument('--updates', type=int, default=150)
    ap.add_argument('--save-every', type=int, default=50)
    ap.add_argument('--out-dir', default='offline_models')
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    with open(args.buffer, 'rb') as f:
        data_buffer = pickle.load(f)['data_buffer']
    print(f"載入 buffer：{len(data_buffer)} 筆（{args.buffer}）")
    if len(data_buffer) <= CONFIG['batch_size']:
        raise SystemExit("資料量不足 batch_size，無法訓練")

    net = PolicyValueNet(model_file=args.init)
    print(f"起點模型：{args.init}")

    state = {'lr': 1e-3, 'mult': 1.0}
    t0 = time.time()
    for i in range(1, args.updates + 1):
        kl, loss, entropy = update_once(net, data_buffer, state)
        print(f"[{i}/{args.updates}] kl:{kl:.5f} lr_mult:{state['mult']:.3f} "
              f"loss:{float(loss):.4f} entropy:{float(entropy):.4f} ({time.time() - t0:.0f}s)")
        if i % args.save_every == 0 or i == args.updates:
            path = os.path.join(args.out_dir, f"offline_batch{i}.pth")
            net.save_model(path)
            print(f"已存 {path}")


if __name__ == '__main__':
    main()
