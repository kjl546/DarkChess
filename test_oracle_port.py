"""天眼(oracle)移植測試：python -m unittest test_oracle_port -v
遊戲邏輯測試不需要 torch；網路測試(62->76 輸出不變)需要 torch，沒有就自動跳過。
"""
import copy
import random
import tempfile
import os
import unittest
from collections import Counter

import numpy as np

from config import CONFIG
from game import Board, Game, move_id2move_action
from mcts import MCTSPlayer

try:
    import torch  # noqa: F401
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


class UniformNet:
    """假網路：均勻 policy、value=0，宣告自己是 76 channel 天眼網路。"""
    input_channels = 76
    oracle_active = True

    def policy_value_fn(self, board):
        legal = board.availables
        return zip(legal, np.ones(len(legal)) / len(legal)), np.array([[0.0]])


def new_board(start_player=1):
    b = Board()
    b.init_board(start_player)
    return b


def reveal_moves(b):
    return [m for m in b.availables if b.is_reveal_move(m)]


class TestOracleBoard(unittest.TestCase):
    def setUp(self):
        CONFIG['no_dark_mode'] = False

    def test_true_board_matches_remaining(self):
        b = new_board()
        flat = [p for row in b.true_board for p in row]
        self.assertEqual(len(flat), 32)
        self.assertEqual(Counter(flat), Counter(b.remain_pieces))

    def test_oracle_plane_and_state_shape(self):
        b = new_board()
        self.assertEqual(b.oracle_state().shape, (76, 4, 8))
        self.assertEqual(int(b.get_oracle_plane().sum()), 32)
        self.assertEqual(int(b.oracle_state(blind=True)[62:].sum()), 0)
        np.testing.assert_array_equal(b.oracle_state()[:62], b.current_state())
        m = reveal_moves(b)[0]
        y, x, _, _ = map(int, move_id2move_action[m])
        b.do_move(m)
        plane = b.get_oracle_plane()
        self.assertEqual(int(plane.sum()), 31)
        self.assertEqual(float(plane[:, y, x].sum()), 0.0)

    def test_real_flip_follows_true_board(self):
        for _ in range(30):
            b = new_board(random.choice([1, 2]))
            m = random.choice(reveal_moves(b))
            y, x, _, _ = map(int, move_id2move_action[m])
            b.do_move(m)
            piece = b.state_deque[-1][y][x]
            self.assertEqual(piece, b.true_board[y][x])
            self.assertEqual(Counter(p for r in b.true_board for p in r),
                             Counter(p for r in b.state_deque[-1] for p in r if p != '暗棋')
                             + Counter(b.remain_pieces))

    def test_oracle_search_matches_real_outcome(self):
        for _ in range(30):
            b = new_board(random.choice([1, 2]))
            # 先翻幾步讓局面不在 first_move
            for _ in range(random.randint(0, 4)):
                b.do_move(random.choice(reveal_moves(b)))
            m = random.choice(reveal_moves(b))
            y, x, _, _ = map(int, move_id2move_action[m])
            sim = copy.deepcopy(b)
            sim.oracle_search = True
            outcomes = sim.get_reveal_outcomes(m)
            self.assertEqual(len(outcomes), 1)
            self.assertEqual(outcomes[0][1], 1.0)
            sim.do_move(m, is_simulate=True)
            b.do_move(m)
            self.assertEqual(sim.state_deque[-1][y][x], b.state_deque[-1][y][x])
            self.assertEqual(outcomes[0][0], b.state_deque[-1][y][x])

    def test_normal_search_still_samples(self):
        b = new_board()
        for _ in range(3):
            b.do_move(random.choice(reveal_moves(b)))
        sim = copy.deepcopy(b)           # oracle_search 預設 False
        outcomes = sim.get_reveal_outcomes(reveal_moves(sim)[0])
        self.assertGreater(len(outcomes), 1)
        self.assertAlmostEqual(sum(p for _, p in outcomes), 1.0)


class TestOracleSelfPlay(unittest.TestCase):
    def test_auto_detect_and_selfplay(self):
        net = UniformNet()
        player = MCTSPlayer(net.policy_value_fn, c_puct=3, n_playout=8, is_selfplay=1)
        self.assertTrue(player.mcts._oracle)
        plain = MCTSPlayer(lambda b: None, n_playout=8)
        self.assertFalse(plain.mcts._oracle)
        game = Game(Board())
        winner, data = game.start_self_play(player, temp=1, use_oracle=True)
        data = list(data)
        self.assertGreater(len(data), 0)
        for state, probs, z in data:
            self.assertEqual(np.asarray(state).shape, (76, 4, 8))
            self.assertEqual(np.asarray(probs).shape, (384,))


@unittest.skipUnless(HAS_TORCH, 'needs torch')
class TestOracleNet(unittest.TestCase):
    def test_62_to_76_is_identity_at_start(self):
        from pytorch_net import PolicyValueNet
        base = PolicyValueNet(input_channels=62, use_gpu=False, device='cpu')
        # 讓 BN running stats 也不是預設值
        base.policy_value_net.train()
        for _ in range(3):
            base.policy_value_net(torch.randn(8, 62, 4, 8))
        path = os.path.join(tempfile.mkdtemp(), 'base62.pth')
        base.save_model(path)
        wide = PolicyValueNet(model_file=path, input_channels=76, use_gpu=False, device='cpu')
        x62 = torch.randn(4, 62, 4, 8)
        x76 = torch.cat([x62, torch.randn(4, 14, 4, 8)], dim=1)   # 天眼平面亂填也不應影響
        base.policy_value_net.eval(); wide.policy_value_net.eval()
        with torch.no_grad():
            p0, v0 = base.policy_value_net(x62)
            p1, v1 = wide.policy_value_net(x76)
        self.assertTrue(torch.allclose(p0, p1, atol=1e-5))
        self.assertTrue(torch.allclose(v0, v1, atol=1e-5))

    def test_wide_cannot_load_into_narrow(self):
        from pytorch_net import PolicyValueNet
        wide = PolicyValueNet(input_channels=76, use_gpu=False, device='cpu')
        path = os.path.join(tempfile.mkdtemp(), 'w76.pth')
        wide.save_model(path)
        with self.assertRaises(ValueError):
            PolicyValueNet(model_file=path, input_channels=62, use_gpu=False, device='cpu')

    def test_oracle_active_flags(self):
        from pytorch_net import PolicyValueNet
        self.assertTrue(PolicyValueNet(input_channels=76, use_gpu=False, device='cpu').oracle_active)
        self.assertFalse(PolicyValueNet(input_channels=76, blind=True, use_gpu=False, device='cpu').oracle_active)
        self.assertFalse(PolicyValueNet(input_channels=62, use_gpu=False, device='cpu').oracle_active)


if __name__ == '__main__':
    unittest.main()
