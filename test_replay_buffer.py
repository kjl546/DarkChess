"""Replay persistence regressions; only temporary data, no self-play/models."""
import contextlib
from collections import deque
import io
from pathlib import Path
import pickle
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import collect
import train
import zip_array


class ReplayBufferTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='darkchess-replay-test-')
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.path = self.root / 'buffer.pth'
        config = patch.dict(collect.CONFIG, no_dark_mode=False,
                            new_data_buffer_path=str(self.path), buffer_size=5)
        config.start()
        self.addCleanup(config.stop)
        for name in ['Board', 'Game']:
            p = patch.object(collect, name)
            p.start()
            self.addCleanup(p.stop)
        out = contextlib.redirect_stdout(io.StringIO())
        out.__enter__()
        self.addCleanup(out.__exit__, None, None, None)
        self.pipeline = collect.CollectPipeline()

    def read(self):
        with self.path.open('rb') as f:
            return pickle.load(f)

    def assert_snapshot(self, samples, version):
        snapshot = self.read()
        self.assertEqual(list(snapshot['data_buffer']), samples)
        self.assertEqual(snapshot['iters'], version)
        self.assertEqual(list(self.pipeline.data_buffer), samples)
        self.assertEqual(self.pipeline.iters, version)
        self.assertEqual(self.pipeline.data_buffer.maxlen, 5)

    def test_consecutive_batches_append_each_batch_once(self):
        for i, value in enumerate(['A', 'B', 'C'], 1):
            self.pipeline.save_data([value])
            self.assert_snapshot(['A', 'B', 'C'][:i], i)

    def test_restart_loads_existing_data_and_generation(self):
        self.pipeline.save_data(['A', 'B'])
        self.pipeline = collect.CollectPipeline()
        self.pipeline.save_data(['C'])
        self.assert_snapshot(['A', 'B', 'C'], 2)

    def test_sequential_collectors_use_disk_not_stale_memory(self):
        other = collect.CollectPipeline()
        self.pipeline.save_data(['A'])
        other.save_data(['B'])
        self.pipeline.save_data(['C'])
        self.assert_snapshot(['A', 'B', 'C'], 3)

    def test_capacity_keeps_newest_samples_without_duplication(self):
        self.pipeline.save_data(['A', 'B', 'C', 'D'])
        self.pipeline.save_data(['E', 'F', 'G'])
        self.assert_snapshot(['C', 'D', 'E', 'F', 'G'], 2)

    def test_batch_larger_than_capacity_keeps_its_tail(self):
        self.pipeline.save_data(['old'])
        self.pipeline.save_data(list(range(8)))
        self.assert_snapshot([3, 4, 5, 6, 7], 2)

    def test_legitimate_equal_samples_are_not_deduplicated(self):
        self.pipeline.save_data(['same'])
        self.pipeline.save_data(['same'])
        self.assert_snapshot(['same', 'same'], 2)

    def test_empty_batch_does_not_create_file_or_consume_generation(self):
        self.pipeline.save_data([])
        self.assertFalse(self.path.exists())
        self.pipeline.save_data(['A'])
        before = self.path.read_bytes()
        self.pipeline.save_data([])
        self.assertEqual(self.path.read_bytes(), before)
        self.assert_snapshot(['A'], 1)

    def test_unreadable_old_file_is_preserved(self):
        self.pipeline.save_data(['A'])
        self.path.write_bytes(b'invalid pickle')
        before = self.path.read_bytes()
        with self.assertRaises(pickle.UnpicklingError):
            self.pipeline.save_data(['B'])
        self.assertEqual(self.path.read_bytes(), before)
        self.assertEqual(list(self.pipeline.data_buffer), ['A'])
        self.assertEqual(self.pipeline.iters, 1)

    def test_invalid_existing_metadata_is_not_overwritten(self):
        for snapshot in [{'data_buffer': [], 'iters': -1},
                         {'data_buffer': [], 'iters': True},
                         {'data_buffer': 'not samples', 'iters': 1},
                         {'data_buffer': []}]:
            with self.subTest(snapshot=snapshot):
                self.path.write_bytes(pickle.dumps(snapshot))
                before = self.path.read_bytes()
                with self.assertRaises((ValueError, KeyError)):
                    self.pipeline.save_data(['B'])
                self.assertEqual(self.path.read_bytes(), before)

    def test_write_failure_keeps_old_file_and_memory_and_cleans_temp(self):
        self.pipeline.save_data(['A'])
        before = self.path.read_bytes()
        def broken_dump(data, target, **kwargs):
            target.write(b'partial')
            raise OSError('disk full')
        with patch.object(collect.pickle, 'dump', side_effect=broken_dump):
            with self.assertRaises(OSError):
                self.pipeline.save_data(['B'])
        self.assertEqual(self.path.read_bytes(), before)
        self.assert_snapshot(['A'], 1)
        self.assertEqual(list(self.root.iterdir()), [self.path])

    def test_replace_failure_then_retry_does_not_duplicate_new_batch(self):
        self.pipeline.save_data(['A'])
        with patch.object(collect.os, 'replace', side_effect=PermissionError('busy file')):
            with self.assertRaises(PermissionError):
                self.pipeline.save_data(['B'])
        self.assert_snapshot(['A'], 1)
        self.assertEqual(list(self.root.iterdir()), [self.path])
        self.pipeline.save_data(['B'])
        self.assert_snapshot(['A', 'B'], 2)

    def test_reader_sees_complete_old_file_until_replace(self):
        self.pipeline.save_data(['A'])
        replace = collect.os.replace
        def observe(source, target):
            self.assert_snapshot(['A'], 1)
            with open(source, 'rb') as f:
                pending = pickle.load(f)
            self.assertEqual(list(pending['data_buffer']), ['A', 'B'])
            self.assertEqual(pending['iters'], 2)
            self.assertEqual(Path(source).parent, self.path.parent)
            replace(source, target)
        with patch.object(collect.os, 'replace', side_effect=observe):
            self.pipeline.save_data(['B'])
        self.assert_snapshot(['A', 'B'], 2)

    def test_old_list_format_and_training_reader_remain_compatible(self):
        state = np.zeros((62, 4, 8), dtype=np.float32)
        state[0, 0, 0] = 1
        probs = np.zeros(384, dtype=np.float32)
        probs[0] = 1
        sample = zip_array.zip_state_mcts_prob((state, probs, -1.0))
        self.path.write_bytes(pickle.dumps({'data_buffer': [sample], 'iters': 7}))
        self.pipeline.save_data([sample])
        reader = train.TrainPipeline.__new__(train.TrainPipeline)
        reader.buffer_size = 5
        reader.batch_size = 2
        reader.data_poll_interval = 0.01
        with patch.dict(train.CONFIG, use_redis=False), patch.object(train, 'TRAIN_DATA_PATH', str(self.path)):
            self.assertEqual(reader._wait_for_training_data(7), 8)
        self.assertEqual(len(reader.data_buffer), 2)
        for item in reader.data_buffer:
            restored_state, restored_probs, value = zip_array.recovery_state_mcts_prob(item)
            np.testing.assert_array_equal(restored_state, state)
            np.testing.assert_array_equal(restored_probs, probs)
            self.assertEqual(value, -1.0)

    def test_nonpositive_capacity_is_rejected(self):
        with patch.dict(collect.CONFIG, buffer_size=0), self.assertRaises(ValueError):
            collect.CollectPipeline()


if __name__ == '__main__':
    unittest.main()
