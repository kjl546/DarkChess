"""Training-flow regressions: temporary data and mocked models; no training job."""
import contextlib
import io
import os
import pickle
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

import train


class TrainingFlowTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='darkchess-training-test-')
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.models = self.root / 'models'
        self.models.mkdir()
        self.current = self.root / 'current.pth'
        self.data = self.root / 'data.pth'
        for target, value in [('MODELS_DIR', str(self.models)),
                              ('CURRENT_POLICY_PATH', str(self.current)),
                              ('TRAIN_DATA_PATH', str(self.data))]:
            p = patch.object(train, target, value)
            p.start()
            self.addCleanup(p.stop)
        p = patch.dict(train.CONFIG, use_redis=False, batch_size=4, buffer_size=10,
                       game_batch_num=2, train_data_poll_interval=0.01, train_update_interval=0)
        p.start()
        self.addCleanup(p.stop)
        p = patch.object(train, 'PolicyValueNet')
        self.net = p.start()
        self.addCleanup(p.stop)
        for name in ['Board', 'Game']:
            p = patch.object(train, name)
            p.start()
            self.addCleanup(p.stop)
        self.output = contextlib.redirect_stdout(io.StringIO())
        self.output.__enter__()
        self.addCleanup(self.output.__exit__, None, None, None)

    def checkpoint(self, step):
        path = self.models / f'new_current_policy_batch{step}.pth'
        path.touch()
        return path

    def snapshot(self, size, generation):
        with self.data.open('wb') as f:
            pickle.dump({'data_buffer': list(range(size)), 'iters': generation}, f)

    def test_auto_resume_loads_highest_batch_once_even_if_older_mtime(self):
        old = self.checkpoint(9)
        latest = self.checkpoint(100)
        os.utime(latest, (1, 1))
        os.utime(old, (100, 100))
        self.current.touch()
        pipeline = train.TrainPipeline()
        self.net.assert_called_once_with(model_file=str(latest))
        self.assertIs(pipeline.policy_value_net, self.net.return_value)
        self.assertEqual(pipeline.start_iter, 100)

    def test_explicit_checkpoint_has_priority_and_its_own_step(self):
        explicit = self.checkpoint(5)
        self.checkpoint(100)
        pipeline = train.TrainPipeline(init_model=explicit)
        self.net.assert_called_once_with(model_file=str(explicit))
        self.assertEqual(pipeline.start_iter, 5)

    def test_explicit_plain_weights_do_not_inherit_other_checkpoint_step(self):
        self.current.touch()
        self.checkpoint(100)
        pipeline = train.TrainPipeline(init_model=self.current)
        self.net.assert_called_once_with(model_file=str(self.current))
        self.assertEqual(pipeline.start_iter, 0)

    def test_current_weights_fallback(self):
        self.current.touch()
        pipeline = train.TrainPipeline()
        self.net.assert_called_once_with(model_file=str(self.current))
        self.assertEqual(pipeline.start_iter, 0)

    def test_no_model_initializes_once_from_zero(self):
        pipeline = train.TrainPipeline()
        self.net.assert_called_once_with()
        self.assertIsNone(pipeline.loaded_model_path)
        self.assertEqual(pipeline.start_iter, 0)

    def test_invalid_checkpoint_name_is_not_used(self):
        (self.models / 'new_current_policy_batchbroken.pth').touch()
        train.TrainPipeline()
        self.net.assert_called_once_with()

    def test_missing_explicit_model_does_not_start_random_training(self):
        with self.assertRaises(FileNotFoundError):
            train.TrainPipeline(init_model=self.root / 'missing.pth')
        self.net.assert_not_called()

    def test_bad_checkpoint_is_not_silently_replaced(self):
        path = self.checkpoint(50)
        self.net.side_effect = RuntimeError('incompatible weights')
        with self.assertRaisesRegex(RuntimeError, 'incompatible weights'):
            train.TrainPipeline()
        self.net.assert_called_once_with(model_file=str(path))

    def test_poll_reloads_insufficient_data_and_accepts_exact_batch(self):
        pipeline = train.TrainPipeline()
        self.snapshot(2, 1)
        with patch.object(train.time, 'sleep', side_effect=lambda _: self.snapshot(4, 2)) as sleep:
            self.assertEqual(pipeline._wait_for_training_data(None), 2)
        sleep.assert_called_once_with(0.01)
        self.assertEqual(len(pipeline.data_buffer), 4)

    def test_insufficient_snapshot_does_not_consume_its_version(self):
        pipeline = train.TrainPipeline()
        self.snapshot(2, 1)
        with patch.object(train.time, 'sleep', side_effect=lambda _: self.snapshot(4, 1)):
            self.assertEqual(pipeline._wait_for_training_data(None), 1)

    def test_trained_generation_waits_for_new_generation(self):
        pipeline = train.TrainPipeline()
        self.snapshot(4, 1)
        with patch.object(train.time, 'sleep', side_effect=lambda _: self.snapshot(5, 2)) as sleep:
            self.assertEqual(pipeline._wait_for_training_data(1), 2)
        self.assertEqual(sleep.call_count, 1)

    def test_missing_and_partial_files_are_reloaded(self):
        for corrupt in [False, True]:
            with self.subTest(corrupt=corrupt):
                pipeline = train.TrainPipeline()
                if corrupt:
                    self.data.write_bytes(b'not a complete pickle')
                elif self.data.exists():
                    self.data.unlink()
                with patch.object(train.time, 'sleep', side_effect=lambda _: self.snapshot(4, 3)):
                    self.assertEqual(pipeline._wait_for_training_data(None), 3)

    def test_wait_is_interruptible(self):
        pipeline = train.TrainPipeline()
        self.snapshot(0, 0)
        with patch.object(train.time, 'sleep', side_effect=KeyboardInterrupt):
            with self.assertRaises(KeyboardInterrupt):
                pipeline._wait_for_training_data(None)

    def test_run_reloads_then_updates_once_per_generation(self):
        pipeline = train.TrainPipeline()
        self.snapshot(2, 1)
        pipeline.policy_updata = Mock(return_value=(0.5, 0.2))
        def progress(delay):
            if delay == 0.01:
                self.snapshot(4, 1)  # Enough samples in the same, unconsumed generation.
            elif pipeline.policy_updata.call_count == 1:
                self.snapshot(4, 2)
        with patch.object(train.time, 'sleep', side_effect=progress):
            pipeline.run()
        self.assertEqual(pipeline.policy_updata.call_count, 2)
        self.assertEqual(pipeline.iters, 2)

    def test_redis_wait_replaces_snapshot_instead_of_appending_duplicates(self):
        pipeline = train.TrainPipeline()
        pipeline.redis_cli = Mock()
        pipeline.redis_cli.get.side_effect = [b'1', b'1', b'2', b'2']
        helper = Mock()
        helper.get_list_range.side_effect = [[0, 1], [0, 1, 2, 3]]
        with patch.dict(train.CONFIG, use_redis=True), \
                patch.object(train, 'my_redis', helper, create=True), \
                patch.object(train.time, 'sleep'):
            self.assertEqual(pipeline._wait_for_training_data(None), b'2')
        self.assertEqual(list(pipeline.data_buffer), [0, 1, 2, 3])
        self.assertTrue(all(call.args[2:] == (0, -1)
                            for call in helper.get_list_range.call_args_list))
        pipeline.redis_cli.lpop.assert_not_called()

    def test_redis_generation_changed_during_read_retries(self):
        pipeline = train.TrainPipeline()
        pipeline.redis_cli = Mock()
        pipeline.redis_cli.get.side_effect = [b'1', b'2', b'2', b'2']
        helper = Mock()
        helper.get_list_range.return_value = [0, 1, 2, 3]
        with patch.dict(train.CONFIG, use_redis=True), \
                patch.object(train, 'my_redis', helper, create=True), \
                patch.object(train.time, 'sleep') as sleep:
            self.assertEqual(pipeline._wait_for_training_data(None), b'2')
        self.assertEqual(sleep.call_count, 1)

    def test_import_does_not_start_training_or_create_model_directory(self):
        # Fresh process: exercise the real module-level guard, no mocked TrainPipeline.
        env = dict(os.environ)
        module_dir = str(Path(train.__file__).resolve().parent)
        project_dir = str(Path(__file__).resolve().parent)
        env['PYTHONPATH'] = os.pathsep.join([module_dir, project_dir, os.getcwd(), env.get('PYTHONPATH', '')])
        result = subprocess.run([sys.executable, '-B', '-c', 'import train; print("import-only-ok")'],
                                cwd=self.tmp.name, env=env, capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('import-only-ok', result.stdout)
        self.assertFalse((self.root / 'new_models').exists())


if __name__ == '__main__':
    unittest.main()
