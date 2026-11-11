"""Small real-environment checks for independence, selection and persistence."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import jax
import numpy as np
from evorl.recorders import Recorder
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from pop_openes import PopOpenESWorkflow


class CaptureRecorder(Recorder):
    def __init__(self):
        self.records = []

    def init(self):
        pass

    def write(self, data, step):
        self.records.append((step, data))

    def close(self):
        pass


def make_config(*overrides):
    with initialize_config_dir(
        version_base=None,
        config_dir=str(Path(__file__).resolve().parents[1] / "configs"),
    ):
        return compose(
            config_name="config",
            overrides=[
                "agent=meta/pop-openes",
                "env=brax/fast",
                "num_instances=2",
                "num_iters=2",
                "eval_interval=10",
                "env.max_episode_steps=4",
                "target_workflow.pop_size=4",
                "target_workflow.num_envs=2",
                "target_workflow.episodes_for_fitness=2",
                "target_workflow.num_eval_envs=2",
                "target_workflow.eval_episodes=2",
                "target_workflow.random_timesteps=4",
                "target_workflow.agent_network.actor_hidden_layer_sizes=[4,4]",
                "recorders=[]",
                "checkpoint.enable=false",
                *overrides,
            ],
        )


class PopOpenESTest(unittest.TestCase):
    def assert_tree_close(self, actual, expected):
        self.assertEqual(jax.tree.structure(actual), jax.tree.structure(expected))
        for a, b in zip(jax.tree.leaves(actual), jax.tree.leaves(expected)):
            if jax.dtypes.issubdtype(a.dtype, jax.dtypes.prng_key):
                a, b = jax.random.key_data(a), jax.random.key_data(b)
            np.testing.assert_allclose(a, b, rtol=2e-5, atol=2e-6)

    def test_configuration(self):
        config = make_config()
        self.assertNotIn("search_space", OmegaConf.to_yaml(config))
        self.assertIn("target_workflow", config)
        self.assertNotIn("pop_size", config)
        self.assertEqual(config.target_workflow.ec_lr.init, 0.01)
        self.assertEqual(config.target_workflow.ec_noise_std.init, 0.02)
        self.assertEqual(config.num_instances, 2)
        self.assertEqual(config.target_workflow.pop_size, 4)
        with initialize_config_dir(
            version_base=None,
            config_dir=str(Path(__file__).resolve().parents[1] / "configs"),
        ):
            default = compose(
                config_name="config",
                overrides=[
                    "agent=meta/pop-openes",
                    "env=brax/ant",
                    "num_iters=7",
                    "eval_interval=3",
                    "target_workflow.num_iters=1",
                    "target_workflow.eval_interval=1",
                ],
            )
        self.assertEqual(default.num_iters, 7)
        self.assertEqual(default.eval_interval, 3)
        self.assertGreater(default.num_instances, 0)
        self.assertEqual(default.workflow_cls, "pop_openes.PopOpenESWorkflow")
        self.assertEqual(list(default.recorders), ["log", "aim"])
        self.assertEqual(default.target_workflow.pop_size, 128)

    def test_vmap_matches_independent_runs(self):
        config = make_config()
        original = OmegaConf.to_container(config, resolve=True)
        workflow = PopOpenESWorkflow.build_from_config(config)
        self.addCleanup(workflow.close)
        self.assertEqual(OmegaConf.to_container(config, resolve=True), original)
        self.assertFalse(workflow.workflow.config.checkpoint.enable)
        self.assertEqual(
            workflow.workflow.config.pop_size, config.target_workflow.pop_size
        )
        root_key = jax.random.key(42)
        state = workflow.init(root_key)
        inner = workflow.workflow
        # Compile the sequential reference independently of the vectorized methods.
        setup_one, step_one, eval_one = map(
            jax.jit,
            (
                inner.setup,
                inner.step,
                inner.evaluate,
            ),
        )
        references = [setup_one(key) for key in jax.random.split(root_key, 2)]
        key_data = jax.random.key_data(state.pop_workflow_state.key)
        self.assertFalse(np.array_equal(key_data[0], key_data[1]))
        stats = state.pop_workflow_state.agent_state.obs_preprocessor_state
        self.assertEqual(stats.mean.shape[0], 2)
        self.assertFalse(np.array_equal(stats.mean[0], stats.mean[1]))
        with self.assertRaises(ValueError):
            workflow.get_best_agent_state(state)

        for iteration in range(1, 3):
            train_metrics, state = workflow.step(state)
            self.assertEqual(train_metrics.pop_fitnesses.shape, (2, 4))
            self.assertEqual(int(state.metrics.iterations), iteration)
            self.assertEqual(int(state.metrics.sampled_episodes), iteration * 2 * 4 * 2)
            for index in range(2):
                _, references[index] = step_one(references[index])
                actual = jax.tree.map(
                    lambda x, index=index: x[index], state.pop_workflow_state
                )
                self.assert_tree_close(actual, references[index])

            eval_metrics, state = workflow.evaluate(state)
            for index in range(2):
                ref_metrics, references[index] = eval_one(references[index])
                np.testing.assert_allclose(
                    eval_metrics.pop_episode_returns[index],
                    ref_metrics.episode_return,
                    rtol=2e-5,
                    atol=2e-6,
                )
                actual = jax.tree.map(
                    lambda x, index=index: x[index], state.pop_workflow_state
                )
                self.assert_tree_close(actual, references[index])
            winner = int(np.argmax(eval_metrics.pop_episode_returns))
            self.assertEqual(int(state.best_member_index), winner)
            self.assertEqual(
                float(eval_metrics.best_episode_return),
                float(eval_metrics.pop_episode_returns[winner]),
            )
            self.assertEqual(
                float(eval_metrics.best_episode_length),
                float(eval_metrics.pop_episode_lengths[winner]),
            )
            self.assert_tree_close(
                workflow.get_best_workflow_state(state), references[winner]
            )
            self.assert_tree_close(
                workflow.get_best_agent_state(state),
                inner._get_pop_center(references[winner]),
            )
            self.assertAlmostEqual(
                float(state.metrics.sampled_timesteps_m),
                sum(float(ref.metrics.sampled_timesteps_m) for ref in references),
            )

    def test_final_evaluation_logging_and_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            config = make_config(
                "checkpoint.enable=true",
                f"output_dir={directory}",
                "num_iters=5",
                "eval_interval=2",
                "checkpoint.save_interval_steps=3",
                "target_workflow.num_iters=1",
                "target_workflow.eval_interval=1",
            )
            workflow = PopOpenESWorkflow.build_from_config(config)
            recorder = CaptureRecorder()
            workflow.add_recorders([recorder])
            try:
                with patch.object(
                    workflow.checkpoint_manager,
                    "save",
                    wraps=workflow.checkpoint_manager.save,
                ) as save:
                    state = workflow.learn(workflow.init(jax.random.key(43)))
                self.assertEqual(
                    [call.args[0] for call in save.call_args_list], [1, 2, 3, 4, 5]
                )
                self.assertEqual(
                    [call.kwargs["force"] for call in save.call_args_list],
                    [False, False, False, False, True],
                )
                evaluations = [
                    (step, data["eval"])
                    for step, data in recorder.records
                    if "eval" in data
                ]
                self.assertEqual(
                    [step for step, _ in recorder.records], [1, 2, 3, 4, 5]
                )
                self.assertEqual([step for step, _ in evaluations], [2, 4, 5])
                for step, data in recorder.records:
                    self.assertEqual(data["iterations"], step)
                    self.assertEqual(data["sampled_episodes"], step * 16)
                    members = data["train"]["members"]
                    self.assertEqual(
                        sum(m["sampled_episodes"] for m in members.values()),
                        data["sampled_episodes"],
                    )
                    self.assertAlmostEqual(
                        sum(m["sampled_timesteps_m"] for m in members.values()),
                        data["sampled_timesteps_m"],
                    )
                    self.assertIn("std", data["train"]["pop_fitnesses"])
                self.assertEqual(set(evaluations[0][1]["members"]), {"0", "1"})
                self.assertEqual(int(state.metrics.iterations), 5)
                self.assertEqual(int(state.metrics.sampled_episodes), 80)
                np.testing.assert_array_equal(
                    state.pop_workflow_state.metrics.iterations, [5, 5]
                )
                evaluation = evaluations[0][1]
                values = evaluation["pop_episode_returns"]["values"]
                self.assertEqual(evaluation["best_episode_return"], values.max())
                self.assertAlmostEqual(
                    evaluation["pop_episode_returns"]["mean"], values.mean()
                )
                self.assertAlmostEqual(
                    evaluation["pop_episode_returns"]["std"], values.std()
                )
                self.assertTrue(workflow.config.checkpoint.enable)
                self.assertFalse(workflow.workflow.config.checkpoint.enable)
                workflow.checkpoint_manager.wait_until_finished()
                self.assertEqual(
                    list(workflow.checkpoint_manager.all_steps()), [1, 3, 5]
                )
                saved = workflow.checkpoint_manager.restore(3, state)
                self.assertEqual(int(saved.metrics.iterations), 3)
                self.assertEqual(int(saved.best_member_index), -1)
                with self.assertRaises(ValueError):
                    workflow.get_best_workflow_state(saved)
                restored = workflow.checkpoint_manager.restore(5, state)
                self.assert_tree_close(restored, state)
                # Continuing training invalidates the old selection.
                _, next_state = workflow.step(state)
                with self.assertRaises(ValueError):
                    workflow.get_best_workflow_state(next_state)
            finally:
                workflow.close()

    def test_eager_single_seed_without_normalization(self):
        workflow = PopOpenESWorkflow.build_from_config(
            make_config("num_instances=1", "target_workflow.normalize_obs=false"),
            jit=False,
        )
        self.addCleanup(workflow.close)
        state = workflow.init(jax.random.key(44))
        _, state = workflow.step(state)
        metrics, state = workflow.evaluate(state)
        self.assertEqual(int(state.best_member_index), 0)
        self.assertEqual(float(metrics.pop_episode_returns.std()), 0.0)
        self.assertEqual(int(state.metrics.sampled_episodes), 8)
        workflow.get_best_agent_state(state)

    def test_episode_batches_are_validated_before_environment_creation(self):
        cases = (
            (
                (
                    "target_workflow.episodes_for_fitness=3",
                    "target_workflow.num_envs=2",
                ),
                "episodes_for_fitness.*divisible.*num_envs",
            ),
            (
                ("target_workflow.eval_episodes=3", "target_workflow.num_eval_envs=2"),
                "eval_episodes.*divisible.*num_eval_envs",
            ),
            (("target_workflow.num_envs=0",), "num_envs must be a positive integer"),
            (
                ("target_workflow.eval_episodes=0",),
                "eval_episodes must be a positive integer",
            ),
        )
        for overrides, message in cases:
            with (
                self.subTest(overrides=overrides),
                patch("openes.create_env") as create_env,
            ):
                with self.assertRaisesRegex(AssertionError, message):
                    PopOpenESWorkflow.build_from_config(make_config(*overrides))
                create_env.assert_not_called()

    def test_non_finite_results_fail_before_recording(self):
        workflow = PopOpenESWorkflow.build_from_config(
            make_config("num_iters=1", "eval_interval=1")
        )
        self.addCleanup(workflow.close)
        recorder = CaptureRecorder()
        workflow.add_recorders([recorder])
        initial = workflow.init(jax.random.key(46))
        train_metrics, trained = workflow.step(initial)
        bad_train = train_metrics.replace(
            pop_fitnesses=train_metrics.pop_fitnesses.at[0, 0].set(jax.numpy.nan)
        )
        with (
            patch.object(workflow, "step", return_value=(bad_train, trained)),
            self.assertRaisesRegex(FloatingPointError, "training fitness.*iteration 1"),
        ):
            workflow.learn(initial)
        self.assertEqual(recorder.records, [])

        eval_metrics, evaluated = workflow.evaluate(trained)
        bad_eval = eval_metrics.replace(
            pop_episode_returns=eval_metrics.pop_episode_returns.at[0].set(
                jax.numpy.inf
            )
        )
        with (
            patch.object(workflow, "step", return_value=(train_metrics, trained)),
            patch.object(workflow, "evaluate", return_value=(bad_eval, evaluated)),
            self.assertRaisesRegex(
                FloatingPointError, "evaluation return.*iteration 1"
            ),
        ):
            workflow.learn(initial)
        self.assertEqual(recorder.records, [])

    def test_invalid_batch_and_schedule(self):
        for override in ("num_instances=0", "num_iters=0", "eval_interval=0"):
            with self.subTest(override=override), self.assertRaises(ValueError):
                PopOpenESWorkflow.build_from_config(make_config(override))


if __name__ == "__main__":
    unittest.main()
