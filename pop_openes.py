"""Pop-OpenES: a population of independent OpenES instances on one device."""

import copy
from typing import Self

import chex
import jax
import jax.numpy as jnp
import numpy as np
from evorl.agent import AgentState
from evorl.metrics import MetricBase, metric_field
from evorl.types import State
from evorl.workflows import Workflow
from omegaconf import DictConfig, OmegaConf, open_dict, read_write

from openes import OpenESWorkflow


def _array_statistics(values):
    """Summarize host arrays without silently dropping non-finite results."""
    values = np.asarray(values)
    return {
        "min": values.min().item(),
        "max": values.max().item(),
        "mean": values.mean().item(),
        "std": values.std().item(),
    }


def _require_finite(values, name: str, iteration: int):
    invalid = ~np.isfinite(values)
    if invalid.any():
        raise FloatingPointError(
            f"Non-finite {name} at iteration {iteration}; "
            f"invalid indices: {np.argwhere(invalid).tolist()}"
        )


class PopOpenESWorkflowMetric(MetricBase):
    """Progress: updates per member and cumulative training cost over all members.

    Warmup and evaluation are excluded from the sampling counters, following EvoRL.
    """

    iterations: jax.Array = metric_field(
        default_factory=lambda: jnp.zeros((), dtype=jnp.uint32)
    )
    sampled_episodes: jax.Array = metric_field(
        default_factory=lambda: jnp.zeros((), dtype=jnp.uint32)
    )
    sampled_timesteps_m: jax.Array = metric_field(
        default_factory=lambda: jnp.zeros((), dtype=jnp.float32)
    )


class PopOpenESTrainMetric(MetricBase):
    """Training fitness and optimizer diagnostics with a leading member axis.

    ``pop_fitnesses`` has shape [members, candidate policies]; each value is the
    mean training return over that candidate's fitness episodes. Optimizer metrics
    retain their original names inside ``pop_ec_metrics``.
    """

    pop_fitnesses: jax.Array
    pop_ec_metrics: chex.ArrayTree


class PopOpenESEvalMetric(MetricBase):
    """Evaluate the current center policy of every member, then select the best.

    The two population arrays have shape [members] and contain episode means.
    ``best_member_index`` is a zero-based population position; both selected
    scalars belong to that same member. Selection is over this evaluation only.
    Population summary statistics are derived when logging, not stored here.
    """

    pop_episode_returns: jax.Array
    pop_episode_lengths: jax.Array
    best_member_index: jax.Array
    best_episode_return: jax.Array
    best_episode_length: jax.Array


class PopOpenESWorkflow(Workflow):
    """Maintain independent OpenES members and select their best current center.

    Selection never copies parameters, optimizer state, or observation statistics
    between members. ``num_instances`` counts OpenES instances and ``num_iters`` counts
    updates per member. ``target_workflow.pop_size`` counts candidate policies
    inside each instance. Each member starts from an independent random key.
    Training length and evaluation cadence use only the outer config; the inner
    workflow's ``learn()`` loop is never called.
    """

    def __init__(self, config: DictConfig, workflow: OpenESWorkflow):
        super().__init__(config)
        self.workflow = workflow

    @classmethod
    def name(cls):
        return "Pop-OpenES"

    @classmethod
    def build_from_config(cls, config: DictConfig, jit: bool = True) -> Self:
        config = copy.deepcopy(config)
        for field in ("num_instances", "num_iters", "eval_interval"):
            value = config[field]
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{field} must be a positive integer")

        # OpenES is imported directly; only the outer workflow saves checkpoints.
        inner_config = copy.deepcopy(config.target_workflow)
        with read_write(inner_config), open_dict(inner_config):
            inner_config.env = copy.deepcopy(config.env)
            inner_config.checkpoint = {"enable": False}

        # Only the outer population is jitted; the inner workflow uses one device.
        workflow = OpenESWorkflow.build_from_config(
            inner_config, multi_device=False, jit=False
        )
        OmegaConf.set_readonly(config, True)
        instance = cls(config, workflow)
        if jit:
            # Instance-local transforms avoid modifying the single-seed class.
            instance.setup = jax.jit(instance.setup)
            instance.step = jax.jit(instance.step)
            instance.evaluate = jax.jit(instance.evaluate)
        return instance

    def setup(self, key: jax.Array) -> State:
        keys = jax.random.split(key, self.config.num_instances)
        return State(
            pop_workflow_state=jax.vmap(self.workflow.setup)(keys),
            metrics=PopOpenESWorkflowMetric(),
            best_member_index=jnp.int32(-1),
        )

    def step(self, state: State) -> tuple[PopOpenESTrainMetric, State]:
        train_metrics, pop_state = jax.vmap(self.workflow.step)(
            state.pop_workflow_state
        )
        workflow_metrics = PopOpenESWorkflowMetric(
            iterations=state.metrics.iterations + 1,
            sampled_episodes=pop_state.metrics.sampled_episodes.sum(),
            sampled_timesteps_m=pop_state.metrics.sampled_timesteps_m.sum(),
        )
        train_metrics = PopOpenESTrainMetric(
            pop_fitnesses=train_metrics.objectives,
            pop_ec_metrics=train_metrics.ec_metrics,
        )
        return train_metrics, state.replace(
            pop_workflow_state=pop_state,
            metrics=workflow_metrics,
            best_member_index=jnp.int32(-1),
        )

    def evaluate(self, state: State) -> tuple[PopOpenESEvalMetric, State]:
        pop_metrics, pop_state = jax.vmap(self.workflow.evaluate)(
            state.pop_workflow_state
        )
        returns = pop_metrics.episode_return
        best_index = jnp.argmax(returns)
        metrics = PopOpenESEvalMetric(
            pop_episode_returns=returns,
            pop_episode_lengths=pop_metrics.episode_length,
            best_member_index=best_index,
            best_episode_return=returns[best_index],
            best_episode_length=pop_metrics.episode_length[best_index],
        )
        return metrics, state.replace(
            pop_workflow_state=pop_state, best_member_index=best_index
        )

    def get_best_workflow_state(self, state: State) -> State:
        """Extract the winner of the latest evaluation, including its optimizer."""
        index = int(state.best_member_index)
        if index < 0:
            raise ValueError("Evaluate the current state before selecting a member")
        return jax.tree.map(lambda x: x[index], state.pop_workflow_state)

    def get_best_agent_state(self, state: State) -> AgentState:
        """Return the selected center policy with its own observation statistics."""
        return self.workflow._get_pop_center(self.get_best_workflow_state(state))

    def learn(self, state: State) -> State:
        for i in range(int(state.metrics.iterations), self.config.num_iters):
            iteration = i + 1
            train_metrics, state = self.step(state)
            train_data = train_metrics.to_local_dict()
            _require_finite(train_data["pop_fitnesses"], "training fitness", iteration)
            instance_data = state.pop_workflow_state.metrics.to_local_dict()
            record = state.metrics.to_local_dict()
            record["train"] = {
                "pop_fitnesses": _array_statistics(train_data["pop_fitnesses"]),
                "pop_ec_metrics": jax.tree.map(
                    _array_statistics, train_data["pop_ec_metrics"]
                ),
                "members": {
                    str(index): {
                        "fitness": _array_statistics(fitnesses),
                        "sampled_episodes": instance_data["sampled_episodes"][index],
                        "sampled_timesteps_m": instance_data["sampled_timesteps_m"][
                            index
                        ],
                    }
                    for index, fitnesses in enumerate(train_data["pop_fitnesses"])
                },
            }

            if (
                iteration % self.config.eval_interval == 0
                or iteration == self.config.num_iters
            ):
                eval_metrics, state = self.evaluate(state)
                data = eval_metrics.to_local_dict()
                returns = data["pop_episode_returns"]
                lengths = data["pop_episode_lengths"]
                _require_finite(returns, "evaluation return", iteration)
                _require_finite(lengths, "evaluation length", iteration)
                for field in ("pop_episode_returns", "pop_episode_lengths"):
                    values = data[field]
                    data[field] = {**_array_statistics(values), "values": values}
                data["members"] = {
                    str(index): {
                        "episode_return": returns[index],
                        "episode_length": lengths[index],
                    }
                    for index in range(self.config.num_instances)
                }
                record["eval"] = data

            # One event per update keeps progress, training and evaluation aligned.
            self.recorder.write(record, iteration)
            self.checkpoint_manager.save(
                iteration, state, force=iteration == self.config.num_iters
            )
        return state

    def close(self) -> None:
        try:
            self.workflow.close()
        finally:
            super().close()
