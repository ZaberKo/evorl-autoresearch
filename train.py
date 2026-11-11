"""Train local OpenES or its Pop-OpenES baseline with one GPU per job."""

import logging
import os
from contextlib import nullcontext

import hydra
from hydra.core.hydra_config import HydraConfig
from hydra.types import RunMode
from omegaconf import DictConfig, OmegaConf

from accelerator_utils import gpu_visibility_settings
from hydra_utils import get_output_dir, set_absl_log_level, set_omegaconf_resolvers
from recorder_setup import setup_recorders

logger = logging.getLogger("train")

set_absl_log_level("warning")
set_omegaconf_resolvers()


def set_gpu_id() -> None:
    """Select one GPU before importing JAX, including for a single Hydra run."""
    # Allocate on demand on shared GPUs unless the user explicitly opts in.
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    hydra_config = HydraConfig.get()
    job_id = hydra_config.job.num if hydra_config.mode == RunMode.MULTIRUN else 0
    settings, device, num_gpus = gpu_visibility_settings(job_id)
    if job_id >= num_gpus:
        logger.warning("Multiple jobs are sharing a GPU.")
    logger.info(
        "Using %s device %s (%s)",
        settings["JAX_PLATFORMS"],
        device["local_hardware_id"],
        device["device_kind"],
    )
    os.environ.update(settings)


@hydra.main(version_base=None, config_path="configs", config_name="config")
def train(config: DictConfig) -> None:
    """Build and run a local OpenES experiment."""
    workflow_classes = {
        "openes.OpenESWorkflow",
        "pop_openes.PopOpenESWorkflow",
    }
    if config.get("workflow_cls") not in workflow_classes:
        raise ValueError("Select agent=openes or agent=meta/pop-openes.")
    if config.get("env") is None:
        raise ValueError("Select an environment, for example env=brax/ant.")

    set_gpu_id()

    import jax

    if config.workflow_cls == "pop_openes.PopOpenESWorkflow":
        from pop_openes import PopOpenESWorkflow as workflow_cls
    else:
        from openes import OpenESWorkflow as workflow_cls

    config.output_dir = str(get_output_dir())
    logger.info("config:\n%s", OmegaConf.to_yaml(config, resolve=True))

    devices = jax.local_devices()
    if len(devices) != 1:
        raise ValueError(f"Each training job requires one GPU; found {devices}.")

    workflow = workflow_cls.build_from_config(config, jit=config.jit)
    try:
        workflow.add_recorders(setup_recorders(config, workflow.name(), parallel=True))
        context = (
            jax.set_mesh(workflow.mesh) if workflow.mesh is not None else nullcontext()
        )
        with context:
            state = workflow.init(jax.random.key(config.seed))
            workflow.learn(state)
    finally:
        workflow.close()


if __name__ == "__main__":
    train()
