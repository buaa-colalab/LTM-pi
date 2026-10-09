import dataclasses
import functools
import logging
import os
import platform
from typing import Any

import etils.epath as epath
import flax.nnx as nnx
import flax.traverse_util as traverse_util
import jax
import jax.experimental
import jax.numpy as jnp
import numpy as np
import optax
import tqdm_loggable.auto as tqdm
import wandb
from flax.training import common_utils

import openpi.models.model as _model
import openpi.shared.array_typing as at
import openpi.shared.nnx_utils as nnx_utils
import openpi.training.checkpoints as _checkpoints
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader
import openpi.training.optimizer as _optimizer
import openpi.training.robomme_adaptive_mse_eval as _robomme_adaptive_mse_eval
import openpi.training.sharding as sharding
import openpi.training.utils as training_utils
import openpi.training.weight_loaders as _weight_loaders

_MEMORY_ADAPTER_MODULES = frozenset(
    {
        "memory_projector_in",
        "memory_projector_out",
        "memory_projector_norm",
        "pre_attention_norm_2",
        "pre_ffw_norm_2",
        "final_norm_2",
    }
)

_MEMORY_PROJECTOR_LR_MODULES = frozenset(
    {
        "memory_projector_in",
        "memory_projector_out",
        "memory_projector_norm",
        "memory_segment_embed",
    }
)


def _path_component(entry: Any) -> str:
    """Return a stable string for JAX DictKey/GetAttrKey path entries."""
    return str(getattr(entry, "key", getattr(entry, "name", entry)))


def _is_memory_adapter_path(path: tuple[Any, ...]) -> bool:
    return any(_path_component(entry) in _MEMORY_ADAPTER_MODULES for entry in path)


def _memory_projector_lr_labels(params: nnx.State) -> nnx.State:
    """Label only the newly initialized memory projection/segment parameters."""
    flat_labels = {
        path: "memory_projector"
        if any(_path_component(entry) in _MEMORY_PROJECTOR_LR_MODULES for entry in path)
        else "base"
        for path in params.flat_state()
    }
    projector_paths = [
        path for path, label in flat_labels.items() if label == "memory_projector"
    ]
    if not projector_paths:
        raise ValueError("memory_projector_lr_multiplier matched no trainable parameters")
    return nnx.State.from_flat_path(flat_labels)


def _memory_adapter_filter() -> nnx.filterlib.Filter:
    modules = "|".join(sorted(_MEMORY_ADAPTER_MODULES))
    return nnx.All(nnx.Param, nnx_utils.PathRegex(rf".*(?:{modules}).*"))


def _warmup_trainable_filter(config: _config.TrainConfig) -> nnx.filterlib.Filter:
    """Return the static parameter subset updated during staged warmup."""
    if config.warmup_freeze_filter is None:
        return _memory_adapter_filter()
    return nnx.All(nnx.Param, nnx.Not(config.warmup_freeze_filter))


@dataclasses.dataclass(frozen=True)
class _StagedOptimizer:
    """Two static optimizer paths without materializing full-tree dynamic masks."""

    full_tx: optax.GradientTransformation
    warmup_tx: optax.GradientTransformation
    warmup_filter: nnx.filterlib.Filter

    def init(self, params: nnx.State) -> tuple[optax.OptState, optax.OptState]:
        return self.full_tx.init(params), self.warmup_tx.init(params.filter(self.warmup_filter))

    def update(
        self, updates: nnx.State, state: tuple[optax.OptState, optax.OptState], params: nnx.State
    ) -> tuple[nnx.State, tuple[optax.OptState, optax.OptState]]:
        updates, full_state = self.full_tx.update(updates, state[0], params)
        return updates, (full_state, state[1])

    def update_warmup(
        self, updates: nnx.State, state: tuple[optax.OptState, optax.OptState], params: nnx.State
    ) -> tuple[nnx.State, tuple[optax.OptState, optax.OptState]]:
        updates, warmup_state = self.warmup_tx.update(updates, state[1], params)
        return updates, (state[0], warmup_state)


def _create_train_optimizer(
    config: _config.TrainConfig, trainable_params: nnx.State
) -> optax.GradientTransformation | _StagedOptimizer:
    if config.memory_adapter_warmup_steps == 0:
        parameter_labels = (
            _memory_projector_lr_labels(trainable_params)
            if config.memory_projector_lr_multiplier != 1.0
            else None
        )
        return _optimizer.create_optimizer(
            config.optimizer,
            config.lr_schedule,
            weight_decay_mask=None,
            memory_projector_lr_multiplier=config.memory_projector_lr_multiplier,
            parameter_labels=parameter_labels,
        )

    # The full optimizer is idle during adapter warmup, so offset its local
    # schedule to the global unfreeze step. Keeping the two update paths static
    # avoids the large HLO buffers produced by full-tree dynamic masks.
    global_lr = config.lr_schedule.create()
    warmup_steps = config.memory_adapter_warmup_steps
    full_tx = config.optimizer.create(lambda step: global_lr(step + warmup_steps), weight_decay_mask=None)
    warmup_lr_schedule = config.warmup_lr_schedule or config.lr_schedule
    warmup_tx = _optimizer.create_optimizer(config.optimizer, warmup_lr_schedule, weight_decay_mask=None)
    return _StagedOptimizer(
        full_tx=full_tx,
        warmup_tx=warmup_tx,
        warmup_filter=_warmup_trainable_filter(config),
    )


def validate_memory_normalization_requirement(config: _config.TrainConfig) -> None:
    """Require finalized LAM statistics before starting a memory run."""
    if not config.require_memory_latent_normalization:
        return
    if not getattr(config.model, "use_memory", False):
        raise ValueError("Required memory latent normalization is configured for a no-memory model")
    if config.model.memory_latent_mean is None or config.model.memory_latent_std is None:
        raise ValueError(
            "This training config requires dataset-specific memory latent normalization. "
            "Provide mean/std from the finalized memory manifest before launching JAX training."
        )


def _memory_advantage_alpha(config: _config.TrainConfig, step: at.Array | int) -> at.Array:
    """Return the uniform-to-memory-aware weighting interpolation factor."""
    model_config = config.model
    if not getattr(model_config, "memory_advantage_weighting", False):
        return jnp.asarray(0.0, dtype=jnp.float32)
    warmup_steps = round(config.num_train_steps * model_config.memory_advantage_warmup_fraction)
    ramp_steps = round(config.num_train_steps * model_config.memory_advantage_ramp_fraction)
    step = jnp.asarray(step, dtype=jnp.float32)
    if ramp_steps == 0:
        return jnp.asarray(step >= warmup_steps, dtype=jnp.float32)
    return jnp.clip((step - warmup_steps) / ramp_steps, 0.0, 1.0)


def _hardness_alpha(config: _config.TrainConfig, step: at.Array | int) -> at.Array:
    """Return zero before the hardness warm-up boundary and one thereafter."""
    warmup_steps = getattr(config.model, "hardness_warmup_steps", 0)
    return jnp.asarray(step >= warmup_steps, dtype=jnp.float32)


def _physical_batch_size(global_batch_size: int, device_count: int, accumulation_steps: int) -> int:
    """Return the global microbatch size used by the JAX data loader.

    ``global_batch_size`` remains the effective optimizer batch. Each device
    sees ``global_batch_size / accumulation_steps / device_count`` examples in
    one forward/backward pass.
    """
    for name, value in (
        ("global_batch_size", global_batch_size),
        ("device_count", device_count),
        ("accumulation_steps", accumulation_steps),
    ):
        if type(value) is not int or value <= 0:
            raise ValueError(f"{name} must be a positive integer, got {value!r}")
    if global_batch_size % accumulation_steps:
        raise ValueError(
            f"Global batch size {global_batch_size} must be divisible by gradient accumulation steps "
            f"{accumulation_steps}."
        )
    physical_batch_size = global_batch_size // accumulation_steps
    if physical_batch_size % device_count:
        raise ValueError(
            f"Physical global microbatch size {physical_batch_size} must be divisible by the number of devices "
            f"{device_count}; effective global batch size={global_batch_size}, "
            f"gradient accumulation steps={accumulation_steps}."
        )
    return physical_batch_size


def _scale_gradients(grads, accumulation_steps: int):
    """Scale one microbatch gradient for an equal-sized gradient mean."""
    scale = jnp.asarray(1.0 / accumulation_steps, dtype=jnp.float32)
    return jax.tree.map(lambda grad: grad.astype(jnp.float32) * scale, grads)


def _sum_gradients(accumulated_grads, grads):
    """Sum equal-structure sharded gradient trees."""
    return jax.tree.map(lambda accumulated, grad: accumulated + grad, accumulated_grads, grads)


def init_logging():
    """Custom logging format for better readability."""
    level_mapping = {"DEBUG": "D", "INFO": "I", "WARNING": "W", "ERROR": "E", "CRITICAL": "C"}

    class CustomFormatter(logging.Formatter):
        def format(self, record):
            record.levelname = level_mapping.get(record.levelname, record.levelname)
            return super().format(record)

    formatter = CustomFormatter(
        fmt="%(asctime)s.%(msecs)03d [%(levelname)s] %(message)-80s (%(process)d:%(filename)s:%(lineno)s)",
        datefmt="%H:%M:%S",
    )

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.handlers[0].setFormatter(formatter)


def init_wandb(config: _config.TrainConfig, *, resuming: bool, log_code: bool = False, enabled: bool = True):
    if not enabled:
        wandb.init(mode="disabled")
        return

    ckpt_dir = config.checkpoint_dir
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory {ckpt_dir} does not exist.")
    new_run_on_resume = os.environ.get("WANDB_NEW_RUN_ON_RESUME", "").strip().lower()
    if new_run_on_resume not in ("", "0", "1", "false", "true"):
        raise ValueError("WANDB_NEW_RUN_ON_RESUME must be 0/1/false/true")
    if resuming and new_run_on_resume in ("1", "true"):
        wandb.init(
            name=os.environ.get("WANDB_RUN_NAME", f"{config.exp_name}-resume"),
            config=dataclasses.asdict(config),
            project=config.project_name,
        )
        (ckpt_dir / "wandb_id.txt").write_text(wandb.run.id)
    elif resuming:
        run_id = (ckpt_dir / "wandb_id.txt").read_text().strip()
        wandb.init(id=run_id, resume="must", project=config.project_name)
    else:
        wandb.init(
            name=config.exp_name,
            config=dataclasses.asdict(config),
            project=config.project_name,
        )
        (ckpt_dir / "wandb_id.txt").write_text(wandb.run.id)

    if log_code:
        wandb.run.log_code(epath.Path(__file__).parent.parent)


def _load_weights_and_validate(loader: _weight_loaders.WeightLoader, params_shape: at.Params) -> at.Params:
    """Loads and validates the weights. Returns a loaded subset of the weights."""
    loaded_params = loader.load(params_shape)
    at.check_pytree_equality(expected=params_shape, got=loaded_params, check_shapes=True, check_dtypes=True)

    # Remove jax.ShapeDtypeStruct from the loaded params. This makes sure that only the loaded params are returned.
    return traverse_util.unflatten_dict(
        {k: v for k, v in traverse_util.flatten_dict(loaded_params).items() if not isinstance(v, jax.ShapeDtypeStruct)}
    )


@at.typecheck
def init_train_state(
    config: _config.TrainConfig, init_rng: at.KeyArrayLike, mesh: jax.sharding.Mesh, *, resume: bool
) -> tuple[training_utils.TrainState, Any]:
    def initialize_model(
        rng: at.KeyArrayLike, partial_params: at.Params | None = None
    ) -> tuple[nnx.State, nnx.GraphDef[_model.BaseModel]]:
        rng, model_rng = jax.random.split(rng)
        # initialize the model (and its parameters).
        model = config.model.create(model_rng)

        # Merge the partial params into the model.
        if partial_params is not None:
            graphdef, state = nnx.split(model)
            # This will produce an error if the partial params are not a subset of the state.
            state.replace_by_pure_dict(partial_params)
            model = nnx.merge(graphdef, state)

        params = nnx.state(model)
        # Convert frozen params to bfloat16.
        params = nnx_utils.state_map(params, config.freeze_filter, lambda p: p.replace(p.value.astype(jnp.bfloat16)))
        return params, nnx.graphdef(model)

    # Build the optimizer once from the abstract trainable tree. Keeping this
    # transformation outside ``init`` makes the static TrainState metadata
    # identical for eval_shape and the subsequent jitted initialization.
    params_shape, _ = jax.eval_shape(initialize_model, init_rng)
    tx = _create_train_optimizer(config, params_shape.filter(config.trainable_filter))

    def init(rng: at.KeyArrayLike, partial_params: at.Params | None = None) -> training_utils.TrainState:
        params, model_def = initialize_model(rng, partial_params)

        return training_utils.TrainState(
            step=0,
            params=params,
            model_def=model_def,
            tx=tx,
            opt_state=tx.init(params.filter(config.trainable_filter)),
            ema_decay=config.ema_decay,
            ema_params=None if config.ema_decay is None else params,
        )

    train_state_shape = jax.eval_shape(init, init_rng)
    state_sharding = sharding.fsdp_sharding(train_state_shape, mesh, log=True)

    if config.memory_adapter_warmup_steps:
        trainable_params = train_state_shape.params.filter(config.trainable_filter)
        warmup_paths = [
            "/".join(map(str, path)) for path in trainable_params.filter(_warmup_trainable_filter(config)).flat_state()
        ]
        if not warmup_paths:
            raise ValueError("memory_adapter_warmup_steps matched no warmup-trainable parameters")
        logging.info(
            "Staged memory warmup: updating %d parameter leaves for steps [0, %d); "
            "all configured parameters unfreeze at step %d. Matched roots: %s",
            len(warmup_paths),
            config.memory_adapter_warmup_steps,
            config.memory_adapter_warmup_steps,
            ", ".join(sorted({path.split("/")[0] for path in warmup_paths})),
        )

    if resume:
        return train_state_shape, state_sharding

    partial_params = _load_weights_and_validate(config.weight_loader, train_state_shape.params.to_pure_dict())
    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    # Initialize the train state and mix in the partial params.
    train_state = jax.jit(
        init,
        donate_argnums=(1,),  # donate the partial params buffer.
        in_shardings=replicated_sharding,
        out_shardings=state_sharding,
    )(init_rng, partial_params)

    return train_state, state_sharding


@at.typecheck
def _compute_gradients(
    config: _config.TrainConfig,
    rng: at.KeyArrayLike,
    state: training_utils.TrainState,
    batch: tuple[_model.Observation, _model.Actions],
    *,
    adapter_only: bool = False,
    microbatch_index: at.Array | int | None = None,
) -> tuple[nnx.State, dict[str, at.Array]]:
    model = nnx.merge(state.model_def, state.params)
    model.train()
    memory_advantage_enabled = getattr(config.model, "memory_advantage_weighting", False)
    action_motion_weighting_enabled = getattr(config.model, "action_motion_weighting", False)
    baseline_uncertainty_weighting_enabled = getattr(config.model, "baseline_uncertainty_weighting", False)
    hardness_weighting_enabled = getattr(config.model, "hardness_weighting", False)
    memory_advantage_alpha = _memory_advantage_alpha(config, state.step)
    hardness_alpha = _hardness_alpha(config, state.step)

    @at.typecheck
    def loss_fn(
        model: _model.BaseModel, rng: at.KeyArrayLike, observation: _model.Observation, actions: _model.Actions
    ):
        if (
            getattr(config.model, "memory_flow_horizon", 0)
            or memory_advantage_enabled
            or action_motion_weighting_enabled
            or baseline_uncertainty_weighting_enabled
            or hardness_weighting_enabled
            or getattr(config.model, "memory_demo_direction_generation", False)
        ):
            loss_kwargs = {}
            if memory_advantage_enabled:
                loss_kwargs["memory_advantage_alpha"] = memory_advantage_alpha
            if hardness_weighting_enabled:
                loss_kwargs["hardness_alpha"] = hardness_alpha
            chunked_loss, loss_components = model.compute_loss(
                rng,
                observation,
                actions,
                train=True,
                return_components=True,
                **loss_kwargs,
            )
        else:
            chunked_loss = model.compute_loss(rng, observation, actions, train=True)
            loss_components = {}
        return jnp.mean(chunked_loss), loss_components

    train_rng = jax.random.fold_in(rng, state.step)
    if microbatch_index is not None:
        # The optimizer step alone is identical for all microbatches. Fold in
        # the microbatch index as well so flow noise/timesteps remain distinct.
        train_rng = jax.random.fold_in(train_rng, microbatch_index)
    observation, actions = batch

    # The adapter-only and full steps are separately jitted, so each has a
    # static gradient/parameter tree and does not allocate a masked full tree.
    active_filter = _warmup_trainable_filter(config) if adapter_only else config.trainable_filter
    diff_state = nnx.DiffState(0, active_filter)
    (loss, loss_components), grads = nnx.value_and_grad(loss_fn, argnums=diff_state, has_aux=True)(
        model, train_rng, observation, actions
    )

    # Filter out params that aren't kernels.
    kernel_params = nnx.state(
        model,
        nnx.All(
            nnx.Param,
            nnx.Not(nnx_utils.PathRegex(".*/(bias|scale|pos_embedding|input_embedding)")),
            lambda _, x: x.value.ndim > 1,
        ),
    )
    info = {
        "loss": loss,
        "param_norm": optax.global_norm(kernel_params),
        **loss_components,
    }
    if config.memory_adapter_warmup_steps:
        info["base_params_unfrozen"] = state.step >= config.memory_adapter_warmup_steps
    return grads, info


@at.typecheck
def _apply_gradients(
    config: _config.TrainConfig,
    state: training_utils.TrainState,
    grads: nnx.State,
    *,
    adapter_only: bool = False,
) -> tuple[training_utils.TrainState, dict[str, at.Array]]:
    """Apply one already-averaged gradient and advance optimizer time once."""
    model = nnx.merge(state.model_def, state.params)
    active_filter = _warmup_trainable_filter(config) if adapter_only else config.trainable_filter
    lr_schedule = (config.warmup_lr_schedule or config.lr_schedule) if adapter_only else config.lr_schedule
    learning_rate = lr_schedule.create()(state.step)
    params = state.params.filter(active_filter)
    if adapter_only:
        updates, new_opt_state = state.tx.update_warmup(grads, state.opt_state, params)
    else:
        updates, new_opt_state = state.tx.update(grads, state.opt_state, params)
    new_params = optax.apply_updates(params, updates)

    # Update the model in place and return the new full state.
    nnx.update(model, new_params)
    new_params = nnx.state(model)

    new_state = dataclasses.replace(state, step=state.step + 1, params=new_params, opt_state=new_opt_state)
    if state.ema_decay is not None:
        new_state = dataclasses.replace(
            new_state,
            ema_params=jax.tree.map(
                lambda old, new: state.ema_decay * old + (1 - state.ema_decay) * new, state.ema_params, new_params
            ),
        )
    return new_state, {
        "grad_norm": optax.global_norm(grads),
        "learning_rate": learning_rate,
    }


@at.typecheck
def train_step(
    config: _config.TrainConfig,
    rng: at.KeyArrayLike,
    state: training_utils.TrainState,
    batch: tuple[_model.Observation, _model.Actions],
    *,
    adapter_only: bool = False,
) -> tuple[training_utils.TrainState, dict[str, at.Array]]:
    """Run the legacy one-microbatch optimizer step."""
    grads, info = _compute_gradients(config, rng, state, batch, adapter_only=adapter_only)
    new_state, optimizer_info = _apply_gradients(config, state, grads, adapter_only=adapter_only)
    info.update(optimizer_info)
    return new_state, info


@at.typecheck
def microbatch_gradient_step(
    config: _config.TrainConfig,
    rng: at.KeyArrayLike,
    state: training_utils.TrainState,
    batch: tuple[_model.Observation, _model.Actions],
    microbatch_index: at.Array | int,
    *,
    adapter_only: bool = False,
) -> tuple[nnx.State, dict[str, at.Array]]:
    """Compute one scaled microbatch gradient without updating optimizer state."""
    grads, info = _compute_gradients(
        config,
        rng,
        state,
        batch,
        adapter_only=adapter_only,
        microbatch_index=microbatch_index,
    )
    return _scale_gradients(grads, config.gradient_accumulation_steps), info


def main(config: _config.TrainConfig):
    init_logging()
    logging.info(f"Running on: {platform.node()}")
    validate_memory_normalization_requirement(config)

    physical_batch_size = _physical_batch_size(
        config.batch_size,
        jax.device_count(),
        config.gradient_accumulation_steps,
    )

    jax.config.update("jax_compilation_cache_dir", str(epath.Path("~/.cache/jax").expanduser()))

    rng = jax.random.key(config.seed)
    train_rng, init_rng = jax.random.split(rng)

    mesh = sharding.make_mesh(config.fsdp_devices)
    data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))
    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    checkpoint_manager, output_resuming = _checkpoints.initialize_checkpoint_dir(
        config.checkpoint_dir,
        keep_period=config.keep_period,
        overwrite=config.overwrite,
        resume=config.resume,
    )
    source_checkpoint_manager = None
    if config.restore_checkpoint_dir is not None:
        source_checkpoint_manager, source_resuming = _checkpoints.initialize_checkpoint_dir(
            config.restore_checkpoint_dir,
            keep_period=config.keep_period,
            overwrite=False,
            resume=True,
            read_only=True,
        )
        if not source_resuming:
            raise FileNotFoundError(
                "restore_checkpoint_dir does not contain a restorable checkpoint: "
                f"{config.restore_checkpoint_dir}"
            )
        available_steps = tuple(source_checkpoint_manager.all_steps())
        if config.restore_checkpoint_step not in available_steps:
            raise FileNotFoundError(
                "requested restore_checkpoint_step is unavailable: "
                f"step={config.restore_checkpoint_step}, directory={config.restore_checkpoint_dir}, "
                f"available={available_steps}"
            )
        logging.info(
            "Using immutable external restore checkpoint: directory=%s step=%d",
            config.restore_checkpoint_dir,
            config.restore_checkpoint_step,
        )
    resuming = output_resuming or source_checkpoint_manager is not None
    # An external restore begins a distinct output experiment, so it must
    # create its own W&B run instead of trying to reuse the source run ID.
    init_wandb(config, resuming=output_resuming, enabled=config.wandb_enabled)

    # Keep config.batch_size as the effective optimizer batch for W&B and the
    # training contract. The loader emits one physical microbatch at a time so
    # long-history activations do not double when effective batch size doubles.
    loader_config = dataclasses.replace(config, batch_size=physical_batch_size)
    data_loader = _data_loader.create_data_loader(
        loader_config,
        sharding=data_sharding,
        shuffle=True,
    )
    data_iter = iter(data_loader)
    batch = next(data_iter)
    logging.info(f"Initialized data loader:\n{training_utils.array_tree_to_info(batch)}")
    logging.info(
        "Effective global batch size=%d from %d global microbatches of %d (%d samples/device/microbatch)",
        config.batch_size,
        config.gradient_accumulation_steps,
        physical_batch_size,
        physical_batch_size // jax.device_count(),
    )

    # Log images from first batch to sanity check.
    images_to_log = [
        wandb.Image(np.concatenate([np.array(img[i]) for img in batch[0].images.values()], axis=1))
        for i in range(min(5, len(next(iter(batch[0].images.values())))))
    ]
    wandb.log({"camera_views": images_to_log}, step=0)

    train_state, train_state_sharding = init_train_state(config, init_rng, mesh, resume=resuming)
    jax.block_until_ready(train_state)
    logging.info(f"Initialized train state:\n{training_utils.array_tree_to_info(train_state.params)}")

    if resuming:
        restore_manager = source_checkpoint_manager if source_checkpoint_manager is not None else checkpoint_manager
        restore_step = config.restore_checkpoint_step if source_checkpoint_manager is not None else None
        train_state = _checkpoints.restore_state(
            restore_manager,
            train_state,
            data_loader,
            state_sharding=train_state_sharding,
            step=restore_step,
        )
        if source_checkpoint_manager is not None:
            # The source manager has completed its only job. Close it before
            # creating the resident evaluator or saving the isolated output
            # checkpoint, so only the output manager owns async write state.
            source_checkpoint_manager.close()
            source_checkpoint_manager = None
            logging.info("Closed read-only external restore checkpoint manager")
        # Checkpoints intentionally exclude the Python Torch DataLoader
        # iterator.  For an adaptive plan-gated sampler, discard the batch
        # constructed before restore and begin a new iterator at the completed
        # checkpoint boundary so plan weights remain aligned with optimizer
        # steps after a process restart.
        restored_step = int(train_state.step)
        if data_loader.set_adaptive_mse_block_start(restored_step):
            data_iter = iter(data_loader)
            batch = next(data_iter)
            logging.info("Reset adaptive MSE sampler to restored plan boundary step=%d", restored_step)

    adaptive_mse_evaluator = None
    if config.adaptive_mse_eval_pack_path is not None:
        if config.gradient_accumulation_steps != 1:
            raise ValueError("resident adaptive MSE evaluation currently requires gradient_accumulation_steps=1")
        adaptive_mse_evaluator = _robomme_adaptive_mse_eval.ResidentAdaptiveMseEvaluator(
            config=config,
            data_config=data_loader.data_config(),
            model_def=train_state.model_def,
            params_sharding=train_state_sharding.params,
            replicated_sharding=replicated_sharding,
            pack_path=epath.Path(config.adaptive_mse_eval_pack_path),
            output_dir=epath.Path(config.adaptive_mse_eval_result_dir),
            plan_dir=epath.Path(data_loader.data_config().diversity_sampling_mse_plan_dir),
            diversity_cache_dir=epath.Path(config.adaptive_mse_eval_diversity_cache_dir),
        )

    ptrain_step = jax.jit(
        functools.partial(train_step, config, adapter_only=False),
        in_shardings=(replicated_sharding, train_state_sharding, data_sharding),
        out_shardings=(train_state_sharding, replicated_sharding),
        donate_argnums=(1,),
    )
    ptrain_adapter_step = None
    if config.memory_adapter_warmup_steps:
        ptrain_adapter_step = jax.jit(
            functools.partial(train_step, config, adapter_only=True),
            in_shardings=(replicated_sharding, train_state_sharding, data_sharding),
            out_shardings=(train_state_sharding, replicated_sharding),
            donate_argnums=(1,),
        )

    accumulated_step_fns = {}
    if config.gradient_accumulation_steps > 1:
        for adapter_only in (False, True) if config.memory_adapter_warmup_steps else (False,):
            active_filter = _warmup_trainable_filter(config) if adapter_only else config.trainable_filter
            gradient_sharding = train_state_sharding.params.filter(active_filter)
            accumulated_step_fns[adapter_only] = (
                jax.jit(
                    functools.partial(microbatch_gradient_step, config, adapter_only=adapter_only),
                    in_shardings=(
                        replicated_sharding,
                        train_state_sharding,
                        data_sharding,
                        replicated_sharding,
                    ),
                    out_shardings=(gradient_sharding, replicated_sharding),
                ),
                jax.jit(
                    _sum_gradients,
                    in_shardings=(gradient_sharding, gradient_sharding),
                    out_shardings=gradient_sharding,
                    donate_argnums=(0, 1),
                ),
                jax.jit(
                    functools.partial(_apply_gradients, config, adapter_only=adapter_only),
                    in_shardings=(train_state_sharding, gradient_sharding),
                    out_shardings=(train_state_sharding, replicated_sharding),
                    donate_argnums=(0, 1),
                ),
            )

    start_step = int(train_state.step)
    pbar = tqdm.tqdm(
        range(start_step, config.num_train_steps),
        initial=start_step,
        total=config.num_train_steps,
        dynamic_ncols=True,
    )

    infos = []
    for step in pbar:
        adapter_only = ptrain_adapter_step is not None and step < config.memory_adapter_warmup_steps
        if config.gradient_accumulation_steps == 1:
            step_fn = ptrain_adapter_step if adapter_only else ptrain_step
            with sharding.set_mesh(mesh):
                train_state, info = step_fn(train_rng, train_state, batch)
        else:
            gradient_step, sum_gradients, apply_gradient_step = accumulated_step_fns[adapter_only]
            microbatch_infos = []
            with sharding.set_mesh(mesh):
                accumulated_grads, microbatch_info = gradient_step(
                    train_rng,
                    train_state,
                    batch,
                    jnp.asarray(0, dtype=jnp.uint32),
                )
            microbatch_infos.append(microbatch_info)
            batch = next(data_iter)
            for microbatch_index in range(1, config.gradient_accumulation_steps):
                with sharding.set_mesh(mesh):
                    microbatch_grads, microbatch_info = gradient_step(
                        train_rng,
                        train_state,
                        batch,
                        jnp.asarray(microbatch_index, dtype=jnp.uint32),
                    )
                    accumulated_grads = sum_gradients(accumulated_grads, microbatch_grads)
                microbatch_infos.append(microbatch_info)
                batch = next(data_iter)
            with sharding.set_mesh(mesh):
                train_state, optimizer_info = apply_gradient_step(train_state, accumulated_grads)
            stacked_microbatch_infos = common_utils.stack_forest(microbatch_infos)
            info = jax.tree.map(jnp.mean, stacked_microbatch_infos)
            info.update(optimizer_info)
        infos.append(info)
        if step % config.log_interval == 0:
            stacked_infos = common_utils.stack_forest(infos)
            reduced_info = jax.device_get(jax.tree.map(jnp.mean, stacked_infos))
            # Keep the round-trip representation of each scalar.  The text
            # log is also consumed by the live W&B sidecar, so formatting
            # with ``.4f`` would irreversibly quantize metrics before upload.
            info_str = ", ".join(f"{k}={np.asarray(v).item()!r}" for k, v in reduced_info.items())
            pbar.write(f"Step {step}: {info_str}")
            wandb.log(reduced_info, step=step)
            infos = []
        completed_step = step + 1
        if config.save_steps:
            should_save = completed_step in config.save_steps
            checkpoint_step = completed_step
        else:
            should_save = (step % config.save_interval == 0 and step > start_step) or step == config.num_train_steps - 1
            checkpoint_step = step

        should_evaluate_adaptive_mse = (
            adaptive_mse_evaluator is not None
            and completed_step % config.adaptive_mse_eval_interval_steps == 0
        )
        if should_save:
            _checkpoints.save_state(checkpoint_manager, train_state, data_loader, checkpoint_step)
        if should_evaluate_adaptive_mse:
            if should_save:
                # At a save/eval coincidence, evaluate the fully committed
                # state. At eval-only boundaries there is no new save to wait
                # for, so the resident evaluator runs immediately.
                checkpoint_manager.wait_until_finished()
            adaptive_mse_evaluator.evaluate_and_publish(train_state.params, completed_step)
            if data_loader.set_adaptive_mse_block_start(completed_step):
                logging.info(
                    "Advanced adaptive MSE sampler to published plan boundary step=%d",
                    completed_step,
                )
        # The plan-aware sampler exhausts this iterator at a block boundary.
        # The preceding evaluation has already published and selected the next
        # plan, so the next request recreates its Torch iterator without any
        # cross-boundary worker prefetch. The adaptive run has one microbatch.
        if config.gradient_accumulation_steps == 1 and completed_step < config.num_train_steps:
            batch = next(data_iter)

    logging.info("Waiting for checkpoint manager to finish")
    checkpoint_manager.wait_until_finished()

    # End the public generator before explicitly reaping persistent Torch
    # workers. Otherwise their destruction may race interpreter/JAX teardown
    # and turn a completed run into a late C++ abort.
    close_data_iter = getattr(data_iter, "close", None)
    if callable(close_data_iter):
        close_data_iter()
    close_data_loader = getattr(data_loader, "close", None)
    if callable(close_data_loader):
        close_data_loader()

if __name__ == "__main__":
    main(_config.cli())
