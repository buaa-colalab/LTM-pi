import logging
import math

import torch
from torch import Tensor
from torch import nn
import torch.nn.functional as F  # noqa: N812

import openpi.models.gemma as _gemma
from openpi.models_pytorch.gemma_pytorch import PaliGemmaWithExpertModel
import openpi.models_pytorch.preprocessing_pytorch as _preprocessing


def get_safe_dtype(target_dtype, device_type):
    """Get a safe dtype for the given device type."""
    if device_type == "cpu":
        # CPU doesn't support bfloat16, use float32 instead
        if target_dtype == torch.bfloat16:
            return torch.float32
        if target_dtype == torch.float64:
            return torch.float64
    return target_dtype


def create_sinusoidal_pos_embedding(
    time: torch.tensor, dimension: int, min_period: float, max_period: float, device="cpu"
) -> Tensor:
    """Computes sine-cosine positional embedding vectors for scalar positions."""
    if dimension % 2 != 0:
        raise ValueError(f"dimension ({dimension}) must be divisible by 2")

    if time.ndim != 1:
        raise ValueError("The time tensor is expected to be of shape `(batch_size, )`.")

    dtype = get_safe_dtype(torch.float64, device.type)
    fraction = torch.linspace(0.0, 1.0, dimension // 2, dtype=dtype, device=device)
    period = min_period * (max_period / min_period) ** fraction

    # Compute the outer product
    scaling_factor = 1.0 / period * 2 * math.pi
    sin_input = scaling_factor[None, :] * time[:, None]
    return torch.cat([torch.sin(sin_input), torch.cos(sin_input)], dim=1)


def sample_beta(alpha, beta, bsize, device):
    alpha_t = torch.as_tensor(alpha, dtype=torch.float32, device=device)
    beta_t = torch.as_tensor(beta, dtype=torch.float32, device=device)
    dist = torch.distributions.Beta(alpha_t, beta_t)
    return dist.sample((bsize,))


def make_att_2d_masks(pad_masks, att_masks):
    """Copied from big_vision.

    Tokens can attend to valid inputs tokens which have a cumulative mask_ar
    smaller or equal to theirs. This way `mask_ar` int[B, N] can be used to
    setup several types of attention, for example:

      [[1 1 1 1 1 1]]: pure causal attention.

      [[0 0 0 1 1 1]]: prefix-lm attention. The first 3 tokens can attend between
          themselves and the last 3 tokens have a causal attention. The first
          entry could also be a 1 without changing behaviour.

      [[1 0 1 0 1 0 0 1 0 0]]: causal attention between 4 blocks. Tokens of a
          block can attend all previous blocks and all tokens on the same block.

    Args:
      input_mask: bool[B, N] true if its part of the input, false if padding.
      mask_ar: int32[B, N] mask that's 1 where previous tokens cannot depend on
        it and 0 where it shares the same attention mask as the previous token.
    """
    if att_masks.ndim != 2:
        raise ValueError(att_masks.ndim)
    if pad_masks.ndim != 2:
        raise ValueError(pad_masks.ndim)

    cumsum = torch.cumsum(att_masks, dim=1)
    att_2d_masks = cumsum[:, None, :] <= cumsum[:, :, None]
    pad_2d_masks = pad_masks[:, None, :] * pad_masks[:, :, None]
    return att_2d_masks & pad_2d_masks


class PI0Pytorch(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.pi05 = config.pi05
        self.use_memory = getattr(config, "use_memory", False)

        paligemma_config = _gemma.get_config(config.paligemma_variant)
        action_expert_config = _gemma.get_config(config.action_expert_variant)

        self.paligemma_with_expert = PaliGemmaWithExpertModel(
            paligemma_config,
            action_expert_config,
            use_adarms=[False, True] if self.pi05 else [False, False],
            use_memory_expert=self.use_memory,
            share_memory_attention=config.share_memory_attention,
            precision=config.dtype,
            mot_attention_backend=config.mot_attention_backend,
        )

        if self.use_memory:
            if config.memory_latent_mean is None:
                memory_latent_mean = torch.zeros(config.memory_latent_dim, dtype=torch.float32)
                memory_latent_std = torch.ones(config.memory_latent_dim, dtype=torch.float32)
            else:
                memory_latent_mean = torch.tensor(config.memory_latent_mean, dtype=torch.float32)
                memory_latent_std = torch.tensor(config.memory_latent_std, dtype=torch.float32)
            self.register_buffer("memory_latent_mean", memory_latent_mean, persistent=True)
            self.register_buffer("memory_latent_std", memory_latent_std, persistent=True)
            self.memory_projector = nn.Sequential(
                nn.Linear(
                    config.memory_latent_dim * config.memory_frames_per_token,
                    config.memory_projector_hidden_dim,
                ),
                nn.GELU(),
                nn.Linear(config.memory_projector_hidden_dim, action_expert_config.width),
                nn.LayerNorm(action_expert_config.width),
            )
            if config.memory_segment_embedding:
                self.memory_segment_embed = nn.Embedding(4, action_expert_config.width)

        self.action_in_proj = nn.Linear(config.action_dim, action_expert_config.width)
        self.action_out_proj = nn.Linear(action_expert_config.width, config.action_dim)

        if self.pi05:
            self.time_mlp_in = nn.Linear(action_expert_config.width, action_expert_config.width)
            self.time_mlp_out = nn.Linear(action_expert_config.width, action_expert_config.width)
        else:
            self.state_proj = nn.Linear(config.action_dim, action_expert_config.width)
            self.action_time_mlp_in = nn.Linear(2 * action_expert_config.width, action_expert_config.width)
            self.action_time_mlp_out = nn.Linear(action_expert_config.width, action_expert_config.width)

        torch.set_float32_matmul_precision("high")
        if config.pytorch_compile_mode is not None:
            self.sample_actions = torch.compile(self.sample_actions, mode=config.pytorch_compile_mode)

        # Initialize gradient checkpointing flag
        self.gradient_checkpointing_enabled = False

        msg = "transformers_replace is not installed correctly. Please install it with `uv pip install transformers==4.53.2` and `cp -r ./src/openpi/models_pytorch/transformers_replace/* .venv/lib/python3.11/site-packages/transformers/`."
        try:
            from transformers.models.siglip import check

            if not check.check_whether_transformers_replace_is_installed_correctly():
                raise ValueError(msg)
        except ImportError:
            raise ValueError(msg) from None

    def gradient_checkpointing_enable(self):
        """Enable gradient checkpointing for memory optimization."""
        self.gradient_checkpointing_enabled = True
        self.paligemma_with_expert.paligemma.language_model.gradient_checkpointing = True
        self.paligemma_with_expert.paligemma.vision_tower.gradient_checkpointing = True
        self.paligemma_with_expert.gemma_expert.model.gradient_checkpointing = True
        if self.paligemma_with_expert.memory_expert is not None:
            self.paligemma_with_expert.memory_expert.model.gradient_checkpointing = True

        logging.info("Enabled gradient checkpointing for PI0Pytorch model")

    def gradient_checkpointing_disable(self):
        """Disable gradient checkpointing."""
        self.gradient_checkpointing_enabled = False
        self.paligemma_with_expert.paligemma.language_model.gradient_checkpointing = False
        self.paligemma_with_expert.paligemma.vision_tower.gradient_checkpointing = False
        self.paligemma_with_expert.gemma_expert.model.gradient_checkpointing = False
        if self.paligemma_with_expert.memory_expert is not None:
            self.paligemma_with_expert.memory_expert.model.gradient_checkpointing = False

        logging.info("Disabled gradient checkpointing for PI0Pytorch model")

    def is_gradient_checkpointing_enabled(self):
        """Check if gradient checkpointing is enabled."""
        return self.gradient_checkpointing_enabled

    def _apply_checkpoint(self, func, *args, **kwargs):
        """Helper method to apply gradient checkpointing if enabled."""
        if self.gradient_checkpointing_enabled and self.training:
            return torch.utils.checkpoint.checkpoint(
                func, *args, use_reentrant=False, preserve_rng_state=False, **kwargs
            )
        return func(*args, **kwargs)

    def _prepare_attention_masks_4d(self, att_2d_masks):
        """Helper method to prepare 4D attention masks for transformer."""
        att_2d_masks_4d = att_2d_masks[:, None, :, :]
        return torch.where(att_2d_masks_4d, 0.0, -2.3819763e38)

    def _prepare_mot_attention_mask(self, att_2d_masks):
        """Keep an allowed-token boolean mask for fused joint SDPA."""
        if self.config.mot_attention_backend == "sdpa":
            return att_2d_masks[:, None, :, :]
        return self._prepare_attention_masks_4d(att_2d_masks)

    def _preprocess_observation(self, observation, *, train=True):
        """Helper method to preprocess observation."""
        observation = _preprocessing.preprocess_observation_pytorch(
            observation,
            train=train,
            image_keys=self.config.active_image_keys,
        )
        return (
            list(observation.images.values()),
            list(observation.image_masks.values()),
            observation.tokenized_prompt,
            observation.tokenized_prompt_mask,
            observation.state,
            observation.memory_latents,
            observation.memory_mask,
            observation.memory_segment_ids,
            observation.memory_demo_start_image,
            observation.memory_demo_start_mask,
        )

    def sample_noise(self, shape, device):
        return torch.normal(
            mean=0.0,
            std=1.0,
            size=shape,
            dtype=torch.float32,
            device=device,
        )

    def sample_time(self, bsize, device):
        time_beta = sample_beta(1.5, 1.0, bsize, device)
        time = time_beta * 0.999 + 0.001
        return time.to(dtype=torch.float32, device=device)

    def embed_prefix(
        self, images, img_masks, lang_tokens, lang_masks, memory_demo_start_image=None, memory_demo_start_mask=None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Embed images with SigLIP and language tokens with embedding layer to prepare
        for PaliGemma transformer processing.
        """
        embs = []
        pad_masks = []
        att_masks = []

        # Process images
        for img, img_mask in zip(images, img_masks, strict=True):

            def image_embed_func(img):
                return self.paligemma_with_expert.embed_image(img)

            img_emb = self._apply_checkpoint(image_embed_func, img)

            bsize, num_img_embs = img_emb.shape[:2]

            embs.append(img_emb)
            pad_masks.append(img_mask[:, None].expand(bsize, num_img_embs))

            # Create attention masks so that image tokens attend to each other
            att_masks += [0] * num_img_embs

        # Process language tokens
        def lang_embed_func(lang_tokens):
            lang_emb = self.paligemma_with_expert.embed_language_tokens(lang_tokens)
            lang_emb_dim = lang_emb.shape[-1]
            return lang_emb * math.sqrt(lang_emb_dim)

        lang_emb = self._apply_checkpoint(lang_embed_func, lang_tokens)

        embs.append(lang_emb)
        pad_masks.append(lang_masks)

        # full attention between image and language inputs
        num_lang_embs = lang_emb.shape[1]
        att_masks += [0] * num_lang_embs

        if getattr(self.config, "memory_demo_anchor", False):
            if memory_demo_start_image is None or memory_demo_start_mask is None:
                raise ValueError(
                    "memory_demo_anchor requires memory_demo_start_image and memory_demo_start_mask"
                )
            anchor_emb = self._apply_checkpoint(
                self.paligemma_with_expert.embed_image, memory_demo_start_image
            )
            embs.append(anchor_emb)
            pad_masks.append(memory_demo_start_mask[:, None].expand(anchor_emb.shape[:2]))
            att_masks += [0] * anchor_emb.shape[1]

        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        att_masks = torch.tensor(att_masks, dtype=torch.bool, device=pad_masks.device)

        # Get batch size from the first dimension of the concatenated tensors
        bsize = pad_masks.shape[0]
        att_masks = att_masks[None, :].expand(bsize, len(att_masks))

        return embs, pad_masks, att_masks

    def embed_memory(self, memory_latents, memory_mask, memory_segment_ids=None, *, batch_size, device):
        """Project LAM z_mu values into the independent memory MoT stream."""
        if not self.use_memory:
            raise RuntimeError("embed_memory called while memory is disabled")

        if memory_latents is None:
            raise ValueError(
                "Memory-enabled PI0 requires memory_latents and memory_mask. "
                "Pass an explicit all-false mask for a no-memory ablation."
            )
        actual_shape = tuple(memory_latents.shape)
        if (
            memory_latents.ndim != 3
            or actual_shape[0] != batch_size
            or actual_shape[2] != self.config.memory_latent_dim
        ):
            raise ValueError(
                "Expected memory_latents shape "
                f"[B, T, {self.config.memory_latent_dim}] with B={batch_size}, got {actual_shape}"
            )
        memory_length = actual_shape[1]
        if memory_length < 1 or memory_length > self.config.memory_horizon:
            raise ValueError(f"Memory length must be in [1, {self.config.memory_horizon}], got {memory_length}")
        expected_mask_shape = (batch_size, memory_length)
        if memory_mask is None or tuple(memory_mask.shape) != expected_mask_shape:
            actual = None if memory_mask is None else tuple(memory_mask.shape)
            raise ValueError(f"Expected memory_mask shape {expected_mask_shape}, got {actual}")
        use_segments = bool(getattr(self.config, "memory_segment_embedding", False))
        if use_segments:
            if memory_segment_ids is None or tuple(memory_segment_ids.shape) != expected_mask_shape:
                actual = None if memory_segment_ids is None else tuple(memory_segment_ids.shape)
                raise ValueError(f"Expected memory_segment_ids shape {expected_mask_shape}, got {actual}")
            memory_segment_ids = memory_segment_ids.to(device=device, dtype=torch.long)
            if torch.any((memory_segment_ids < 0) | (memory_segment_ids > 3)):
                raise ValueError("memory_segment_ids must be in [0, 3]")

        memory_mask = memory_mask.to(device=device, dtype=torch.bool)
        memory_latents = memory_latents.to(device=device, dtype=torch.float32)
        projector_dtype = self.memory_projector[0].weight.dtype
        normalized_latents = (memory_latents - self.memory_latent_mean) / self.memory_latent_std
        normalized_latents = torch.where(
            memory_mask[..., None], normalized_latents, torch.zeros((), device=device, dtype=torch.float32)
        )
        frames_per_token = self.config.memory_frames_per_token
        left_padding = (-memory_length) % frames_per_token
        if left_padding:
            normalized_latents = F.pad(normalized_latents, (0, 0, left_padding, 0), value=0.0)
            memory_mask = F.pad(memory_mask, (left_padding, 0), value=False)
        padded_length = normalized_latents.shape[1]
        token_count = padded_length // frames_per_token
        grouped_latents = normalized_latents.reshape(
            batch_size, token_count, frames_per_token * self.config.memory_latent_dim
        )
        grouped_mask = memory_mask.reshape(batch_size, token_count, frames_per_token).any(dim=2)
        if use_segments:
            projected = self._apply_checkpoint(self.memory_projector[:-1], grouped_latents.to(projector_dtype))
            memory_embs = self.memory_projector[-1](projected + self.memory_segment_embed(memory_segment_ids))
        else:
            memory_embs = self._apply_checkpoint(self.memory_projector, grouped_latents.to(projector_dtype))
        memory_embs = torch.where(grouped_mask[..., None], memory_embs, torch.zeros_like(memory_embs))

        # Start a new full-attention block. Prefix cannot read memory, while the
        # following action block can read every valid memory token.
        memory_att_masks = torch.zeros(batch_size, token_count, dtype=torch.bool, device=device)
        memory_att_masks[:, 0] = True
        return memory_embs, grouped_mask, memory_att_masks

    def embed_suffix(self, state, noisy_actions, timestep):
        """Embed state, noisy_actions, timestep to prepare for Expert Gemma processing."""
        embs = []
        pad_masks = []
        att_masks = []

        if not self.pi05:
            if self.state_proj.weight.dtype == torch.float32:
                state = state.to(torch.float32)

            # Embed state
            def state_proj_func(state):
                return self.state_proj(state)

            state_emb = self._apply_checkpoint(state_proj_func, state)

            embs.append(state_emb[:, None, :])
            bsize = state_emb.shape[0]
            device = state_emb.device

            state_mask = torch.ones(bsize, 1, dtype=torch.bool, device=device)
            pad_masks.append(state_mask)

            # Set attention masks so that image and language inputs do not attend to state or actions
            att_masks += [1]

        # Embed timestep using sine-cosine positional encoding with sensitivity in the range [0, 1]
        time_emb = create_sinusoidal_pos_embedding(
            timestep, self.action_in_proj.out_features, min_period=4e-3, max_period=4.0, device=timestep.device
        )
        time_emb = time_emb.type(dtype=timestep.dtype)

        # Fuse timestep + action information using an MLP
        def action_proj_func(noisy_actions):
            return self.action_in_proj(noisy_actions)

        action_emb = self._apply_checkpoint(action_proj_func, noisy_actions)

        if not self.pi05:
            time_emb = time_emb[:, None, :].expand_as(action_emb)
            action_time_emb = torch.cat([action_emb, time_emb], dim=2)

            # Apply MLP layers
            def mlp_func(action_time_emb):
                x = self.action_time_mlp_in(action_time_emb)
                x = F.silu(x)  # swish == silu
                return self.action_time_mlp_out(x)

            action_time_emb = self._apply_checkpoint(mlp_func, action_time_emb)
            adarms_cond = None
        else:
            # time MLP (for adaRMS)
            def time_mlp_func(time_emb):
                x = self.time_mlp_in(time_emb)
                x = F.silu(x)  # swish == silu
                x = self.time_mlp_out(x)
                return F.silu(x)

            time_emb = self._apply_checkpoint(time_mlp_func, time_emb)
            action_time_emb = action_emb
            adarms_cond = time_emb

        # Add to input tokens
        embs.append(action_time_emb)

        bsize, action_time_dim = action_time_emb.shape[:2]
        action_time_mask = torch.ones(bsize, action_time_dim, dtype=torch.bool, device=timestep.device)
        pad_masks.append(action_time_mask)

        # Set attention masks so that image, language and state inputs do not attend to action tokens
        att_masks += [1] + ([0] * (self.config.action_horizon - 1))

        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        att_masks = torch.tensor(att_masks, dtype=embs.dtype, device=embs.device)
        att_masks = att_masks[None, :].expand(bsize, len(att_masks))

        return embs, pad_masks, att_masks, adarms_cond

    def forward(self, observation, actions, noise=None, time=None) -> Tensor:
        """Do a full training forward pass and compute the loss (batch_size x num_steps x num_motors)"""
        (
            images,
            img_masks,
            lang_tokens,
            lang_masks,
            state,
            memory_latents,
            memory_mask,
            memory_segment_ids,
            memory_demo_start_image,
            memory_demo_start_mask,
        ) = self._preprocess_observation(observation, train=True)

        if noise is None:
            noise = self.sample_noise(actions.shape, actions.device)

        if time is None:
            time = self.sample_time(actions.shape[0], actions.device)

        time_expanded = time[:, None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions

        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
            images,
            img_masks,
            lang_tokens,
            lang_masks,
            memory_demo_start_image,
            memory_demo_start_mask,
        )
        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = self.embed_suffix(state, x_t, time)
        if self.use_memory:
            memory_embs, memory_pad_masks, memory_att_masks = self.embed_memory(
                memory_latents,
                memory_mask,
                memory_segment_ids,
                batch_size=actions.shape[0],
                device=actions.device,
            )
        if (
            self.paligemma_with_expert.paligemma.language_model.layers[0].self_attn.q_proj.weight.dtype
            == torch.bfloat16
        ):
            suffix_embs = suffix_embs.to(dtype=torch.bfloat16)
            prefix_embs = prefix_embs.to(dtype=torch.bfloat16)
            if self.use_memory:
                memory_embs = memory_embs.to(dtype=torch.bfloat16)

        if self.use_memory:
            pad_masks = torch.cat([prefix_pad_masks, memory_pad_masks, suffix_pad_masks], dim=1)
            att_masks = torch.cat([prefix_att_masks, memory_att_masks, suffix_att_masks], dim=1)
            stream_embs = [prefix_embs, memory_embs, suffix_embs]
            stream_conds = [None, None, adarms_cond]
        else:
            pad_masks = torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1)
            att_masks = torch.cat([prefix_att_masks, suffix_att_masks], dim=1)
            stream_embs = [prefix_embs, suffix_embs]
            stream_conds = [None, adarms_cond]

        att_2d_masks = make_att_2d_masks(pad_masks, att_masks)
        position_ids = torch.cumsum(pad_masks, dim=1) - 1

        # Prepare attention masks
        att_2d_masks_4d = self._prepare_mot_attention_mask(att_2d_masks)

        # Apply gradient checkpointing if enabled
        def forward_func(stream_embs, att_2d_masks_4d, position_ids, stream_conds):
            outputs, _ = self.paligemma_with_expert.forward(
                attention_mask=att_2d_masks_4d,
                position_ids=position_ids,
                past_key_values=None,
                inputs_embeds=stream_embs,
                use_cache=False,
                adarms_cond=stream_conds,
            )
            return outputs[-1]

        # PaliGemmaWithExpertModel already checkpoints every transformer layer.
        # Wrapping the whole stack again makes backward recompute the complete
        # transformer in addition to the per-layer recomputations.
        suffix_out = forward_func(stream_embs, att_2d_masks_4d, position_ids, stream_conds)

        suffix_out = suffix_out[:, -self.config.action_horizon :]
        suffix_out = suffix_out.to(dtype=torch.float32)

        # Apply gradient checkpointing to final action projection if enabled
        def action_out_proj_func(suffix_out):
            return self.action_out_proj(suffix_out)

        v_t = self._apply_checkpoint(action_out_proj_func, suffix_out)

        return F.mse_loss(u_t, v_t, reduction="none")

    @torch.no_grad()
    def sample_actions(self, device, observation, noise=None, num_steps=10) -> Tensor:
        """Do a full inference forward and compute the action (batch_size x num_steps x num_motors)"""
        bsize = observation.state.shape[0]
        if noise is None:
            actions_shape = (bsize, self.config.action_horizon, self.config.action_dim)
            noise = self.sample_noise(actions_shape, device)

        (
            images,
            img_masks,
            lang_tokens,
            lang_masks,
            state,
            memory_latents,
            memory_mask,
            memory_segment_ids,
            memory_demo_start_image,
            memory_demo_start_mask,
        ) = self._preprocess_observation(observation, train=False)

        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
            images,
            img_masks,
            lang_tokens,
            lang_masks,
            memory_demo_start_image,
            memory_demo_start_mask,
        )

        if self.use_memory:
            memory_embs, memory_pad_masks, memory_att_masks = self.embed_memory(
                memory_latents,
                memory_mask,
                memory_segment_ids,
                batch_size=bsize,
                device=device,
            )
            dt = torch.tensor(-1.0 / num_steps, dtype=torch.float32, device=device)
            x_t = noise
            time = torch.tensor(1.0, dtype=torch.float32, device=device)
            while time >= -dt / 2:
                x_t = x_t + dt * self.denoise_step_with_memory(
                    state,
                    prefix_embs,
                    prefix_pad_masks,
                    prefix_att_masks,
                    memory_embs,
                    memory_pad_masks,
                    memory_att_masks,
                    x_t,
                    time.expand(bsize),
                )
                time += dt
            return x_t

        prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1

        # Compute image and language key value cache
        prefix_att_2d_masks_4d = self._prepare_attention_masks_4d(prefix_att_2d_masks)
        self.paligemma_with_expert.paligemma.language_model.config._attn_implementation = "eager"  # noqa: SLF001

        _, past_key_values = self.paligemma_with_expert.forward(
            attention_mask=prefix_att_2d_masks_4d,
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=True,
        )

        dt = -1.0 / num_steps
        dt = torch.tensor(dt, dtype=torch.float32, device=device)

        x_t = noise
        time = torch.tensor(1.0, dtype=torch.float32, device=device)
        while time >= -dt / 2:
            expanded_time = time.expand(bsize)
            v_t = self.denoise_step(
                state,
                prefix_pad_masks,
                past_key_values,
                x_t,
                expanded_time,
            )

            # Euler step - use new tensor assignment instead of in-place operation
            x_t = x_t + dt * v_t
            time += dt
        return x_t

    def denoise_step_with_memory(
        self,
        state,
        prefix_embs,
        prefix_pad_masks,
        prefix_att_masks,
        memory_embs,
        memory_pad_masks,
        memory_att_masks,
        x_t,
        timestep,
    ):
        """Correct three-stream denoising path (recomputes joint context each step)."""
        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = self.embed_suffix(state, x_t, timestep)

        target_dtype = self.paligemma_with_expert.paligemma.language_model.layers[0].self_attn.q_proj.weight.dtype
        prefix_embs = prefix_embs.to(target_dtype)
        memory_embs = memory_embs.to(target_dtype)
        suffix_embs = suffix_embs.to(target_dtype)

        pad_masks = torch.cat([prefix_pad_masks, memory_pad_masks, suffix_pad_masks], dim=1)
        att_masks = torch.cat([prefix_att_masks, memory_att_masks, suffix_att_masks], dim=1)
        attention_mask = self._prepare_mot_attention_mask(make_att_2d_masks(pad_masks, att_masks))
        position_ids = torch.cumsum(pad_masks, dim=1) - 1

        outputs, _ = self.paligemma_with_expert.forward(
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, memory_embs, suffix_embs],
            use_cache=False,
            adarms_cond=[None, None, adarms_cond],
        )
        suffix_out = outputs[-1][:, -self.config.action_horizon :].to(dtype=torch.float32)
        return self.action_out_proj(suffix_out)

    def denoise_step(
        self,
        state,
        prefix_pad_masks,
        past_key_values,
        x_t,
        timestep,
    ):
        """Apply one denoising step of the noise `x_t` at a given timestep."""
        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = self.embed_suffix(state, x_t, timestep)

        suffix_len = suffix_pad_masks.shape[1]
        batch_size = prefix_pad_masks.shape[0]
        prefix_len = prefix_pad_masks.shape[1]

        prefix_pad_2d_masks = prefix_pad_masks[:, None, :].expand(batch_size, suffix_len, prefix_len)

        suffix_att_2d_masks = make_att_2d_masks(suffix_pad_masks, suffix_att_masks)

        full_att_2d_masks = torch.cat([prefix_pad_2d_masks, suffix_att_2d_masks], dim=2)

        prefix_offsets = torch.sum(prefix_pad_masks, dim=-1)[:, None]
        position_ids = prefix_offsets + torch.cumsum(suffix_pad_masks, dim=1) - 1

        # Prepare attention masks
        full_att_2d_masks_4d = self._prepare_attention_masks_4d(full_att_2d_masks)
        self.paligemma_with_expert.gemma_expert.model.config._attn_implementation = "eager"  # noqa: SLF001

        outputs_embeds, _ = self.paligemma_with_expert.forward(
            attention_mask=full_att_2d_masks_4d,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=[None, suffix_embs],
            use_cache=False,
            adarms_cond=[None, adarms_cond],
        )

        suffix_out = outputs_embeds[1]
        suffix_out = suffix_out[:, -self.config.action_horizon :]
        suffix_out = suffix_out.to(dtype=torch.float32)
        return self.action_out_proj(suffix_out)
