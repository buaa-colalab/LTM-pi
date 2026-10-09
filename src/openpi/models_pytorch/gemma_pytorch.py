import contextlib
from typing import Literal

import torch
from torch import nn
from torch.nn.attention import SDPBackend
from torch.nn.attention import sdpa_kernel
import torch.nn.functional as F  # noqa: N812
from transformers import GemmaModel
from transformers import PaliGemmaForConditionalGeneration
from transformers.models.auto import CONFIG_MAPPING
from transformers.models.gemma import modeling_gemma


def _sdpa_attention_forward(module, query, key, value, attention_mask, scaling):
    """Match Gemma eager attention without materializing attention weights."""
    # PyTorch 2.7's fused efficient backend requires dense Q/K/V to have the
    # same head count. repeat_kv uses an expand/reshape and is still much less
    # expensive than materializing [B, H, L, L] attention probabilities.
    key = modeling_gemma.repeat_kv(key, module.num_key_value_groups)
    value = modeling_gemma.repeat_kv(value, module.num_key_value_groups)
    if attention_mask is not None and attention_mask.dtype not in (torch.bool, query.dtype):
        attention_mask = attention_mask.to(dtype=query.dtype)
    backend = sdpa_kernel(backends=[SDPBackend.EFFICIENT_ATTENTION]) if query.is_cuda else contextlib.nullcontext()
    with backend:
        output = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=attention_mask,
            dropout_p=0.0,
            is_causal=False,
            scale=scaling,
        )
    return output.transpose(1, 2).contiguous()


class PaliGemmaWithExpertModel(nn.Module):
    def __init__(
        self,
        vlm_config,
        action_expert_config,
        use_adarms=None,
        *,
        use_memory_expert: bool = False,
        share_memory_attention: bool = False,
        precision: Literal["bfloat16", "float32"] = "bfloat16",
        mot_attention_backend: Literal["sdpa", "eager"] = "sdpa",
    ):
        if use_adarms is None:
            use_adarms = [False, False]
        super().__init__()
        if mot_attention_backend not in ("sdpa", "eager"):
            raise ValueError(f"Unsupported MoT attention backend: {mot_attention_backend}")
        self.mot_attention_backend = mot_attention_backend
        if share_memory_attention and not use_memory_expert:
            raise ValueError("share_memory_attention requires use_memory_expert=True")
        self.share_memory_attention = share_memory_attention

        vlm_config_hf = CONFIG_MAPPING["paligemma"]()
        vlm_config_hf._vocab_size = 257152  # noqa: SLF001
        vlm_config_hf.image_token_index = 257152
        vlm_config_hf.text_config.hidden_size = vlm_config.width
        vlm_config_hf.text_config.intermediate_size = vlm_config.mlp_dim
        vlm_config_hf.text_config.num_attention_heads = vlm_config.num_heads
        vlm_config_hf.text_config.head_dim = vlm_config.head_dim
        vlm_config_hf.text_config.num_hidden_layers = vlm_config.depth
        vlm_config_hf.text_config.num_key_value_heads = vlm_config.num_kv_heads
        vlm_config_hf.text_config.hidden_activation = "gelu_pytorch_tanh"
        vlm_config_hf.text_config.torch_dtype = "float32"
        vlm_config_hf.text_config.vocab_size = 257152
        vlm_config_hf.text_config.use_adarms = use_adarms[0]
        vlm_config_hf.text_config.adarms_cond_dim = vlm_config.width if use_adarms[0] else None
        vlm_config_hf.vision_config.intermediate_size = 4304
        vlm_config_hf.vision_config.projection_dim = 2048
        vlm_config_hf.vision_config.projector_hidden_act = "gelu_fast"
        vlm_config_hf.vision_config.torch_dtype = "float32"
        vlm_config_hf.vision_config._attn_implementation = "sdpa"  # noqa: SLF001

        action_expert_config_hf = CONFIG_MAPPING["gemma"](
            head_dim=action_expert_config.head_dim,
            hidden_size=action_expert_config.width,
            intermediate_size=action_expert_config.mlp_dim,
            num_attention_heads=action_expert_config.num_heads,
            num_hidden_layers=action_expert_config.depth,
            num_key_value_heads=action_expert_config.num_kv_heads,
            vocab_size=257152,
            hidden_activation="gelu_pytorch_tanh",
            torch_dtype="float32",
            use_adarms=use_adarms[1],
            adarms_cond_dim=action_expert_config.width if use_adarms[1] else None,
        )

        self.paligemma = PaliGemmaForConditionalGeneration(config=vlm_config_hf)
        self.memory_expert = None
        if use_memory_expert:
            memory_expert_config_hf = CONFIG_MAPPING["gemma"](
                head_dim=action_expert_config.head_dim,
                hidden_size=action_expert_config.width,
                intermediate_size=action_expert_config.mlp_dim,
                num_attention_heads=action_expert_config.num_heads,
                num_hidden_layers=action_expert_config.depth,
                num_key_value_heads=action_expert_config.num_kv_heads,
                vocab_size=257152,
                hidden_activation="gelu_pytorch_tanh",
                torch_dtype="float32",
                use_adarms=False,
                adarms_cond_dim=None,
            )
            self.memory_expert = nn.Module()
            self.memory_expert.model = GemmaModel(config=memory_expert_config_hf)
            self.memory_expert.model.embed_tokens = None
        self.gemma_expert = nn.Module()
        self.gemma_expert.model = GemmaModel(config=action_expert_config_hf)
        self.gemma_expert.model.embed_tokens = None
        if self.share_memory_attention:
            for memory_layer, action_layer in zip(
                self.memory_expert.model.layers,
                self.gemma_expert.model.layers,
                strict=True,
            ):
                memory_layer.self_attn = action_layer.self_attn

        self.to_bfloat16_for_selected_params(precision)

    def to_bfloat16_for_selected_params(self, precision: Literal["bfloat16", "float32"] = "bfloat16"):
        if precision == "bfloat16":
            self.to(dtype=torch.bfloat16)
        elif precision == "float32":
            self.to(dtype=torch.float32)
            return
        else:
            raise ValueError(f"Invalid precision: {precision}")

        params_to_keep_float32 = [
            "vision_tower.vision_model.embeddings.patch_embedding.weight",
            "vision_tower.vision_model.embeddings.patch_embedding.bias",
            "vision_tower.vision_model.embeddings.position_embedding.weight",
            "input_layernorm",
            "post_attention_layernorm",
            "model.norm",
        ]

        for name, param in self.named_parameters():
            if any(selector in name for selector in params_to_keep_float32):
                param.data = param.data.to(dtype=torch.float32)

    def embed_image(self, image: torch.Tensor):
        return self.paligemma.model.get_image_features(image)

    def embed_language_tokens(self, tokens: torch.Tensor):
        return self.paligemma.language_model.embed_tokens(tokens)

    def stream_models(self):
        """Return MoT streams in context order: VLM, optional memory, action."""
        models = [self.paligemma.language_model]
        if self.memory_expert is not None:
            models.append(self.memory_expert.model)
        models.append(self.gemma_expert.model)
        return models

    @torch.no_grad()
    def initialize_memory_from_action_expert(self) -> int:
        """Warm-start compatible memory attention/FFN tensors from the action expert.

        PI0.5's action expert uses adaRMSNorm while the memory stream uses regular
        RMSNorm, so norm parameters intentionally remain at their native init.
        """
        if self.memory_expert is None:
            return 0
        source = dict(self.gemma_expert.model.named_parameters())
        copied = 0
        for name, target in self.memory_expert.model.named_parameters():
            value = source.get(name)
            if value is None or value.shape != target.shape:
                continue
            if ".mlp." not in name and (self.share_memory_attention or ".self_attn." not in name):
                continue
            target.copy_(value)
            copied += 1
        return copied

    def forward(
        self,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: list[torch.FloatTensor] | None = None,
        inputs_embeds: list[torch.FloatTensor] | None = None,
        use_cache: bool | None = None,
        adarms_cond: list[torch.Tensor] | None = None,
    ):
        models = self.stream_models()
        if inputs_embeds is None or len(inputs_embeds) != len(models):
            raise ValueError(
                f"Expected {len(models)} input streams, got {0 if inputs_embeds is None else len(inputs_embeds)}"
            )
        if adarms_cond is None:
            adarms_cond = [None] * len(models)
        if len(adarms_cond) != len(models):
            raise ValueError(f"Expected {len(models)} adaRMS conditions, got {len(adarms_cond)}")

        active_indices = [i for i, value in enumerate(inputs_embeds) if value is not None]
        if not active_indices:
            raise ValueError("At least one MoT stream must be provided")

        # A single active stream can use the native HF forward path, including
        # KV caching. Joint streams use the MoT path below.
        if len(active_indices) == 1:
            stream_idx = active_indices[0]
            output = models[stream_idx].forward(
                inputs_embeds=inputs_embeds[stream_idx],
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                adarms_cond=adarms_cond[stream_idx],
            )
            outputs = [None] * len(models)
            outputs[stream_idx] = output.last_hidden_state
            return outputs, output.past_key_values if stream_idx == 0 else None

        if use_cache:
            raise ValueError("KV caching is only supported for a single active MoT stream")

        active_models = [models[i] for i in active_indices]
        active_embeds = [inputs_embeds[i] for i in active_indices]
        active_conds = [adarms_cond[i] for i in active_indices]
        num_layers = self.paligemma.config.text_config.num_hidden_layers

        if any(len(model.layers) != num_layers for model in active_models):
            raise ValueError("All MoT streams must have the same transformer depth")

        # Check if gradient checkpointing is enabled for any stream.
        use_gradient_checkpointing = self.training and any(
            getattr(model, "gradient_checkpointing", False) for model in active_models
        )

        def compute_layer_complete(layer_idx, stream_embeds, attention_mask, position_ids, stream_conds):
            query_states = []
            key_states = []
            value_states = []
            gates = []
            for model, hidden_states, cond in zip(active_models, stream_embeds, stream_conds, strict=True):
                layer = model.layers[layer_idx]
                normalized, gate = layer.input_layernorm(hidden_states, cond=cond)
                gates.append(gate)

                input_shape = normalized.shape[:-1]
                hidden_shape = (*input_shape, -1, layer.self_attn.head_dim)
                query_states.append(layer.self_attn.q_proj(normalized).view(hidden_shape).transpose(1, 2))
                key_states.append(layer.self_attn.k_proj(normalized).view(hidden_shape).transpose(1, 2))
                value_states.append(layer.self_attn.v_proj(normalized).view(hidden_shape).transpose(1, 2))

            query_states = torch.cat(query_states, dim=2)
            key_states = torch.cat(key_states, dim=2)
            value_states = torch.cat(value_states, dim=2)

            # Rotary embedding reads only device/dtype from its tensor input.
            # Reuse Q instead of allocating and zeroing [B, L, D] every layer.
            cos, sin = self.paligemma.model.language_model.rotary_emb(query_states, position_ids)
            query_states, key_states = modeling_gemma.apply_rotary_pos_emb(
                query_states, key_states, cos, sin, unsqueeze_dim=1
            )

            batch_size = query_states.shape[0]
            attention_module = active_models[0].layers[layer_idx].self_attn
            if self.mot_attention_backend == "sdpa":
                # The three MoT streams share attention but retain their own
                # projections and FFNs.  The eager path materializes the full
                # [B, H, L, L] score/probability tensors. SDPA keeps the same
                # block mask while letting PyTorch select a fused Hopper
                # backend, avoiding that quadratic intermediate.
                att_output = _sdpa_attention_forward(
                    attention_module,
                    query_states,
                    key_states,
                    value_states,
                    attention_mask,
                    attention_module.scaling,
                )
            else:
                att_output, _ = modeling_gemma.eager_attention_forward(
                    attention_module,
                    query_states,
                    key_states,
                    value_states,
                    attention_mask,
                    attention_module.scaling,
                )
            att_output = att_output.reshape(batch_size, query_states.shape[2], -1)

            outputs_embeds = []
            start_pos = 0
            for model, hidden_states, cond, input_gate in zip(
                active_models, stream_embeds, stream_conds, gates, strict=True
            ):
                layer = model.layers[layer_idx]
                end_pos = start_pos + hidden_states.shape[1]

                stream_att_output = att_output[:, start_pos:end_pos]
                if stream_att_output.dtype != layer.self_attn.o_proj.weight.dtype:
                    stream_att_output = stream_att_output.to(layer.self_attn.o_proj.weight.dtype)
                out_emb = layer.self_attn.o_proj(stream_att_output)

                out_emb = modeling_gemma._gated_residual(hidden_states, out_emb, input_gate)  # noqa: SLF001
                after_first_residual = out_emb
                out_emb, mlp_gate = layer.post_attention_layernorm(out_emb, cond=cond)
                if out_emb.dtype != layer.mlp.up_proj.weight.dtype:
                    out_emb = out_emb.to(dtype=layer.mlp.up_proj.weight.dtype)

                out_emb = layer.mlp(out_emb)
                out_emb = modeling_gemma._gated_residual(after_first_residual, out_emb, mlp_gate)  # noqa: SLF001
                outputs_embeds.append(out_emb)
                start_pos = end_pos

            return outputs_embeds

        for layer_idx in range(num_layers):
            if use_gradient_checkpointing:
                active_embeds = torch.utils.checkpoint.checkpoint(
                    compute_layer_complete,
                    layer_idx,
                    active_embeds,
                    attention_mask,
                    position_ids,
                    active_conds,
                    use_reentrant=False,
                    preserve_rng_state=False,
                )
            else:
                active_embeds = compute_layer_complete(
                    layer_idx, active_embeds, attention_mask, position_ids, active_conds
                )

        def compute_final_norms(stream_embeds, stream_conds):
            outputs_embeds = []
            for model, hidden_states, cond in zip(active_models, stream_embeds, stream_conds, strict=True):
                out_emb, _ = model.norm(hidden_states, cond=cond)
                outputs_embeds.append(out_emb)
            return outputs_embeds

        if use_gradient_checkpointing:
            active_outputs = torch.utils.checkpoint.checkpoint(
                compute_final_norms,
                active_embeds,
                active_conds,
                use_reentrant=False,
                preserve_rng_state=False,
            )
        else:
            active_outputs = compute_final_norms(active_embeds, active_conds)

        outputs = [None] * len(models)
        for stream_idx, output in zip(active_indices, active_outputs, strict=True):
            outputs[stream_idx] = output
        return outputs, None
