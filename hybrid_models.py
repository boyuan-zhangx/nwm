"""Long-term-memory extension for Navigation World Models.

The memory lifecycle intentionally lives outside the diffusion model. A diffusion
sampler calls the denoiser hundreds of times for one output frame; mutating a
buffer inside ``forward`` would therefore store denoising steps instead of video
frames. Callers retrieve memory once per predicted frame and pass the selected
VAE latents to :class:`HybridCDiT` through ``memory_latents``.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Optional, Sequence

import torch
import torch.nn as nn
from timm.models.vision_transformer import Attention, Mlp, PatchEmbed

from models import ActionEmbedder, FinalLayer, TimestepEmbedder, modulate


@dataclass(frozen=True)
class RetrievalResult:
    """Frames and diagnostics returned by a memory query."""

    frames: torch.Tensor
    frame_indices: torch.Tensor
    scores: torch.Tensor


class MemoryBuffer:
    """Trajectory-local FIFO memory with pose/action-aware retrieval.

    Frames are stored as VAE latents on CPU. The buffer is deliberately not an
    ``nn.Module`` and is never checkpointed as model state. Create one buffer per
    rollout (or per sample) to avoid information leaking between trajectories.
    """

    def __init__(
        self,
        max_size: int = 100,
        *,
        spatial_weight: float = 0.4,
        action_weight: float = 0.35,
        storage_weight: float = 0.2,
        usage_weight: float = 0.05,
        spatial_scale: float = 10.0,
    ) -> None:
        if max_size < 1:
            raise ValueError("max_size must be positive")
        weights = spatial_weight + action_weight + storage_weight + usage_weight
        if not math.isclose(weights, 1.0, abs_tol=1e-6):
            raise ValueError(f"retrieval weights must sum to 1.0, got {weights}")
        if spatial_scale <= 0:
            raise ValueError("spatial_scale must be positive")

        self.max_size = max_size
        self.spatial_weight = spatial_weight
        self.action_weight = action_weight
        self.storage_weight = storage_weight
        self.usage_weight = usage_weight
        self.spatial_scale = spatial_scale
        self.frames: list[torch.Tensor] = []
        self.poses: list[torch.Tensor] = []
        self.actions: list[torch.Tensor] = []
        self.frame_indices: list[int] = []
        self.storage_scores: list[float] = []
        self.usage_counts: list[int] = []

    def __len__(self) -> int:
        return len(self.frames)

    def clear(self) -> None:
        self.frames.clear()
        self.poses.clear()
        self.actions.clear()
        self.frame_indices.clear()
        self.storage_scores.clear()
        self.usage_counts.clear()

    def add_frame(
        self,
        frame_latent: torch.Tensor,
        pose: torch.Tensor,
        action: Optional[torch.Tensor] = None,
        frame_idx: int = 0,
        storage_score: float = 1.0,
    ) -> None:
        """Store one ``[C,H,W]`` VAE latent and its retrieval metadata."""

        if frame_latent.ndim != 3:
            raise ValueError(
                f"frame_latent must have shape [C,H,W], got {tuple(frame_latent.shape)}"
            )
        if pose.ndim != 1 or pose.numel() < 2:
            raise ValueError(f"pose must be a 1-D vector with x/y, got {tuple(pose.shape)}")
        if action is None:
            action = torch.zeros(3, dtype=pose.dtype, device=pose.device)
        if action.ndim != 1 or action.numel() != 3:
            raise ValueError(f"action must have shape [3], got {tuple(action.shape)}")

        self.frames.append(frame_latent.detach().to(device="cpu"))
        self.poses.append(pose.detach().to(device="cpu"))
        self.actions.append(action.detach().to(device="cpu"))
        self.frame_indices.append(int(frame_idx))
        self.storage_scores.append(float(max(0.0, min(storage_score, 1.0))))
        self.usage_counts.append(0)

        if len(self) > self.max_size:
            for values in (
                self.frames,
                self.poses,
                self.actions,
                self.frame_indices,
                self.storage_scores,
                self.usage_counts,
            ):
                values.pop(0)

    def query(
        self,
        current_pose: torch.Tensor,
        target_action: Optional[torch.Tensor] = None,
        *,
        k: int = 8,
        update_usage: bool = True,
    ) -> Optional[RetrievalResult]:
        """Return the top-k memories and the exact scores used to select them."""

        if len(self) == 0:
            return None
        if k < 1:
            raise ValueError("k must be positive")
        if current_pose.ndim != 1 or current_pose.numel() < 2:
            raise ValueError("current_pose must be a 1-D vector containing x/y")

        device = current_pose.device
        poses = torch.stack(self.poses).to(device=device, dtype=current_pose.dtype)
        pose_dims = min(current_pose.numel(), poses.shape[1], 3)
        spatial_distance = torch.linalg.vector_norm(
            poses[:, :pose_dims] - current_pose[:pose_dims], dim=1
        )
        spatial_similarity = torch.exp(-spatial_distance / self.spatial_scale)

        if target_action is None:
            action_similarity = torch.zeros_like(spatial_similarity)
        else:
            actions = torch.stack(self.actions).to(device=device, dtype=target_action.dtype)
            action_similarity = self._compute_behavioral_similarity(target_action, actions)

        storage = torch.tensor(self.storage_scores, device=device, dtype=spatial_similarity.dtype)
        usage = torch.tensor(self.usage_counts, device=device, dtype=spatial_similarity.dtype)
        usage = torch.clamp(torch.log1p(usage) / 5.0, max=1.0)
        scores = (
            self.spatial_weight * spatial_similarity
            + self.action_weight * action_similarity
            + self.storage_weight * storage
            + self.usage_weight * usage
        )

        count = min(k, len(self))
        top = torch.topk(scores, count, sorted=True).indices
        host_indices = top.detach().cpu().tolist()
        if update_usage:
            for index in host_indices:
                self.usage_counts[index] += 1

        frames = torch.stack([self.frames[index] for index in host_indices]).to(device)
        frame_indices = torch.tensor(
            [self.frame_indices[index] for index in host_indices], device=device, dtype=torch.long
        )
        return RetrievalResult(frames=frames, frame_indices=frame_indices, scores=scores[top])

    def get_relevant_frames(
        self,
        current_pose: torch.Tensor,
        target_action: Optional[torch.Tensor] = None,
        k: int = 8,
    ) -> Optional[torch.Tensor]:
        """Compatibility helper returning only the selected frame latents."""

        result = self.query(current_pose, target_action, k=k)
        return None if result is None else result.frames

    @staticmethod
    def _compute_behavioral_similarity(
        target_action: torch.Tensor, memory_actions: torch.Tensor
    ) -> torch.Tensor:
        if target_action.ndim != 1 or target_action.numel() != 3:
            raise ValueError("target_action must have shape [3]")
        if memory_actions.ndim != 2 or memory_actions.shape[1] != 3:
            raise ValueError("memory_actions must have shape [M,3]")

        target_linear = target_action[:2]
        memory_linear = memory_actions[:, :2]
        target_norm = torch.linalg.vector_norm(target_linear)
        memory_norm = torch.linalg.vector_norm(memory_linear, dim=1)
        magnitude = torch.exp(-torch.abs(target_norm - memory_norm) / 2.0)

        direction = torch.ones_like(magnitude)
        if bool(target_norm > 0.1):
            valid = memory_norm > 0.1
            if bool(valid.any()):
                target_unit = target_linear / target_norm
                memory_unit = memory_linear[valid] / memory_norm[valid, None]
                direction[valid] = (memory_unit @ target_unit + 1.0) / 2.0

        yaw = MemoryBuffer._compute_rotation_similarity(
            target_action[2], memory_actions[:, 2]
        )
        return 0.4 * magnitude + 0.3 * direction + 0.3 * yaw

    @staticmethod
    def _compute_rotation_similarity(
        target_yaw: torch.Tensor, memory_yaw: torch.Tensor
    ) -> torch.Tensor:
        def category(values: torch.Tensor) -> torch.Tensor:
            return torch.where(
                values < 0.1,
                0,
                torch.where(values < 0.3, 1, torch.where(values < 1.0, 2, 3)),
            )

        target_abs = torch.abs(target_yaw)
        memory_abs = torch.abs(memory_yaw)
        target_category = category(target_abs)
        memory_category = category(memory_abs)
        direction_match = torch.sign(target_yaw) == torch.sign(memory_yaw)
        category_match = target_category == memory_category

        scores = torch.zeros_like(memory_yaw)
        if int(target_category.item()) == 0:
            scores[memory_category == 0] = 1.0
            scores[memory_category == 1] = 0.3
            scores[memory_category >= 2] = 0.1
            return scores

        base_scores = memory_yaw.new_tensor([0.9, 0.7, 0.5, 0.3])
        base = base_scores[target_category]
        perfect = direction_match & category_match
        scores[perfect] = base

        direction_only = direction_match & ~category_match
        scores[direction_only] = (
            torch.exp(-torch.abs(target_abs - memory_abs[direction_only]) / 0.5) * base * 0.7
        )
        scores[(~direction_match) & (memory_category > 0)] = 0.05
        scores[(target_category > 0) & (memory_category == 0)] = 0.1
        return scores


class SelectiveMemoryAttention(nn.Module):
    """Cross-attention from current CDiT tokens to retrieved memory tokens."""

    def __init__(self, hidden_size: int, num_heads: int) -> None:
        super().__init__()
        if hidden_size % num_heads != 0:
            raise ValueError("hidden_size must be divisible by num_heads")
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.to_q = nn.Linear(hidden_size, hidden_size, bias=False)
        self.to_k = nn.Linear(hidden_size, hidden_size, bias=False)
        self.to_v = nn.Linear(hidden_size, hidden_size, bias=False)
        self.to_out = nn.Linear(hidden_size, hidden_size)

    def forward(
        self,
        query_tokens: torch.Tensor,
        memory_tokens: torch.Tensor,
        memory_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if query_tokens.ndim != 3:
            raise ValueError("query_tokens must have shape [B,N,D]")
        if memory_tokens.ndim != 4:
            raise ValueError("memory_tokens must have shape [B,M,N,D]")

        batch, query_count, hidden = query_tokens.shape
        memory_batch, memory_count, token_count, memory_hidden = memory_tokens.shape
        if (memory_batch, memory_hidden) != (batch, hidden):
            raise ValueError("memory/query batch and hidden dimensions must match")

        flattened = memory_tokens.reshape(batch, memory_count * token_count, hidden)
        query = self.to_q(query_tokens).reshape(
            batch, query_count, self.num_heads, self.head_dim
        ).transpose(1, 2)
        key = self.to_k(flattened).reshape(
            batch, memory_count * token_count, self.num_heads, self.head_dim
        ).transpose(1, 2)
        value = self.to_v(flattened).reshape(
            batch, memory_count * token_count, self.num_heads, self.head_dim
        ).transpose(1, 2)

        logits = query @ key.transpose(-2, -1) / math.sqrt(self.head_dim)
        if memory_mask is not None:
            if memory_mask.shape != (batch, memory_count):
                raise ValueError("memory_mask must have shape [B,M]")
            token_mask = memory_mask[:, :, None].expand(-1, -1, token_count).reshape(batch, -1)
            logits = logits.masked_fill(~token_mask[:, None, None, :], -torch.inf)
            all_masked = ~token_mask.any(dim=1)
            if bool(all_masked.any()):
                logits[all_masked] = 0

        attention = torch.softmax(logits, dim=-1)
        output = attention @ value
        output = output.transpose(1, 2).reshape(batch, query_count, hidden)
        output = self.to_out(output)
        if memory_mask is not None:
            output = output * memory_mask.any(dim=1)[:, None, None]
        return output


class HybridCDiTBlock(nn.Module):
    """Checkpoint-compatible CDiT block with a zero-initialized memory branch."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        *,
        enable_memory: bool = True,
        **block_kwargs,
    ) -> None:
        super().__init__()
        self.enable_memory = enable_memory

        # Keep baseline module names and shapes unchanged for checkpoint loading.
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.norm_cond = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.cttn = nn.MultiheadAttention(
            hidden_size,
            num_heads=num_heads,
            add_bias_kv=True,
            bias=True,
            batch_first=True,
            **block_kwargs,
        )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_size, 11 * hidden_size, bias=True)
        )
        self.norm3 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.mlp = Mlp(
            in_features=hidden_size,
            hidden_features=mlp_hidden_dim,
            act_layer=lambda: nn.GELU(approximate="tanh"),
            drop=0,
        )

        if enable_memory:
            self.memory_norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
            self.memory_attn = SelectiveMemoryAttention(hidden_size, num_heads)
            self.memory_adaLN = nn.Sequential(
                nn.SiLU(), nn.Linear(hidden_size, 3 * hidden_size, bias=True)
            )

    def forward(
        self,
        x: torch.Tensor,
        c: torch.Tensor,
        x_cond: torch.Tensor,
        memory_tokens: Optional[torch.Tensor] = None,
        memory_mask: Optional[torch.Tensor] = None,
        memory_activation: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        (
            shift_msa,
            scale_msa,
            gate_msa,
            shift_ca_xcond,
            scale_ca_xcond,
            shift_ca_x,
            scale_ca_x,
            gate_ca_x,
            shift_mlp,
            scale_mlp,
            gate_mlp,
        ) = self.adaLN_modulation(c).chunk(11, dim=1)
        x = x + gate_msa[:, None] * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        x_cond_norm = modulate(self.norm_cond(x_cond), shift_ca_xcond, scale_ca_xcond)
        x = x + gate_ca_x[:, None] * self.cttn(
            query=modulate(self.norm2(x), shift_ca_x, scale_ca_x),
            key=x_cond_norm,
            value=x_cond_norm,
            need_weights=False,
        )[0]

        if self.enable_memory and memory_tokens is not None:
            shift_mem, scale_mem, gate_mem = self.memory_adaLN(c).chunk(3, dim=1)
            memory_output = self.memory_attn(
                modulate(self.memory_norm(x), shift_mem, scale_mem),
                memory_tokens,
                memory_mask,
            )
            if memory_activation is not None:
                gate_mem = gate_mem * memory_activation.reshape(-1, 1)
            x = x + gate_mem[:, None] * memory_output

        x = x + gate_mlp[:, None] * self.mlp(modulate(self.norm3(x), shift_mlp, scale_mlp))
        return x


class HybridCDiT(nn.Module):
    """CDiT with explicit, externally retrieved long-term-memory latents."""

    def __init__(
        self,
        input_size: int = 32,
        context_size: int = 4,
        patch_size: int = 2,
        in_channels: int = 4,
        hidden_size: int = 1152,
        depth: int = 28,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        learn_sigma: bool = True,
        memory_enabled: bool = True,
        memory_layers: Optional[Sequence[int]] = None,
        **_: object,
    ) -> None:
        super().__init__()
        self.context_size = context_size
        self.learn_sigma = learn_sigma
        self.in_channels = in_channels
        self.out_channels = in_channels * 2 if learn_sigma else in_channels
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.memory_enabled = memory_enabled

        if memory_layers is None:
            memory_layers = range(depth // 2, depth)
        invalid_layers = sorted(set(memory_layers) - set(range(depth)))
        if invalid_layers:
            raise ValueError(f"memory_layers outside model depth: {invalid_layers}")
        self.memory_layers = frozenset(memory_layers)

        self.x_embedder = PatchEmbed(input_size, patch_size, in_channels, hidden_size, bias=True)
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.y_embedder = ActionEmbedder(hidden_size)
        self.time_embedder = TimestepEmbedder(hidden_size)
        num_patches = self.x_embedder.num_patches
        self.pos_embed = nn.Parameter(
            torch.zeros(context_size + 1, num_patches, hidden_size), requires_grad=True
        )
        self.blocks = nn.ModuleList(
            [
                HybridCDiTBlock(
                    hidden_size,
                    num_heads,
                    mlp_ratio=mlp_ratio,
                    enable_memory=memory_enabled and index in self.memory_layers,
                )
                for index in range(depth)
            ]
        )
        self.final_layer = FinalLayer(hidden_size, patch_size, self.out_channels)
        self.initialize_weights()

    def initialize_weights(self) -> None:
        def initialize(module: nn.Module) -> None:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(initialize)
        nn.init.normal_(self.pos_embed, std=0.02)
        weight = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(weight.view(weight.shape[0], -1))
        nn.init.constant_(self.x_embedder.proj.bias, 0)
        for embedder in (
            self.y_embedder.x_emb,
            self.y_embedder.y_emb,
            self.y_embedder.angle_emb,
            self.t_embedder,
            self.time_embedder,
        ):
            nn.init.normal_(embedder.mlp[0].weight, std=0.02)
            nn.init.normal_(embedder.mlp[2].weight, std=0.02)

        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)
            if block.enable_memory:
                # The hybrid is exactly the baseline at initialization.
                nn.init.constant_(block.memory_adaLN[-1].weight, 0)
                nn.init.constant_(block.memory_adaLN[-1].bias, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def encode_memory_latents(self, memory_latents: torch.Tensor) -> torch.Tensor:
        if memory_latents.ndim != 5:
            raise ValueError("memory_latents must have shape [B,M,C,H,W]")
        batch, memories = memory_latents.shape[:2]
        tokens = self.x_embedder(memory_latents.flatten(0, 1))
        tokens = tokens + self.pos_embed[self.context_size]
        return tokens.unflatten(0, (batch, memories))

    def unpatchify(self, x: torch.Tensor) -> torch.Tensor:
        channels = self.out_channels
        patch = self.x_embedder.patch_size[0]
        height = width = int(x.shape[1] ** 0.5)
        if height * width != x.shape[1]:
            raise ValueError("token count must form a square grid")
        x = x.reshape(x.shape[0], height, width, patch, patch, channels)
        x = torch.einsum("nhwpqc->nchpwq", x)
        return x.reshape(x.shape[0], channels, height * patch, height * patch)

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        y: torch.Tensor,
        x_cond: torch.Tensor,
        rel_t: torch.Tensor,
        memory_latents: Optional[torch.Tensor] = None,
        memory_mask: Optional[torch.Tensor] = None,
        memory_activation: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = self.x_embedder(x) + self.pos_embed[self.context_size]
        x_cond = self.x_embedder(x_cond.flatten(0, 1)).unflatten(
            0, (x_cond.shape[0], x_cond.shape[1])
        )
        x_cond = x_cond + self.pos_embed[: self.context_size]
        x_cond = x_cond.flatten(1, 2)

        conditioning = (
            self.t_embedder(t[..., None])
            + self.y_embedder(y)
            + self.time_embedder(rel_t[..., None])
        )
        memory_tokens = None
        if memory_latents is not None:
            if not self.memory_enabled:
                raise ValueError("memory_latents supplied to a memory-disabled model")
            memory_tokens = self.encode_memory_latents(memory_latents)

        for block in self.blocks:
            x = block(
                x,
                conditioning,
                x_cond,
                memory_tokens,
                memory_mask,
                memory_activation,
            )
        return self.unpatchify(self.final_layer(x, conditioning))


def HybridCDiT_XL_2(**kwargs: object) -> HybridCDiT:
    return HybridCDiT(depth=28, hidden_size=1152, patch_size=2, num_heads=16, **kwargs)


def HybridCDiT_L_2(**kwargs: object) -> HybridCDiT:
    return HybridCDiT(depth=24, hidden_size=1024, patch_size=2, num_heads=16, **kwargs)


def HybridCDiT_B_2(**kwargs: object) -> HybridCDiT:
    return HybridCDiT(depth=12, hidden_size=768, patch_size=2, num_heads=12, **kwargs)


def HybridCDiT_S_2(**kwargs: object) -> HybridCDiT:
    return HybridCDiT(depth=12, hidden_size=384, patch_size=2, num_heads=6, **kwargs)


HybridCDiT_models = {
    "HybridCDiT-XL/2": HybridCDiT_XL_2,
    "HybridCDiT-L/2": HybridCDiT_L_2,
    "HybridCDiT-B/2": HybridCDiT_B_2,
    "HybridCDiT-S/2": HybridCDiT_S_2,
}
