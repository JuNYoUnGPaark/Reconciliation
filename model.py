import math
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


"""
  - Lightweight Conflict-aware Reconciliation framework
  - Author: JunYoung Park and Myung-Kyu Yi
"""


class MultiScaleLearnableSTFT(nn.Module):
    """Learnable multi-scale temporal filterbank for frequency-oriented representation"""
    def __init__(
        self,
        in_channels: int,
        bins_per_scale: int,
        kernel_sizes: list[int],
        hop: int,
    ):
        super().__init__()

        self.banks = nn.ModuleList()
        self.norms = nn.ModuleList()

        for kernel_size in kernel_sizes:
            self.banks.append(
                nn.Conv1d(
                    in_channels=in_channels,
                    out_channels=bins_per_scale,
                    kernel_size=kernel_size,
                    stride=hop,
                    padding=kernel_size // 2,
                    bias=False,
                )
            )
            self.norms.append(nn.BatchNorm1d(bins_per_scale))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        specs = []

        for conv, bn in zip(self.banks, self.norms):
            c = conv(x)
            mag = torch.sqrt(c.pow(2) + 1e-6)
            log_mag = bn(torch.log1p(mag))
            specs.append(log_mag)

        min_t = min(s.size(2) for s in specs)
        specs = [s[:, :, :min_t] for s in specs]

        return torch.cat(specs, dim=1).unsqueeze(1)


class FreqEncoder(nn.Module):
    """Compact CNN encoder for filter-time representations"""

    def __init__(self, out_dim: int):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.GELU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

        self.fc = nn.Sequential(
            nn.Linear(32, out_dim),
            nn.LayerNorm(out_dim),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.net(x).view(x.size(0), -1)
        return self.fc(h)

      
class PerceiverCrossAttention(nn.Module):
    """oss-attention module where learnable latent tokens attend to temporal input tokens"""
    def __init__(
        self,
        d_latent: int,
        d_input: int,
        n_heads: int,
        dropout: float,
    ):
        super().__init__()

        self.attn = nn.MultiheadAttention(
            embed_dim=d_latent,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.norm_q = nn.LayerNorm(d_latent)
        self.norm_kv = nn.LayerNorm(d_input)
        self.proj_kv = nn.Linear(d_input, d_latent) if d_input != d_latent else nn.Identity()

    def forward(
        self,
        latents: torch.Tensor,
        inputs: torch.Tensor,
    ) -> torch.Tensor:
        q = self.norm_q(latents)
        kv = self.proj_kv(self.norm_kv(inputs))

        out, _ = self.attn(q, kv, kv)

        return latents + out


class PerceiverSelfBlock(nn.Module):
    """Self-attention block for refining latent temporal representations"""
    def __init__(
        self,
        d_latent: int,
        n_heads: int,
        ff_mult: int,
        dropout: float,
    ):
        super().__init__()

        self.norm1 = nn.LayerNorm(d_latent)
        self.attn = nn.MultiheadAttention(
            embed_dim=d_latent,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.norm2 = nn.LayerNorm(d_latent)

        self.ffn = nn.Sequential(
            nn.Linear(d_latent, d_latent * ff_mult),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_latent * ff_mult, d_latent),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        x = x + self.attn(h, h, h)[0]
        x = x + self.ffn(self.norm2(x))

        return x


class PerceiverStyleEncoder(nn.Module):
    """Perceiver-style time-domain encoder for multichannel sensor sequences"""
    def __init__(
        self,
        input_dim: int,
        latent_n: int,
        latent_d: int,
        n_heads: int,
        num_blocks: int,
        output_dim: int,
        dropout: float,
    ):
        super().__init__()

        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, latent_d),
            nn.LayerNorm(latent_d),
        )

        self.pos_enc = nn.Parameter(torch.randn(1, 512, latent_d) * 0.02)
        self.latents = nn.Parameter(torch.randn(1, latent_n, latent_d) * 0.02)

        self.cross_attn = PerceiverCrossAttention(
            d_latent=latent_d,
            d_input=latent_d,
            n_heads=n_heads,
            dropout=dropout,
        )

        self.self_attn_blocks = nn.ModuleList(
            [
                PerceiverSelfBlock(
                    d_latent=latent_d,
                    n_heads=n_heads,
                    ff_mult=2,
                    dropout=dropout,
                )
                for _ in range(num_blocks)
            ]
        )

        self.norm = nn.LayerNorm(latent_d)

        self.fc_out = nn.Sequential(
            nn.Linear(latent_d, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape

        inp = self.input_proj(x) + self.pos_enc[:, :seq_len, :]
        latents = self.latents.expand(batch_size, -1, -1)

        latents = self.cross_attn(latents, inp)

        for block in self.self_attn_blocks:
            latents = block(latents)

        h = self.norm(latents).mean(dim=1)

        return self.fc_out(h)


class ConflictEstimator(nn.Module):
    """Sample-wise coefficient estimator based on branch similarity, distance, uncertainty, and agreement"""
    def __init__(
        self,
        n_features: int,
        lambda_floor: float,
        n_classes: int,
    ):
        super().__init__()

        self.lambda_floor = lambda_floor
        self.log_k = math.log(n_classes)

        self.mlp = nn.Sequential(
            nn.Linear(n_features, 16),
            nn.GELU(),
            nn.Linear(16, 1),
        )

    def forward(
        self,
        z_t: torch.Tensor,
        z_f: torch.Tensor,
        p_t: torch.Tensor,
        p_f: torch.Tensor,
    ) -> torch.Tensor:
        z_t_d = z_t.detach()
        z_f_d = z_f.detach()
        p_t_d = p_t.detach()
        p_f_d = p_f.detach()

        cos_sim = F.cosine_similarity(z_t_d, z_f_d, dim=-1, eps=1e-6)
        cos_norm = (cos_sim + 1.0) / 2.0

        l2_dist = torch.norm(z_t_d - z_f_d, dim=-1, p=2)
        denom = z_t_d.norm(dim=-1) + z_f_d.norm(dim=-1) + 1e-6
        l2_norm = (l2_dist / denom).clamp(0.0, 1.0)

        ent_t = -(p_t_d * torch.log(p_t_d + 1e-8)).sum(dim=-1)
        ent_f = -(p_f_d * torch.log(p_f_d + 1e-8)).sum(dim=-1)

        ent_t = (ent_t / self.log_k).clamp(0.0, 1.0)
        ent_f = (ent_f / self.log_k).clamp(0.0, 1.0)

        top1_agree = (p_t_d.argmax(dim=-1) == p_f_d.argmax(dim=-1)).float()

        features = torch.stack(
            [cos_norm, l2_norm, ent_t, ent_f, top1_agree],
            dim=-1,
        )

        raw = torch.sigmoid(self.mlp(features).squeeze(-1))
        lam = self.lambda_floor + (1.0 - self.lambda_floor) * raw

        return lam


class ConflictAwareFusion(nn.Module):
    """Conflict-aware reconciliation module combining selective gated fusion and symmetric mean fusion"""
    def __init__(
        self,
        feat_dim: int,
        dropout: float,
    ):
        super().__init__()

        self.proj_t = nn.Linear(feat_dim, feat_dim)
        self.proj_f = nn.Linear(feat_dim, feat_dim)

        self.gate = nn.Sequential(
            nn.Linear(feat_dim * 2 + 1, feat_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(feat_dim, feat_dim),
            nn.Sigmoid(),
        )

        self.out_proj = nn.Sequential(
            nn.Linear(feat_dim, feat_dim),
            nn.LayerNorm(feat_dim),
            nn.GELU(),
        )

    def forward(
        self,
        z_t: torch.Tensor,
        z_f: torch.Tensor,
        lam: torch.Tensor,
    ) -> torch.Tensor:
        lam_e = lam.unsqueeze(-1)

        gate = self.gate(torch.cat([z_t, z_f, lam_e], dim=-1))

        z_cross = gate * self.proj_t(z_t) + (1.0 - gate) * self.proj_f(z_f)
        z_mean = 0.5 * (z_t + z_f)

        z_fused = (1.0 - lam_e) * z_cross + lam_e * z_mean

        return self.out_proj(z_fused)


class ReconcileHAR(nn.Module):
    """Lightweight conflict-aware time-frequency reconciliation model"""
    def __init__(
        self,
        n_channels: int,
        n_classes: int,
        stft_bins_per_scale: int,
        stft_kernel_sizes: list[int],
        stft_hop: int,
        freq_cnn_dim: int,
        latent_n: int,
        latent_d: int,
        perceiver_heads: int,
        perceiver_blocks: int,
        perceiver_dim: int,
        dropout: float,
        lambda_floor: float,
        residual_branch_weight: float,
    ):
        super().__init__()
        self.residual_branch_weight = residual_branch_weight

        self.stft = MultiScaleLearnableSTFT(
            in_channels=n_channels,
            bins_per_scale=stft_bins_per_scale,
            kernel_sizes=stft_kernel_sizes,
            hop=stft_hop,
        )

        self.freq_encoder = FreqEncoder(out_dim=freq_cnn_dim)

        self.time_encoder = PerceiverStyleEncoder(
            input_dim=n_channels,
            latent_n=latent_n,
            latent_d=latent_d,
            n_heads=perceiver_heads,
            num_blocks=perceiver_blocks,
            output_dim=perceiver_dim,
            dropout=dropout,
        )

        assert freq_cnn_dim == perceiver_dim

        feat_dim = perceiver_dim

        self.fusion = ConflictAwareFusion(
            feat_dim=feat_dim,
            dropout=dropout,
        )

        def build_classifier(drop_rate: float) -> nn.Sequential:
            return nn.Sequential(
                nn.Dropout(drop_rate),
                nn.Linear(feat_dim, n_classes),
            )

        self.cls_time = build_classifier(0.1)
        self.cls_freq = build_classifier(0.1)
        self.classifier = build_classifier(dropout)

        self.residual_proj = nn.Sequential(
            nn.Linear(feat_dim, feat_dim),
            nn.LayerNorm(feat_dim),
        )

        self.conflict_est = ConflictEstimator(
            n_features=5,
            lambda_floor=lambda_floor,
            n_classes=n_classes,
        )

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)

                if m.bias is not None:
                    nn.init.zeros_(m.bias)

            elif isinstance(m, (nn.Conv1d, nn.Conv2d)):
                nn.init.kaiming_normal_(
                    m.weight,
                    mode="fan_out",
                    nonlinearity="relu",
                )

    def extract_features(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        spec = self.stft(x)

        z_f = self.freq_encoder(spec)
        z_t = self.time_encoder(x.permute(0, 2, 1))

        logits_t = self.cls_time(z_t)
        logits_f = self.cls_freq(z_f)

        p_t = F.softmax(logits_t, dim=-1)
        p_f = F.softmax(logits_f, dim=-1)

        lam = self.conflict_est(z_t, z_f, p_t, p_f)
        z_fused = self.fusion(z_t, z_f, lam)

        w = self.residual_branch_weight
        z_out = z_fused + w * self.residual_proj(z_t + z_f)

        features = {
            "z_t": z_t,
            "z_f": z_f,
            "z_fused": z_fused,
            "z_out": z_out,
            "lam": lam,
            "p_t": p_t,
            "p_f": p_f,
            "logits_t": logits_t,
            "logits_f": logits_f,
        }

        return features

    def forward(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        features = self.extract_features(x)
        logits = self.classifier(features["z_out"])

        return logits, features["logits_t"], features["logits_f"], features
