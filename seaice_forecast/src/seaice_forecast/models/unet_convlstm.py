"""
U-Net + ConvLSTM Architecture for Spatial-Temporal Sea-Ice Forecasting.

Integrates a recurrent ConvLSTM module into the U-Net architecture at the bottleneck
(Option B: Hybrid Bottleneck ConvLSTM).

Input:  [Batch, T=7, C=7, Height=332, Width=316]
Output: [Batch, 1, Height=332, Width=316] (next-day SIC forecast, sigmoid bounded [0, 1])

Design rationale:
- Bottleneck ConvLSTM operates on low-resolution feature maps (e.g. 20x19), capturing
  synoptic-scale temporal advection and dynamics with high memory efficiency and numerical stability.
- Reuses U-Net encoder and decoder blocks with skip connections to preserve fine spatial details.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional, Union
import logging

from seaice_forecast.models.unet import ConvBlock, EncoderBlock, DecoderBlock
from seaice_forecast.models.convlstm import ConvLSTM

logger = logging.getLogger(__name__)


class UNetConvLSTM(nn.Module):
    """
    Hybrid U-Net with Bottleneck ConvLSTM for spatial-temporal sequence forecasting.
    """

    def __init__(
        self,
        in_channels: int = 7,
        output_channels: int = 1,
        seq_len: int = 7,
        encoder_channels: List[int] = [32, 64, 128, 256],
        bottleneck_channels: Optional[int] = None,
        convlstm_layers: int = 1,
        use_batch_norm: bool = True,
        dropout: float = 0.0,
        output_activation: str = "sigmoid",
        residual: bool = False,
        sic_channel: int = 0
    ):
        """
        Args:
            in_channels: Number of physical variables per timestep (default: 7 for Phase C).
            output_channels: Number of forecast target channels (default: 1 for SIC).
            seq_len: History sequence length in days (default: 7).
            encoder_channels: Channel progression through spatial encoder.
            bottleneck_channels: Number of channels in the bottleneck ConvLSTM (defaults to encoder_channels[-1] * 2).
            convlstm_layers: Number of stacked ConvLSTM layers at bottleneck.
            use_batch_norm: Whether to use batch normalization.
            dropout: Spatial dropout probability.
            output_activation: Output activation ('sigmoid' or 'none').
            residual: If True, predict the CHANGE in SIC rather than SIC itself:

                          out = clamp(SIC[t] + delta, 0, 1)

                      The encoder/decoder path downsamples 332x316 to roughly
                      20x19 before reconstructing, so reproducing today's field
                      exactly - which is what persistence does, and persistence
                      is a very strong baseline for daily SIC - means pushing
                      the whole field through that bottleneck and back. In
                      residual mode a zero output *is* persistence, so the
                      network starts at persistence-level skill and only has to
                      learn the correction. Standard practice in sea-ice and
                      weather nowcasting.
            sic_channel: Index of SIC within the per-timestep variable channels
                      (0 for the standard ordering).
        """
        super().__init__()

        self.in_channels = in_channels
        self.output_channels = output_channels
        self.seq_len = seq_len
        self.residual = residual
        self.sic_channel = sic_channel
        self.encoder_channels = encoder_channels
        self.bottleneck_channels = bottleneck_channels or (encoder_channels[-1] * 2)
        self.convlstm_layers = convlstm_layers

        # 1. Spatial Encoder (shared across all timesteps)
        self.encoders = nn.ModuleList()
        in_ch = in_channels
        for out_ch in encoder_channels:
            self.encoders.append(
                EncoderBlock(in_ch, out_ch, use_batch_norm, dropout)
            )
            in_ch = out_ch

        # Pre-bottleneck projection
        self.bottleneck_conv = ConvBlock(
            encoder_channels[-1],
            self.bottleneck_channels,
            use_batch_norm,
            dropout
        )

        # 2. Temporal Bottleneck (ConvLSTM)
        self.temporal_bottleneck = ConvLSTM(
            in_channels=self.bottleneck_channels,
            hidden_channels=self.bottleneck_channels,
            num_layers=convlstm_layers,
            kernel_size=3,
            dropout=dropout
        )

        # Post-bottleneck refinement
        self.post_bottleneck_conv = ConvBlock(
            self.bottleneck_channels,
            self.bottleneck_channels,
            use_batch_norm,
            dropout
        )

        # 3. Spatial Decoder
        self.decoders = nn.ModuleList()
        decoder_channels = list(reversed(encoder_channels))
        in_ch = self.bottleneck_channels

        for out_ch in decoder_channels:
            self.decoders.append(
                DecoderBlock(in_ch, out_ch, use_batch_norm, dropout)
            )
            in_ch = out_ch

        # 4. Output Layer
        self.output_conv = nn.Conv2d(
            encoder_channels[0],
            output_channels,
            kernel_size=1
        )

        if output_activation == "sigmoid":
            self.output_activation = nn.Sigmoid()
        else:
            self.output_activation = nn.Identity()

        if self.residual:
            # Start training exactly at persistence: a zero delta reproduces
            # today's SIC, so epoch 0 already has persistence-level skill and
            # gradients only need to learn the correction. Same idea as
            # zero-initialising residual branches in ResNet.
            nn.init.zeros_(self.output_conv.weight)
            if self.output_conv.bias is not None:
                nn.init.zeros_(self.output_conv.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Spatial-temporal sequence tensor [B, T, C, H, W] or [B, T*C, H, W].

        Returns:
            Forecast tensor [B, output_channels, H, W], bounded in [0, 1].
        """
        # Handle 4D input if passed as concatenated channels [B, T*C, H, W]
        if x.dim() == 4:
            B, TC, H, W = x.shape
            T = self.seq_len
            C = self.in_channels
            if TC == T * C:
                x = x.view(B, T, C, H, W)
            else:
                raise ValueError(f"Input channels {TC} does not match seq_len ({T}) * in_channels ({C})")

        B, T, C, H, W = x.shape

        # Step 1: Encode each timestep with shared spatial encoder
        # Flatten [B, T, C, H, W] -> [B*T, C, H, W] for high-throughput vectorized 2D CNN
        x_flat = x.view(B * T, C, H, W)

        skip_connections = []
        cur = x_flat
        for encoder in self.encoders:
            cur, skip = encoder(cur)
            # Reshape skip to [B, T, C_skip, H_skip, W_skip]
            skip_connections.append(skip.view(B, T, skip.shape[1], skip.shape[2], skip.shape[3]))

        # Project to bottleneck: [B*T, bottleneck_channels, H_bot, W_bot]
        bottleneck_flat = self.bottleneck_conv(cur)
        _, C_bot, H_bot, W_bot = bottleneck_flat.shape
        bottleneck_seq = bottleneck_flat.view(B, T, C_bot, H_bot, W_bot)

        # Step 2: Temporal Modeling via ConvLSTM at bottleneck
        # bottleneck_out: [B, T, bottleneck_channels, H_bot, W_bot]
        bottleneck_out, last_states = self.temporal_bottleneck(bottleneck_seq)

        # Extract the final timestep representation H_{T-1}
        # containing the integrated temporal trajectory across all T days
        h_final = bottleneck_out[:, -1]  # [B, bottleneck_channels, H_bot, W_bot]
        h_refined = self.post_bottleneck_conv(h_final)

        # Step 3: Decode with skip connections
        # Use skip connections from the most recent observed timestep (T-1)
        # to ensure optimal spatial boundary alignment with the target
        dec = h_refined
        num_stages = len(self.decoders)

        for i, decoder in enumerate(self.decoders):
            # Skip from corresponding encoder stage for final timestep
            skip_t = skip_connections[num_stages - 1 - i][:, -1]  # [B, C_skip, H_skip, W_skip]
            dec = decoder(dec, skip_t)

        # Step 4: Final 1x1 conv and activation
        out = self.output_conv(dec)

        if self.residual:
            # Predict the day-on-day CHANGE and add it to the last observed SIC.
            # tanh bounds the correction to [-1, 1], which spans the full range
            # of a physically possible one-step SIC change; a zero output
            # reproduces persistence exactly.
            delta = torch.tanh(out)
            last_sic = x[:, -1, self.sic_channel:self.sic_channel + 1]  # [B,1,H,W]
            return torch.clamp(last_sic + delta, 0.0, 1.0)

        out = self.output_activation(out)

        return out


def create_unet_convlstm_from_config(config: dict) -> UNetConvLSTM:
    """Factory function to build UNetConvLSTM from project config dictionary."""
    model_cfg = config.get("model", {})
    return UNetConvLSTM(
        in_channels=model_cfg.get("in_channels", 7),
        output_channels=model_cfg.get("output_channels", 1),
        seq_len=model_cfg.get("seq_len", 7),
        encoder_channels=model_cfg.get("encoder_channels", [32, 64, 128, 256]),
        bottleneck_channels=model_cfg.get("bottleneck_channels", None),
        convlstm_layers=model_cfg.get("convlstm_layers", 1),
        use_batch_norm=model_cfg.get("use_batch_norm", True),
        dropout=model_cfg.get("dropout", 0.0),
        output_activation=model_cfg.get("output_activation", "sigmoid")
    )
