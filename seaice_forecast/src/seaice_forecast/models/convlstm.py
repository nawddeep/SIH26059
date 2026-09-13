"""
ConvLSTM (Convolutional Long Short-Term Memory) Module for Spatial-Temporal Modeling.

Implements ConvLSTMCell and multi-layer ConvLSTM for capturing 2D spatial-temporal
dynamics in polar sea-ice and environmental forcing sequences.

Reference:
Shi et al., "Convolutional LSTM Network: A Machine Learning Approach for Precipitation Nowcasting", NIPS 2015.
"""

import torch
import torch.nn as nn
from typing import Tuple, List, Optional


class ConvLSTMCell(nn.Module):
    """
    2D Convolutional LSTM Cell.

    Computes input, forget, cell, and output gates simultaneously via a unified 2D convolution
    over concatenated inputs and hidden states:
        [i_t, f_t, g_t, o_t] = Conv2d([X_t, H_{t-1}])
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        kernel_size: int = 3,
        bias: bool = True
    ):
        """
        Args:
            in_channels: Number of input channels.
            hidden_channels: Number of hidden/state channels.
            kernel_size: Size of convolutional kernel (default: 3).
            bias: Whether to add a learnable bias.
        """
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2

        # 4 gates: input (i), forget (f), candidate (g), output (o)
        self.conv = nn.Conv2d(
            in_channels=in_channels + hidden_channels,
            out_channels=4 * hidden_channels,
            kernel_size=kernel_size,
            padding=self.padding,
            bias=bias
        )

        self._reset_parameters()

    def _reset_parameters(self):
        """Initialize weights with Xavier uniform and positive forget gate bias."""
        nn.init.xavier_uniform_(self.conv.weight)
        if self.conv.bias is not None:
            nn.init.zeros_(self.conv.bias)
            # Initialize forget gate bias to 1.0 (recommended for LSTM stability)
            self.conv.bias.data[self.hidden_channels:2 * self.hidden_channels].fill_(1.0)

    def forward(
        self,
        x: torch.Tensor,
        state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for a single timestep.

        Args:
            x: Input tensor of shape [B, in_channels, H, W].
            state: Tuple of (h, c), each of shape [B, hidden_channels, H, W].
                   If None, initialized to zeros.

        Returns:
            Tuple of (h_next, c_next), each of shape [B, hidden_channels, H, W].
        """
        batch_size, _, height, width = x.shape

        if state is None:
            device = x.device
            dtype = x.dtype
            h = torch.zeros(batch_size, self.hidden_channels, height, width, device=device, dtype=dtype)
            c = torch.zeros(batch_size, self.hidden_channels, height, width, device=device, dtype=dtype)
        else:
            h, c = state

        # Concatenate along channel dimension
        combined = torch.cat([x, h], dim=1)
        gates = self.conv(combined)

        # Split into 4 gates
        i, f, g, o = torch.split(gates, self.hidden_channels, dim=1)

        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        g = torch.tanh(g)
        o = torch.sigmoid(o)

        c_next = f * c + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next


class ConvLSTM(nn.Module):
    """
    Multi-layer Convolutional LSTM.

    Processes sequences of spatial tensors:
        Input:  [B, T, in_channels, H, W]
        Output: [B, T, hidden_channels, H, W] (or last hidden state [B, hidden_channels, H, W])
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        num_layers: int = 1,
        kernel_size: int = 3,
        return_all_layers: bool = False,
        dropout: float = 0.0
    ):
        """
        Args:
            in_channels: Number of input channels.
            hidden_channels: Number of hidden channels per layer.
            num_layers: Number of stacked ConvLSTM layers.
            kernel_size: Convolution kernel size.
            return_all_layers: If True, returns outputs from all layers.
            dropout: Dropout applied between stacked layers.
        """
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.return_all_layers = return_all_layers

        cell_list = []
        for i in range(num_layers):
            cur_in_ch = in_channels if i == 0 else hidden_channels
            cell_list.append(
                ConvLSTMCell(
                    in_channels=cur_in_ch,
                    hidden_channels=hidden_channels,
                    kernel_size=kernel_size
                )
            )

        self.cell_list = nn.ModuleList(cell_list)
        self.dropout = nn.Dropout2d(dropout) if dropout > 0.0 else nn.Identity()

    def forward(
        self,
        x: torch.Tensor,
        hidden_states: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None
    ) -> Tuple[torch.Tensor, List[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Forward pass across all timesteps.

        Args:
            x: Input sequence [B, T, C_in, H, W].
            hidden_states: Optional initial list of (h, c) tuples for each layer.

        Returns:
            layer_output: Sequence of hidden states from the top layer [B, T, C_out, H, W].
            last_states: List of (h, c) tuples for each layer at final timestep T-1.
        """
        batch_size, seq_len, _, height, width = x.shape
        device = x.device
        dtype = x.dtype

        if hidden_states is None:
            hidden_states = []
            for _ in range(self.num_layers):
                h0 = torch.zeros(batch_size, self.hidden_channels, height, width, device=device, dtype=dtype)
                c0 = torch.zeros(batch_size, self.hidden_channels, height, width, device=device, dtype=dtype)
                hidden_states.append((h0, c0))

        cur_layer_input = x
        all_layer_outputs = []
        last_states = []

        for layer_idx in range(self.num_layers):
            cell = self.cell_list[layer_idx]
            h, c = hidden_states[layer_idx]
            output_seq = []

            for t in range(seq_len):
                x_t = cur_layer_input[:, t]  # [B, C, H, W]
                h, c = cell(x_t, (h, c))
                output_seq.append(h)

            # Stack along sequence dimension: [B, T, C_hidden, H, W]
            layer_output = torch.stack(output_seq, dim=1)
            if layer_idx < self.num_layers - 1:
                # Apply dropout between layers
                B, T, C, H, W = layer_output.shape
                layer_output = self.dropout(layer_output.view(B * T, C, H, W)).view(B, T, C, H, W)

            cur_layer_input = layer_output
            all_layer_outputs.append(layer_output)
            last_states.append((h, c))

        if self.return_all_layers:
            return all_layer_outputs, last_states
        return all_layer_outputs[-1], last_states
