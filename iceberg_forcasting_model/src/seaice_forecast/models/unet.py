"""
U-Net architecture for sea-ice concentration forecasting.

Input: [Batch, 7, Height, Width] - 7 days of SIC history
Output: [Batch, 1, Height, Width] - Next-day SIC forecast

Architecture:
- Encoder: Progressive downsampling with Conv2D blocks
- Bottleneck: Lowest resolution representation
- Decoder: Progressive upsampling with skip connections
- Output: Sigmoid activation for [0,1] bounded SIC prediction
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ConvBlock(nn.Module):
    """
    Convolutional block: Conv2D -> BatchNorm -> ReLU -> Conv2D -> BatchNorm -> ReLU
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        use_batch_norm: bool = True,
        dropout: float = 0.0
    ):
        """
        Initialize convolutional block.
        
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            use_batch_norm: Whether to use batch normalization
            dropout: Dropout probability (0 = no dropout)
        """
        super().__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels) if use_batch_norm else nn.Identity()
        self.relu1 = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels) if use_batch_norm else nn.Identity()
        self.relu2 = nn.ReLU(inplace=True)
        
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        
        x = self.dropout(x)
        
        return x


class EncoderBlock(nn.Module):
    """
    Encoder block: ConvBlock -> MaxPool
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        use_batch_norm: bool = True,
        dropout: float = 0.0
    ):
        """
        Initialize encoder block.
        
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            use_batch_norm: Whether to use batch normalization
            dropout: Dropout probability
        """
        super().__init__()
        
        self.conv_block = ConvBlock(in_channels, out_channels, use_batch_norm, dropout)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
    
    def forward(self, x: torch.Tensor) -> tuple:
        """
        Forward pass.
        
        Returns:
            Tuple of (pooled output, skip connection)
        """
        skip = self.conv_block(x)
        x = self.pool(skip)
        return x, skip


class DecoderBlock(nn.Module):
    """
    Decoder block: Upsample -> Concatenate with skip -> ConvBlock
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        use_batch_norm: bool = True,
        dropout: float = 0.0
    ):
        """
        Initialize decoder block.
        
        Args:
            in_channels: Number of input channels (before concatenation)
            out_channels: Number of output channels
            use_batch_norm: Whether to use batch normalization
            dropout: Dropout probability
        """
        super().__init__()
        
        # Upsample using transposed convolution
        self.upsample = nn.ConvTranspose2d(
            in_channels, 
            in_channels // 2, 
            kernel_size=2, 
            stride=2
        )
        
        # Conv block takes upsampled + skip connection
        self.conv_block = ConvBlock(
            in_channels,  # in_channels//2 from upsample + in_channels//2 from skip
            out_channels,
            use_batch_norm,
            dropout
        )
    
    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input from previous decoder layer
            skip: Skip connection from encoder
            
        Returns:
            Decoded output
        """
        x = self.upsample(x)
        
        # Handle size mismatch due to odd-sized inputs
        if x.shape != skip.shape:
            x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
        
        # Concatenate with skip connection
        x = torch.cat([x, skip], dim=1)
        
        x = self.conv_block(x)
        
        return x


class UNet(nn.Module):
    """
    U-Net for sea-ice concentration forecasting.
    
    Architecture follows the classic U-Net design with encoder-decoder
    structure and skip connections.
    """
    
    def __init__(
        self,
        input_channels: int = 7,
        output_channels: int = 1,
        encoder_channels: List[int] = [32, 64, 128, 256],
        use_batch_norm: bool = True,
        dropout: float = 0.0,
        output_activation: str = 'sigmoid'
    ):
        """
        Initialize U-Net.
        
        Args:
            input_channels: Number of input timesteps/channels
            output_channels: Number of output channels (1 for single-day forecast)
            encoder_channels: List of channel sizes for encoder stages
            use_batch_norm: Whether to use batch normalization
            dropout: Dropout probability
            output_activation: Output activation ('sigmoid' or 'none')
        """
        super().__init__()
        
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.encoder_channels = encoder_channels
        self.output_activation_type = output_activation
        
        # Build encoder
        self.encoders = nn.ModuleList()
        in_ch = input_channels
        for out_ch in encoder_channels:
            self.encoders.append(
                EncoderBlock(in_ch, out_ch, use_batch_norm, dropout)
            )
            in_ch = out_ch
        
        # Bottleneck
        self.bottleneck = ConvBlock(
            encoder_channels[-1], 
            encoder_channels[-1] * 2,
            use_batch_norm,
            dropout
        )
        
        # Build decoder
        self.decoders = nn.ModuleList()
        decoder_channels = list(reversed(encoder_channels))
        in_ch = encoder_channels[-1] * 2
        
        for out_ch in decoder_channels:
            self.decoders.append(
                DecoderBlock(in_ch, out_ch, use_batch_norm, dropout)
            )
            in_ch = out_ch
        
        # Output layer
        self.output_conv = nn.Conv2d(
            encoder_channels[0],
            output_channels,
            kernel_size=1
        )
        
        # Output activation
        if output_activation == 'sigmoid':
            self.output_activation = nn.Sigmoid()
        else:
            self.output_activation = nn.Identity()
        
        # Initialize weights
        self._initialize_weights()
        
        # Log architecture
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info(f"U-Net initialized: {total_params:,} total params, "
                   f"{trainable_params:,} trainable")
    
    def _initialize_weights(self):
        """Initialize network weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor [Batch, input_channels, Height, Width]
            
        Returns:
            Output tensor [Batch, output_channels, Height, Width]
        """
        # Encoder path with skip connections
        skips = []
        for encoder in self.encoders:
            x, skip = encoder(x)
            skips.append(skip)
        
        # Bottleneck
        x = self.bottleneck(x)
        
        # Decoder path with skip connections
        for decoder, skip in zip(self.decoders, reversed(skips)):
            x = decoder(x, skip)
        
        # Output
        x = self.output_conv(x)
        x = self.output_activation(x)
        
        return x
    
    def get_num_parameters(self) -> dict:
        """Get number of parameters."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'total': total,
            'trainable': trainable,
            'non_trainable': total - trainable
        }
    
    def __repr__(self):
        params = self.get_num_parameters()
        return (f"UNet(\n"
                f"  input_channels={self.input_channels},\n"
                f"  output_channels={self.output_channels},\n"
                f"  encoder_channels={self.encoder_channels},\n"
                f"  parameters={params['total']:,}\n"
                f")")


def create_unet_from_config(config: dict) -> UNet:
    """
    Create U-Net model from configuration.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Initialized U-Net model
    """
    arch_config = config['architecture']
    
    # Get input channels (number of timesteps)
    input_channels = config['forecast']['input_window']
    
    # Get encoder channels (skip first element which is input size in config)
    encoder_channels = arch_config['encoder_channels'][1:]  # Skip input size
    
    model = UNet(
        input_channels=input_channels,
        output_channels=arch_config['output_shape'][0],  # Should be 1
        encoder_channels=encoder_channels,
        use_batch_norm=arch_config['use_batch_norm'],
        dropout=0.0,  # Can be added to config if needed
        output_activation=arch_config['output_activation']
    )
    
    return model


def test_unet():
    """Test U-Net architecture."""
    # Create model
    model = UNet(
        input_channels=7,
        output_channels=1,
        encoder_channels=[32, 64, 128, 256]
    )
    
    print(model)
    print()
    
    # Test forward pass
    batch_size = 4
    H, W = 316, 332  # Antarctic grid size
    
    x = torch.randn(batch_size, 7, H, W)
    
    print(f"Input shape: {x.shape}")
    
    with torch.no_grad():
        output = model(x)
    
    print(f"Output shape: {output.shape}")
    print(f"Output range: [{output.min():.3f}, {output.max():.3f}]")
    
    # Check output is in [0, 1] due to sigmoid
    assert output.min() >= 0 and output.max() <= 1, "Output not in [0, 1] range"
    print("\n✓ Output correctly bounded to [0, 1]")
    
    # Check parameters
    params = model.get_num_parameters()
    print(f"\nParameters: {params['total']:,}")


if __name__ == "__main__":
    test_unet()
