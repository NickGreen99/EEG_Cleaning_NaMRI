import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock1D(nn.Module):
    def __init__(self, channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv1 = nn.Conv1d(
            in_channels=channels,
            out_channels=channels,
            stride=1,
            kernel_size=kernel_size,
            padding=padding,
            bias=True
        )
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(
            in_channels=channels,
            out_channels=channels,
            stride=1,
            kernel_size=kernel_size,
            padding=padding,
            bias=True
        )

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        # Note: no activation here, since UNetRes’s ResBlock does: out = x + res
        return out + residual


class DeepDSP_UNetRes(nn.Module):
    """
    - 1D U-Net with 3 downsampling stages + a bottleneck + 3 upsampling stages.
    - Each down/up stage contains nb=4 ResBlock1D modules.
    - Skip connections are added (not concatenated).
    - Head & tail use kernel_size=9, padding=4 to mirror the default `conv(..., kernel_size=9)` in UNetRes.
    - Default in_channels=2 (noisy_EEG + MRI_trace), out_channels=2 (predict MRI_noise channels), nb=4.
    """
    def __init__(self, in_channels=2, out_channels=1, nb: int = 4):
        super().__init__()
        self.nb = nb

        # ─── HEAD ───
        #(in_channels → 32) with kernel_size=9, padding=4.
        self.head = nn.Conv1d(in_channels, 32, kernel_size=9, padding=4, bias=True)

        # ─── ENCODER STAGE 1 ───
        # 4 × ResBlock1D(32 → 32)
        self.down1_res = nn.Sequential(
            *[ResBlock1D(32) for _ in range(self.nb)]
        )
        # Then a stride-2 Conv1d(32 → 64). (No BatchNorm; add ReLU in forward.)
        self.down1_conv = nn.Conv1d(32, 64, kernel_size=2, stride=2, bias=True)

        # ─── ENCODER STAGE 2 ───
        self.down2_res = nn.Sequential(
            *[ResBlock1D(64) for _ in range(self.nb)]
        )
        self.down2_conv = nn.Conv1d(64, 128, kernel_size=2, stride=2, bias=True)

        # ─── ENCODER STAGE 3 ───
        self.down3_res = nn.Sequential(
            *[ResBlock1D(128) for _ in range(self.nb)]
        )
        self.down3_conv = nn.Conv1d(128, 256, kernel_size=2, stride=2, bias=True)

        # ─── BOTTLENECK ───
        # 4 × ResBlock1D(256 → 256)
        self.body = nn.Sequential(
            *[ResBlock1D(256) for _ in range(self.nb)]
        )

        # ─── DECODER STAGE 1 (upsample from 256 → 128) ───
        # ConvTranspose1d(256 → 128, stride=2), then 4 × ResBlock1D(128)
        self.up3_convtrans = nn.ConvTranspose1d(256, 128, kernel_size=2, stride=2, output_padding=1, bias=True)
        self.up3_res = nn.Sequential(
            *[ResBlock1D(128) for _ in range(self.nb)]
        )

        # ─── DECODER STAGE 2 (upsample from 128 → 64) ───
        self.up2_convtrans = nn.ConvTranspose1d(128, 64, kernel_size=2, stride=2, bias=True)
        self.up2_res = nn.Sequential(
            *[ResBlock1D(64) for _ in range(self.nb)]
        )

        # ─── DECODER STAGE 3 (upsample from 64 → 32) ───
        self.up1_convtrans = nn.ConvTranspose1d(64, 32, kernel_size=2, stride=2, bias=True)
        self.up1_res = nn.Sequential(
            *[ResBlock1D(32) for _ in range(self.nb)]
        )

        # ─── TAIL ───
        # Conv1d(32 → out_channels) with kernel_size=9, padding=4
        self.tail = nn.Conv1d(32, out_channels, kernel_size=9, padding=4, bias=True)


    def forward(self, x):
        """
        x: tensor of shape [B, in_channels, T]
        Returns: tensor of shape [B, out_channels, T]
        """
        # ─── ENCODER ───
        # HEAD
        x1 = self.head(x)           # → [B, 32, T]
        x1 = F.relu(x1)              # Activation after head

        # stage 1
        d1 = self.down1_res(x1)      # → [B, 32, T]
        d1 = F.relu(d1)
        x2 = self.down1_conv(d1)     # → [B, 64, T/2]
        x2 = F.relu(x2)

        # stage 2
        d2 = self.down2_res(x2)      # → [B, 64, T/2]
        d2 = F.relu(d2)
        x3 = self.down2_conv(d2)     # → [B, 128, T/4]
        x3 = F.relu(x3)

        # stage 3
        d3 = self.down3_res(x3)      # → [B, 128, T/4]
        d3 = F.relu(d3)
        x4 = self.down3_conv(d3)     # → [B, 256, T/8]
        x4 = F.relu(x4)

        # ─── BOTTLENECK ───
        b = self.body(x4)            # → [B, 256, T/8]
        b = F.relu(b)

        # ─── DECODER ───
        # up + skip-add from x4 (also 256 channels)
        u3 = self.up3_convtrans(b + x4)  # → [B, 128, T/4]
        u3 = F.relu(u3)
        u3 = self.up3_res(u3)            # → [B, 128, T/4]

        # up + skip-add from x3 (128 channels)
        u2 = self.up2_convtrans(u3 + x3) # → [B, 64, T/2]
        u2 = F.relu(u2)
        u2 = self.up2_res(u2)            # → [B, 64, T/2]

        # up + skip-add from x2 (64 channels)
        u1 = self.up1_convtrans(u2 + x2) # → [B, 32, T]
        u1 = F.relu(u1)
        u1 = self.up1_res(u1)            # → [B, 32, T]

        # TAIL + skip-add from x1 (32 channels)
        out = self.tail(u1 + x1)         # → [B, out_channels, T]
        return out
