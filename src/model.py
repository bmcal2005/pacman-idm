"""
Inverse dynamics model (IDM) for 16-frame Atari clips.

    clip (B, 16, 84, 84) uint8
      -> 3D conv            mixes each frame with its neighbours: sees motion
      -> 2D CNN per frame   84x84 image -> 512 numbers per frame
      -> + position         tells the transformer which frame is which
      -> transformer        every frame looks at every other frame, past AND future
      -> token 8            the features of the labeled (middle) frame
      -> head               9 scores, one per action

Same architecture as chiaravongerlach/vpt-idm-clean, written to be read.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    """Adds a fixed sin/cos pattern per position, so frame #3 and frame #12 look different."""

    def __init__(self, dim, max_len=16):
        super().__init__()
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))
        pe = torch.zeros(1, max_len, dim)
        pe[0, :, 0::2] = torch.sin(pos * div)
        pe[0, :, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe)           # saved with the model, but not trained

    def forward(self, x):
        return x + self.pe[:, : x.size(1)]


class IDMBackbone(nn.Module):
    def __init__(self, seq=16, img=84, dim=512, layers=2, heads=4, dropout=0.1):
        super().__init__()
        self.seq, self.img, self.dim = seq, img, dim
        self.temporal = nn.Conv3d(1, 32, kernel_size=3, padding=1)
        chans = [32, 64, 128, 256, dim]
        self.convs = nn.ModuleList(nn.Conv2d(a, b, 3, stride=2, padding=1) for a, b in zip(chans, chans[1:]))
        self.norms = nn.ModuleList(nn.BatchNorm2d(b) for b in chans[1:])
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.pos = PositionalEncoding(dim, seq)
        layer = nn.TransformerEncoderLayer(dim, heads, dim * 4, dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(layer, layers)

    def forward(self, x, trace=False):
        show = (lambda name, t: print(f"{name:42s} {tuple(t.shape)}")) if trace else (lambda *a: None)
        B = x.size(0)
        show("input (uint8)", x)
        x = x.float().div(255).unsqueeze(1)                      # add a channel dim
        show("scale to 0-1, add channel", x)
        x = F.relu(self.temporal(x))
        show("3D conv: each frame + neighbours", x)
        x = x.permute(0, 2, 1, 3, 4).reshape(B * self.seq, 32, self.img, self.img)
        show("fold 16 frames into the batch", x)
        for i, (conv, bn) in enumerate(zip(self.convs, self.norms), 1):
            x = F.relu(bn(conv(x)))
            show(f"2D conv {i} (stride 2 halves the size)", x)
        x = self.pool(x).flatten(1).view(B, self.seq, self.dim)
        show("average-pool -> one vector per frame", x)
        x = self.transformer(self.pos(x))
        show("+ position, transformer (no mask)", x)
        x = x[:, self.seq // 2]
        show("take token 8 (the labeled frame)", x)
        return x


def _head(dim, n_out, dropout=0.1):
    return nn.Sequential(nn.Linear(dim, 128), nn.ReLU(), nn.Dropout(dropout), nn.Linear(128, n_out))


def _init_bias_to_priors(head, priors):
    """Start the head predicting the class frequencies instead of uniform (see Phase 6)."""
    with torch.no_grad():
        head[-1].bias.copy_(torch.log(torch.as_tensor(priors, dtype=torch.float).clamp(min=1e-6)))


class IDM9Way(nn.Module):
    """One head: 9 scores, one per action."""

    def __init__(self, priors=None, **kw):
        super().__init__()
        self.backbone = IDMBackbone(**kw)
        self.head = _head(self.backbone.dim, 9)
        if priors is not None:
            _init_bias_to_priors(self.head, priors)

    def forward(self, x, trace=False):
        out = self.head(self.backbone(x, trace))
        if trace:
            print(f"{'head -> 9 action scores (logits)':42s} {tuple(out.shape)}")
        return out


class IDMFactorized(nn.Module):
    """Two heads: horizontal (LEFT/NONE/RIGHT) and vertical (UP/NONE/DOWN). Used in Phase 6."""

    def __init__(self, h_priors=None, v_priors=None, **kw):
        super().__init__()
        self.backbone = IDMBackbone(**kw)
        self.hhead = _head(self.backbone.dim, 3)
        self.vhead = _head(self.backbone.dim, 3)
        if h_priors is not None:
            _init_bias_to_priors(self.hhead, h_priors)
        if v_priors is not None:
            _init_bias_to_priors(self.vhead, v_priors)

    def forward(self, x, trace=False):
        f = self.backbone(x, trace)
        return self.hhead(f), self.vhead(f)

    @staticmethod
    def to_9(h_logits, v_logits):
        """Combine the two heads into a 9-way prediction: class = 3*v + h."""
        return 3 * v_logits.argmax(1) + h_logits.argmax(1)


def count_params(module):
    return sum(p.numel() for p in module.parameters() if p.requires_grad)
