import torch
import torch.nn as nn


class ConvLSTMCell(nn.Module):
    """
    一个简单的 ConvLSTM Cell 实现。
    输入: x_t: (B, C_in, H, W), h_{t-1}, c_{t-1}: (B, C_hidden, H, W)
    输出: h_t, c_t
    """

    def __init__(self, input_dim, hidden_dim, kernel_size, bias=True):
        super(ConvLSTMCell, self).__init__()

        padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.conv = nn.Conv2d(
            in_channels=input_dim + hidden_dim,
            out_channels=4 * hidden_dim,
            kernel_size=kernel_size,
            padding=padding,
            bias=bias,
        )

    def forward(self, x, state):
        h_prev, c_prev = state

        combined = torch.cat([x, h_prev], dim=1)  # (B, C_in + C_hidden, H, W)
        conv_out = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.chunk(conv_out, 4, dim=1)

        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c = f * c_prev + i * g
        h = o * torch.tanh(c)

        return h, c

    def init_state(self, batch_size, spatial_size, device=None, dtype=None):
        H, W = spatial_size
        kwargs = {}
        if device is not None:
            kwargs["device"] = device
        if dtype is not None:
            kwargs["dtype"] = dtype
        h = torch.zeros(batch_size, self.hidden_dim, H, W, **kwargs)
        c = torch.zeros(batch_size, self.hidden_dim, H, W, **kwargs)
        return h, c


class ConvLSTM(nn.Module):
    """
    单层 ConvLSTM，用于处理整段序列。
    输入: x_seq: (B, T, C_in, H, W)
    输出:
        h_seq: (B, T, C_hidden, H, W) - 每个时间步的隐藏状态
        (h_T, c_T): 最后一个时间步的隐藏/记忆状态
    """

    def __init__(self, input_dim, hidden_dim, kernel_size=(3, 3), bias=True):
        super(ConvLSTM, self).__init__()
        self.cell = ConvLSTMCell(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            kernel_size=kernel_size,
            bias=bias,
        )

    def forward(self, x_seq):
        # x_seq: (B, T, C_in, H, W)
        B, T, C, H, W = x_seq.shape
        device = x_seq.device
        dtype = x_seq.dtype

        h, c = self.cell.init_state(
            batch_size=B,
            spatial_size=(H, W),
            device=device,
            dtype=dtype,
        )

        h_list = []
        for t in range(T):
            x_t = x_seq[:, t]  # (B, C_in, H, W)
            h, c = self.cell(x_t, (h, c))
            h_list.append(h)

        h_seq = torch.stack(h_list, dim=1)  # (B, T, C_hidden, H, W)
        return h_seq, (h, c)
