import torch
import torch.nn as nn

from anti_spoof.base.base_model import BaseModel
from .sync_layer import SincConv_fast


LRELU_SLOPE = 0.1


class FMS(nn.Module):
    def __init__(self, n_filters):
        super().__init__()
        self.fc = nn.Linear(n_filters, n_filters)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        '''
        :param x: (batch_size, out_channels, n_samples)
        :return: (batch_size, out_channels, n_samples)
        '''
        averaged = torch.mean(x, dim=-1)
        r = self.sigmoid(self.fc(averaged))

        r = r.unsqueeze(-1)
        return x * r + r


class ResBlock(nn.Module):
    def __init__(self, n_filters_in, n_filters_out, start_with_bn_and_relu=True):
        super().__init__()

        self.start_with_bn_and_relu = start_with_bn_and_relu

        if self.start_with_bn_and_relu:
            self.bn1 = nn.BatchNorm1d(n_filters_in)
            self.lrelu1 = nn.LeakyReLU(LRELU_SLOPE)

        self.conv1 = nn.Conv1d(n_filters_in, n_filters_out, 3, padding=1)
        self.bn2 = nn.BatchNorm1d(n_filters_out)
        self.lrelu2 = nn.LeakyReLU(LRELU_SLOPE)
        self.conv2 = nn.Conv1d(n_filters_out, n_filters_out, 3, padding=1)

        self.downsample = nn.Conv1d(n_filters_in, n_filters_out, 1) if n_filters_in != n_filters_out else None

        self.maxpool = nn.MaxPool1d(3)
        self.fms = FMS(n_filters_out)

    def forward(self, x):
        out = x
        if self.start_with_bn_and_relu:
            out = self.lrelu1(self.bn1(x))

        out = self.conv1(out)
        out = self.bn2(out)
        out = self.lrelu2(out)
        out = self.conv2(out)

        if self.downsample is None:
            out = out + x
        else:
            out = out + self.downsample(x)

        out = self.maxpool(out)
        out = self.fms(out)
        return out


class RawNet(BaseModel):
    def __init__(self, n_sync_filters, sync_kernel_size, sync_min_low_hz, sync_min_band_hz,
                 n_filters_resblock1, n_filters_resblock2, gru_num_layers, gru_hid_size):
        super().__init__()

        self.sync_conv = SincConv_fast(n_sync_filters, sync_kernel_size,
                                       min_low_hz=sync_min_low_hz, min_band_hz=sync_min_band_hz)
        self.maxpool = nn.MaxPool1d(3)
        self.bn = nn.BatchNorm1d(n_sync_filters)
        self.lrelu = nn.LeakyReLU(LRELU_SLOPE)

        self.res_blocks1 = nn.Sequential(
            ResBlock(n_sync_filters, n_filters_resblock1, start_with_bn_and_relu=False),
            ResBlock(n_filters_resblock1, n_filters_resblock1)
        )

        self.res_blocks2 = nn.Sequential(
            *[
                ResBlock(n_filters_resblock1 if i == 0 else n_filters_resblock2,
                         n_filters_resblock2)
                for i in range(4)
            ]
        )

        self.pre_gru_bn = nn.BatchNorm1d(n_filters_resblock2)
        self.pre_gru_lrelu = nn.LeakyReLU(LRELU_SLOPE)
        self.gru = nn.GRU(n_filters_resblock2, gru_hid_size, gru_num_layers, batch_first=True)
        self.fc = nn.Linear(gru_hid_size, 1)

    def forward(self, audio, **batch):
        '''
        :param audio: Batch of waveforms. Shape: (batch_size, 1, n_samples)
        :return: pred scores of shape (batch_size,)
        '''
        x = self.sync_conv(audio)
        x = self.maxpool(torch.abs(x))
        x = self.bn(x)
        x = self.lrelu(x)

        x = self.res_blocks1(x)
        x = self.res_blocks2(x)

        x = self.pre_gru_bn(x)
        x = self.pre_gru_lrelu(x)
        x, _ = self.gru(x.transpose(-1, -2))  # B x time x hidden_size
        x = self.fc(x[:, -1])
        return {'scores': x.squeeze(-1)}
