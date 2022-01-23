# Spider generator using WGAN
# https://arxiv.org/pdf/1701.07875

import torch
import torch.nn as nn


def weights_init(model):
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d)):
            nn.init.normal_(m.weight.data, 0.0, 0.02)


class Critic(nn.Module):
    # features_d = channels changing through layer, 4->8->16->32->64
    def __init__(self, channels_img, features_d):
        super(Critic, self).__init__()
        self.crit = nn.Sequential(
            # Input: N * channels_img * 64 *64
            nn.Conv2d(
                channels_img, features_d, kernel_size=4, stride=2, padding=1
            ),  # 32*32
            nn.LeakyReLU(0.2),
            self._block(
                features_d, features_d * 2, kernel_size=4, stride=2, padding=1
            ),  # 16*16
            self._block(
                features_d * 2, features_d * 4, kernel_size=4, stride=2, padding=1
            ),  # 8*8
            self._block(
                features_d * 4, features_d * 8, kernel_size=4, stride=2, padding=1
            ),  # 4*4
            nn.Conv2d(features_d * 8, 1, kernel_size=4, stride=2, padding=0),  # 1*1
        )

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                bias=False,
            ),
            nn.InstanceNorm2d(out_channels, affine=True),
            nn.LeakyReLU(0.2),
        )

    def forward(self, x):
        return self.crit(x)


class Generator(nn.Module):
    def __init__(self, z_dim, channels_img, dimen):
        super(Generator, self).__init__()
        self.gen = nn.Sequential(
            # Input: N * z_dim *1 *1
            self._block(z_dim, dimen * 16, kernel_size=4, stride=1, padding=0),  # 1024
            self._block(
                dimen * 16, dimen * 8, kernel_size=4, stride=2, padding=1
            ),  # 512 \ 8x8
            self._block(
                dimen * 8, dimen * 4, kernel_size=4, stride=2, padding=1
            ),  # 256 \ 16x16
            self._block(
                dimen * 4, dimen * 2, kernel_size=4, stride=2, padding=1
            ),  # 128 \ 32x32
            nn.ConvTranspose2d(
                dimen * 2,
                channels_img,
                kernel_size=4,
                stride=2,
                padding=1,
            ),  # 64x64
            nn.Tanh(),  # [-1.1]
        )

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.ConvTranspose2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.gen(x)


def test():
    N, in_channels, H, W = 8, 3, 64, 64
    z_dim = 100
    x = torch.randn((N, in_channels, H, W))
    crit = Critic(in_channels, 8)
    weights_init(crit)
    # print(crit(x).shape)
    assert crit(x).shape == (N, 1, 1, 1), "Critic test failed"
    gen = Generator(z_dim, in_channels, 8)
    z = torch.randn((N, z_dim, 1, 1))
    assert gen(z).shape == (N, in_channels, H, W), "Generator test failed"


test()
