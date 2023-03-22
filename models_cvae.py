import torch
import torch.nn as nn
import torch.nn.functional as F


class CVAE(nn.Module):
    def __init__(self, in_channels=1, dim=64, out_conv_channels=512, hidden_dim=64):
        super(CVAE, self).__init__()
        self.encoder = Encoder(in_channels, dim, out_conv_channels, hidden_dim)

        self.decoder = Decoder(out_conv_channels, dim, in_channels, hidden_dim )

    def reparameterize(self, mu, log_var):
        std = torch.exp(log_var / 2)
        if self.training:
            eps = torch.randn_like(std)
            return mu + eps * std
        else:
            return mu

    def forward(self, x, x_human):
        mu, log_var = self.encoder(x, x_human)
        z = self.reparameterize(mu, log_var)
        x_rec = self.decoder(z, x_human)
        return x_rec, mu, log_var, z


class Encoder(torch.nn.Module):
    def __init__(self, in_channels=1, dim=64, out_conv_channels=512, hidden_dim=64):
        super(Encoder, self).__init__()
        conv1_channels = int(out_conv_channels / 8)
        conv2_channels = int(out_conv_channels / 4)
        conv3_channels = int(out_conv_channels / 2)
        self.out_conv_channels = out_conv_channels
        self.dim = dim
        self.out_dim = int(dim / 16)
        self.hidden_dim = hidden_dim

        self.conv1 = nn.Sequential(
            nn.Conv3d(
                in_channels=in_channels + 1, out_channels=conv1_channels, kernel_size=4,
                stride=2, padding=1, bias=False
            ),
            nn.BatchNorm3d(conv1_channels),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv3d(
                in_channels=conv1_channels + 1, out_channels=conv2_channels, kernel_size=4,
                stride=2, padding=1, bias=False
            ),
            nn.BatchNorm3d(conv2_channels),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.conv3 = nn.Sequential(
            nn.Conv3d(
                in_channels=conv2_channels + 1, out_channels=conv3_channels, kernel_size=4,
                stride=2, padding=1, bias=False
            ),
            nn.BatchNorm3d(conv3_channels),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.conv4 = nn.Sequential(
            nn.Conv3d(
                in_channels=conv3_channels + 1, out_channels=out_conv_channels, kernel_size=4,
                stride=2, padding=1, bias=False
            ),
            nn.BatchNorm3d(out_conv_channels),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.mean_out = nn.Sequential(
            nn.Linear(out_conv_channels * self.out_dim * self.out_dim * self.out_dim, self.hidden_dim),
            #nn.Sigmoid(),
        )
        self.var_out = nn.Sequential(
            nn.Linear(out_conv_channels * self.out_dim * self.out_dim * self.out_dim, self.hidden_dim),
            #nn.Sigmoid(),
        )

    def forward(self, x, x_human):
        x = torch.cat([x, x_human], dim=1)
        x = self.conv1(x)
        x = torch.cat([x, F.interpolate(x_human, (self.dim // 2, self.dim // 2, self.dim // 2))], dim=1)
        x = self.conv2(x)
        x = torch.cat([x, F.interpolate(x_human, (self.dim // 4, self.dim // 4, self.dim // 4))], dim=1)
        x = self.conv3(x)
        x = torch.cat([x, F.interpolate(x_human, (self.dim // 8, self.dim // 8, self.dim // 8))], dim=1)
        x = self.conv4(x)
        # Flatten and apply linear + sigmoid
        x = x.view(-1, self.out_conv_channels * self.out_dim * self.out_dim * self.out_dim)
        mean = self.mean_out(x)
        log_var = self.var_out(x)
        return mean, log_var


class Decoder(torch.nn.Module):
    def __init__(self, in_channels=512, out_dim=64, out_channels=1, noise_dim=64, activation="sigmoid"):
        super(Decoder, self).__init__()
        self.in_channels = in_channels
        self.out_dim = out_dim
        self.in_dim = int(out_dim / 16)
        conv1_out_channels = int(self.in_channels / 2.0)
        conv2_out_channels = int(conv1_out_channels / 2)
        conv3_out_channels = int(conv2_out_channels / 2)
        conv4_out_channels = int(conv3_out_channels / 2)

        self.linear = torch.nn.Linear(noise_dim, in_channels * self.in_dim * self.in_dim * self.in_dim)

        self.conv1 = nn.Sequential(
            nn.ConvTranspose3d(
                in_channels=in_channels, out_channels=conv1_out_channels, kernel_size=(4, 4, 4),
                stride=2, padding=1, bias=False
            ),
            nn.BatchNorm3d(conv1_out_channels),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.ConvTranspose3d(
                in_channels=conv1_out_channels + 1, out_channels=conv2_out_channels, kernel_size=(4, 4, 4),
                stride=2, padding=1, bias=False
            ),
            nn.BatchNorm3d(conv2_out_channels),
            nn.ReLU(inplace=True)
        )
        self.conv3 = nn.Sequential(
            nn.ConvTranspose3d(
                in_channels=conv2_out_channels + 1, out_channels=conv3_out_channels, kernel_size=(4, 4, 4),
                stride=2, padding=1, bias=False
            ),
            nn.BatchNorm3d(conv3_out_channels),
            nn.ReLU(inplace=True)
        )
        self.conv4 = nn.Sequential(
            nn.ConvTranspose3d(
                in_channels=conv3_out_channels + 1, out_channels=conv4_out_channels, kernel_size=(4, 4, 4),
                stride=2, padding=1, bias=False
            ),
            nn.BatchNorm3d(conv4_out_channels),
            nn.ReLU(inplace=True)
        )
        self.conv_out_1 = nn.Sequential(
            nn.Conv3d(
                in_channels=conv4_out_channels + 1, out_channels=conv4_out_channels, kernel_size=(1, 1, 1),
                stride=1, padding=0, bias=False
            ),
            nn.BatchNorm3d(conv4_out_channels),
            nn.ReLU(inplace=True)
        )
        self.conv_out_2= nn.Sequential(
            nn.Conv3d(
                in_channels=conv4_out_channels, out_channels=out_channels, kernel_size=(1, 1, 1),
                stride=1, padding=0, bias=False
            ),
        )
        if activation == "sigmoid":
            self.out = torch.nn.Sigmoid()
        else:
            self.out = torch.nn.Tanh()

    def project(self, x):
        """
        projects and reshapes latent vector to starting volume
        :param x: latent vector
        :return: starting volume
        """
        return x.view(-1, self.in_channels, self.in_dim, self.in_dim, self.in_dim)

    def forward(self, x, x_human):
        x = self.linear(x)
        x = self.project(x)
        # x = torch.cat([x, F.interpolate(x_human, (self.out_dim // 16, self.out_dim // 16, self.out_dim // 16))], dim=1)
        x = self.conv1(x)
        x = torch.cat([x, F.interpolate(x_human, (self.out_dim // 8, self.out_dim // 8, self.out_dim // 8))], dim=1)
        x = self.conv2(x)
        x = torch.cat([x, F.interpolate(x_human, (self.out_dim // 4, self.out_dim // 4, self.out_dim // 4))], dim=1)
        x = self.conv3(x)
        x = torch.cat([x, F.interpolate(x_human, (self.out_dim // 2, self.out_dim // 2, self.out_dim // 2))], dim=1)
        x = self.conv4(x)
        x = torch.cat([x, F.interpolate(x_human, (self.out_dim // 1, self.out_dim // 1, self.out_dim // 1))], dim=1)
        x = self.conv_out_1(x)
        x = self.conv_out_2(x)
        return self.out(x)
