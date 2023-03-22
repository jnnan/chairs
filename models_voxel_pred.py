import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet101


class VoxelPredNet(nn.Module):
    def __init__(self, hidden_size=128):
        super(VoxelPredNet, self).__init__()

        self.image_encoder = resnet101(pretrained=True, num_classes=1000)
        self.image_encoder.fc = nn.Linear(2048, hidden_size)
        self.voxel_decoder = Decoder(input_len=hidden_size)

    def forward(self, img, x_human):
        # img = torch.permute(img, (0, 3, 1, 2))
        image_emb = self.image_encoder(img)
        out = self.voxel_decoder(image_emb, x_human)
        return out


class Decoder(torch.nn.Module):
    def __init__(self, in_channels=512, out_dim=64, out_channels=1, input_len=128, activation="sigmoid"):
        super(Decoder, self).__init__()
        self.in_channels = in_channels
        self.out_dim = out_dim
        self.in_dim = int(out_dim / 16)
        conv1_out_channels = int(self.in_channels / 2.0)
        conv2_out_channels = int(conv1_out_channels / 2)
        conv3_out_channels = int(conv2_out_channels / 2)
        conv4_out_channels = int(conv3_out_channels / 2)

        self.linear = torch.nn.Linear(input_len, in_channels * self.in_dim * self.in_dim * self.in_dim)

        self.conv1 = nn.Sequential(
            nn.ConvTranspose3d(
                in_channels=in_channels, out_channels=conv1_out_channels, kernel_size=(4, 4, 4),
                stride=2, padding=1, bias=False
            ),
            nn.BatchNorm3d(conv1_out_channels),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.ConvTranspose3d(
                in_channels=conv1_out_channels + 1, out_channels=conv2_out_channels, kernel_size=(4, 4, 4),
                stride=2, padding=1, bias=False
            ),
            nn.BatchNorm3d(conv2_out_channels),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.conv3 = nn.Sequential(
            nn.ConvTranspose3d(
                in_channels=conv2_out_channels + 1, out_channels=conv3_out_channels, kernel_size=(4, 4, 4),
                stride=2, padding=1, bias=False
            ),
            nn.BatchNorm3d(conv3_out_channels),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.conv4 = nn.Sequential(
            nn.ConvTranspose3d(
                in_channels=conv3_out_channels + 1, out_channels=conv4_out_channels, kernel_size=(4, 4, 4),
                stride=2, padding=1, bias=False
            ),
            nn.BatchNorm3d(conv4_out_channels),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.conv_out_1 = nn.Sequential(
            nn.Conv3d(
                in_channels=conv4_out_channels + 1, out_channels=conv4_out_channels, kernel_size=(1, 1, 1),
                stride=1, padding=0, bias=False
            ),
            nn.BatchNorm3d(conv4_out_channels),
            nn.LeakyReLU(0.2, inplace=True)
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


class GPose(nn.Module):
    def __init__(self, input_size=63, hidden_size=128):
        super(GPose, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        # self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, hidden_size)
        self.rot_out = nn.Linear(hidden_size, 6)
        self.loc_out = nn.Linear(hidden_size, 3)
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, x):
        out = self.fc1(x)
        out = self.dropout(out)
        out = F.relu(out)
        out = self.fc2(out)
        out = self.dropout(out)
        out = F.relu(out)
        out = self.fc3(out)
        out = self.dropout(out)
        out = F.relu(out)
        out = self.fc4(out)
        out = self.dropout(out)
        out = F.relu(out)
        out_rot = self.rot_out(out)

        out_loc = self.loc_out(out)
        return out_rot, out_loc
