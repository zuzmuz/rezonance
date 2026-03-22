from torch import nn
from torch import Tensor

from rezonance.logger import logger


class ConvModel(nn.Module):
    def __init__(self, output_size: int):
        super(ConvModel, self).__init__()

        self.conv1 = self._block(
            1, 1024, kernel=512, stride=4, padding=254
        )
        self.conv2 = self._block(
            1024, 128, kernel=64, stride=1, padding=32
        )
        self.conv3 = self._block(
            128, 128, kernel=64, stride=1, padding=32
        )
        self.conv4 = self._block(
            128, 128, kernel=64, stride=1, padding=32
        )
        self.conv5 = self._block(
            128, 256, kernel=64, stride=1, padding=32
        )
        self.conv6 = self._block(
            256, 512, kernel=64, stride=1, padding=32
        )

        self.fc = nn.Linear(4 * 512, output_size)

    def _block(self, in_ch, out_ch, kernel, stride, padding):
        return nn.Sequential(
            nn.Conv1d(
                in_ch,
                out_ch,
                kernel_size=kernel,
                stride=stride,
                padding=padding,
            ),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.BatchNorm1d(out_ch),
            nn.Dropout(0.25),
        )

    def forward(self, x: Tensor):
        x = x.unsqueeze(1)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = x.flatten(1)
        return self.fc(x).squeeze(1)
