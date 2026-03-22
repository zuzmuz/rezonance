from torch import nn


class SmallTestModel(nn.Module):
    def __init__(self, buffer_size: int, output_size: int):
        super(SmallTestModel, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(buffer_size, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, output_size),
        )

    def forward(self, X):
        return self.model(X)
