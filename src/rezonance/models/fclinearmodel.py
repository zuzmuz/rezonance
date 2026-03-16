from torch import nn

class FCLinearModel(nn.Module):
    def __init__(self, buffer_size: int):
        super(FCLinearModel, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(buffer_size, 4096),
            nn.ReLU(),
            nn.Linear(4096, 2048),
            nn.ReLU(),
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(self, X):
        return self.model(X).squeeze(1)
