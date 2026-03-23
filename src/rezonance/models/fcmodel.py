from torch import nn

class FCModel(nn.Module):
    def __init__(self, buffer_size: int, output_size: int):
        super(FCModel, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(buffer_size, 8192),
            nn.ReLU(),
            nn.Linear(8192, 4096),
            nn.ReLU(),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Linear(4096, 2048),
            nn.ReLU(),
            nn.Linear(2048, 2048),
            nn.ReLU(),
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, output_size),
        )

    def forward(self, X):
        return self.model(X)
