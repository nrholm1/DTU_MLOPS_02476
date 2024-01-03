from torch import nn


class MyAwesomeModel(nn.Module):
    """My awesome model."""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=16,
                kernel_size=5,
                stride=1,
                padding=2,
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=16,
                out_channels=32,
                kernel_size=5,
                stride=1,
                padding=2,
            ),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.fc1 = nn.Linear(32 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.2)
        self.logsm = nn.LogSoftmax(dim=1)

        self.ffnn = nn.Sequential(self.fc1, self.relu, self.fc2, self.relu, self.fc3, self.logsm)

    def forward(self, x):
        conv_out_1 = self.conv1(x)
        conv_out_2 = self.conv2(conv_out_1)
        z = conv_out_2.view(x.size(0), -1)
        output = self.ffnn(z)
        return output, z
