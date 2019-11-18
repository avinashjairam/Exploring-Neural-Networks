from torch import nn


class MyConvNet(nn.Module):
    def __init__(self, num_classes, batch_norm=False, xavier=False):
        super(MyConvNet, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(in_channels=6, out_channels=12, kernel_size=5)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv2d(in_channels=12, out_channels=18, kernel_size=5)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv4 = nn.Conv2d(in_channels=18, out_channels=24, kernel_size=5)
        self.relu4 = nn.ReLU()
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        # self.conv1_bn = nn.BatchNorm2d(num_features=6)

        self.fc1 = nn.Linear(24 * 4 * 4, 256)
        self.reluf1 = nn.ReLU()
        self.fc2 = nn.Linear(256, 64)
        self.reluf2 = nn.ReLU()
        # self.fc1_bn = nn.BatchNorm1d(120)
        # self.fc2_bn = nn.BatchNorm1d(84)
        self.fc3 = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.pool3(self.relu3(self.conv3(x)))
        x = self.pool4(self.relu4(self.conv4(x)))
        x = x.view(-1, 24 * 4 * 4)
        x = self.reluf1(self.fc1(x))
        x = self.reluf2(self.fc2(x))
        x = self.fc3(x)
        return x


if __name__ == "__main__":
    from data_utils import MyDataset

    filepaths = ['data/frog/1.jpg', 'data/frog/2.jpg', 'data/deer/1.jpg']
    dataset = MyDataset(filepaths=filepaths)
    model = MyConvNet(num_classes=10)
    print(model(dataset[0][0].unsqueeze(0)))
