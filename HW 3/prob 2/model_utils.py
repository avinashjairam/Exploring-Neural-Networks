from torch import nn


class TrafficSignsConvNet(nn.Module):
    def __init__(self, num_classes, batch_norm=False, dropout=0.5, xavier=False):
        super(TrafficSignsConvNet, self).__init__()

        # store batch_norm used to create model object
        self.batch_norm = batch_norm

        if xavier:
            print('initializing model params with xavier initialization')

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5)
        if xavier:
            nn.init.xavier_normal_(self.conv1.weight.data)
            nn.init.normal_(self.conv1.bias.data)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout1 = nn.Dropout2d(p=dropout)
        if batch_norm:
            self.batchnorm1 = nn.BatchNorm2d(num_features=6)

        self.conv2 = nn.Conv2d(in_channels=6, out_channels=12, kernel_size=5)
        if xavier:
            nn.init.xavier_normal_(self.conv2.weight.data)
            nn.init.normal_(self.conv2.bias.data)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout2 = nn.Dropout2d(p=dropout)
        if batch_norm:
            self.batchnorm2 = nn.BatchNorm2d(num_features=12)

        self.conv3 = nn.Conv2d(in_channels=12, out_channels=18, kernel_size=5)
        if xavier:
            nn.init.xavier_normal_(self.conv3.weight.data)
            nn.init.normal_(self.conv3.bias.data)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout3 = nn.Dropout2d(p=dropout)
        if batch_norm:
            self.batchnorm3 = nn.BatchNorm2d(num_features=18)

        self.conv4 = nn.Conv2d(in_channels=18, out_channels=24, kernel_size=5)
        if xavier:
            nn.init.xavier_normal_(self.conv4.weight.data)
            nn.init.normal_(self.conv4.bias.data)
        self.relu4 = nn.ReLU()
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout4 = nn.Dropout2d(p=dropout)
        if batch_norm:
            self.batchnorm4 = nn.BatchNorm2d(num_features=24)

        self.fc1 = nn.Linear(24 * 4 * 4, 256)
        if xavier:
            nn.init.xavier_normal_(self.fc1.weight.data)
            nn.init.normal_(self.fc1.bias.data)
        self.reluf1 = nn.ReLU()
        if batch_norm:
            self.batchnormfc1 = nn.BatchNorm1d(num_features=256)
        self.fc2 = nn.Linear(256, 64)
        if xavier:
            nn.init.xavier_normal_(self.fc2.weight.data)
            nn.init.normal_(self.fc2.bias.data)
        self.reluf2 = nn.ReLU()
        if batch_norm:
            self.batchnormfc2 = nn.BatchNorm1d(num_features=64)
        self.fc3 = nn.Linear(64, num_classes)
        if xavier:
            nn.init.xavier_normal_(self.fc3.weight.data)
            nn.init.normal_(self.fc3.bias.data)

    def forward(self, x):

        if self.batch_norm:
            x = self.batchnorm1(self.dropout1(self.pool1(self.relu1(self.conv1(x)))))
            x = self.batchnorm2(self.dropout2(self.pool2(self.relu2(self.conv2(x)))))
            x = self.batchnorm3(self.dropout3(self.pool3(self.relu3(self.conv3(x)))))
            x = self.batchnorm4(self.dropout4(self.pool4(self.relu4(self.conv4(x)))))
        else:
            x = self.dropout1(self.pool1(self.relu1(self.conv1(x))))
            x = self.dropout2(self.pool2(self.relu2(self.conv2(x))))
            x = self.dropout3(self.pool3(self.relu3(self.conv3(x))))
            x = self.dropout4(self.pool4(self.relu4(self.conv4(x))))

        # flatten
        x = x.view(-1, 24 * 4 * 4)

        if self.batch_norm:
            x = self.batchnormfc1(self.reluf1(self.fc1(x)))
            x = self.batchnormfc2(self.reluf2(self.fc2(x)))
        else:
            x = self.reluf1(self.fc1(x))
            x = self.reluf2(self.fc2(x))
        x = self.fc3(x)
        return x


if __name__ == "__main__":
    from data_utils import TrafficSignsDataset

    filepaths = [
        "data/gtsrb-german-traffic-sign/Train/1/00001_00000_00007.png",
        "data/gtsrb-german-traffic-sign/Train/1/00001_00000_00008.png",
        "data/gtsrb-german-traffic-sign/Train/1/00001_00000_00009.png",
        "data/gtsrb-german-traffic-sign/Train/1/00001_00000_00010.png",
        "data/gtsrb-german-traffic-sign/Train/17/00017_00000_00002.png",
        "data/gtsrb-german-traffic-sign/Train/1/00001_00000_00011.png"
    ]
    dataset = TrafficSignsDataset(filepaths=filepaths)
    model = TrafficSignsConvNet(num_classes=10, xavier=True, batch_norm=True)
    model.eval()
    print(model(dataset[0][0].unsqueeze(0)))
