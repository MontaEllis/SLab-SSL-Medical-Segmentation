import torch
import torch.nn as nn


class FC3DDiscriminator(nn.Module):
    def __init__(self, num_classes, ndf=64, n_channel=1):
        super().__init__()
        self.conv0 = nn.Conv3d(num_classes, ndf, kernel_size=4, stride=2, padding=1)
        self.conv1 = nn.Conv3d(n_channel, ndf, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv3d(ndf, ndf * 2, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.Conv3d(ndf * 2, ndf * 4, kernel_size=4, stride=2, padding=1)
        self.conv4 = nn.Conv3d(ndf * 4, ndf * 8, kernel_size=4, stride=2, padding=1)
        self.avgpool = nn.AvgPool3d((6, 6, 6))
        self.classifier = nn.Linear(ndf * 8, 2)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.dropout = nn.Dropout3d(0.5)

    def forward(self, segmentation_map, image):
        batch_size = segmentation_map.shape[0]
        map_feature = self.conv0(segmentation_map)
        image_feature = self.conv1(image)
        features = torch.add(map_feature, image_feature)
        features = self.leaky_relu(features)
        features = self.dropout(features)

        features = self.conv2(features)
        features = self.leaky_relu(features)
        features = self.dropout(features)

        features = self.conv3(features)
        features = self.leaky_relu(features)
        features = self.dropout(features)

        features = self.conv4(features)
        features = self.leaky_relu(features)
        features = self.avgpool(features)
        features = features.view(batch_size, -1)
        logits = self.classifier(features)
        return logits.reshape((batch_size, 2))


class FCDiscriminator(nn.Module):
    def __init__(self, num_classes, ndf=64, n_channel=1):
        super().__init__()
        self.conv0 = nn.Conv2d(num_classes, ndf, kernel_size=4, stride=2, padding=1)
        self.conv1 = nn.Conv2d(n_channel, ndf, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(ndf, ndf * 2, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.Conv2d(ndf * 2, ndf * 4, kernel_size=4, stride=2, padding=1)
        self.conv4 = nn.Conv2d(ndf * 4, ndf * 8, kernel_size=4, stride=2, padding=1)
        self.classifier = nn.Linear(ndf * 32, 2)
        self.avgpool = nn.AvgPool2d((7, 7))
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.dropout = nn.Dropout2d(0.5)

    def forward(self, segmentation_map, image):
        map_feature = self.conv0(segmentation_map)
        image_feature = self.conv1(image)
        features = torch.add(map_feature, image_feature)

        features = self.conv2(features)
        features = self.leaky_relu(features)
        features = self.dropout(features)

        features = self.conv3(features)
        features = self.leaky_relu(features)
        features = self.dropout(features)

        features = self.conv4(features)
        features = self.leaky_relu(features)
        features = self.avgpool(features)
        features = features.view(features.size(0), -1)
        return self.classifier(features)
