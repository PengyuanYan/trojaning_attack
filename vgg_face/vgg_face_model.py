import torch
import torch.nn as nn

"""
Defines the VGG-Face model architecture.

The architecture follows the model specification reported in Table 3 of:

    Parkhi, O. M., Vedaldi, A., & Zisserman, A. (2015).
    Deep Face Recognition.
    British Machine Vision Conference (BMVC).
    https://www.robots.ox.ac.uk/~vgg/publications/2015/Parkhi15/parkhi15.pdf

This implementation was informed by the following open-source PyTorch
implementations:

    https://github.com/ProgramComputer/VGGFace-pytorch/blob/main/vgg_face_dag.py
    https://github.com/prlz77/vgg-face.pytorch/blob/master/models/vgg_face.py
"""
class VGGFace(nn.Module):
    def __init__(self):
        super().__init__()
        
        # ---- Model meta data ----
        self.meta = {'mean': [129.186279296875, 104.76238250732422, 93.59396362304688],
                     'std': [1, 1, 1],
                     'image_size': [224, 224, 3],
                     'block_size': [2, 2, 3, 3, 3]}

        # ---- Block 1: 2 conv layers, 64 channels ----
        self.conv_1_1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv_1_2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)

        # ---- Block 2: 2 conv layers, 128 channels ----
        self.conv_2_1 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.conv_2_2 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1)

        # ---- Block 3: 3 conv layers, 256 channels ----
        self.conv_3_1 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.conv_3_2 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.conv_3_3 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1)

        # ---- Block 4: 3 conv layers, 512 channels ----
        self.conv_4_1 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.conv_4_2 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.conv_4_3 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1)

        # ---- Block 5: 3 conv layers, 512 channels ----
        self.conv_5_1 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.conv_5_2 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.conv_5_3 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1)

        # ---- Fully connected layers ----
        self.fc6 = nn.Linear(512 * 7 * 7, 4096)
        self.fc7 = nn.Linear(4096, 4096)
        # 2622 is the number of output class
        self.fc8 = nn.Linear(4096, 2622)

        # Shared operations
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x0):
        # Block 1
        x1 = self.relu(self.conv_1_1(x0))
        x1 = self.relu(self.conv_1_2(x1))
        x1 = self.pool(x1)

        # Block 2
        x2 = self.relu(self.conv_2_1(x1))
        x2 = self.relu(self.conv_2_2(x2))
        x2 = self.pool(x2)

        # Block 3
        x3 = self.relu(self.conv_3_1(x2))
        x3 = self.relu(self.conv_3_2(x3))
        x3 = self.relu(self.conv_3_3(x3))
        x3 = self.pool(x3)

        # Block 4
        x4 = self.relu(self.conv_4_1(x3))
        x4 = self.relu(self.conv_4_2(x4))
        x4 = self.relu(self.conv_4_3(x4))
        x4 = self.pool(x4)

        # Block 5
        x5 = self.relu(self.conv_5_1(x4))
        x5 = self.relu(self.conv_5_2(x5))
        x5 = self.relu(self.conv_5_3(x5))
        x5 = self.pool(x5)

        # Flatten: (batch, 512, 7, 7) -> (batch, 25088)
        xf = x5.view(x5.size(0), -1)

        # FC layers
        xf = self.dropout(self.relu(self.fc6(xf)))
        xf = self.dropout(self.relu(self.fc7(xf)))
        xf = self.fc8(xf)

        return xf