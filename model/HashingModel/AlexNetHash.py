import torch.nn as nn
from torchvision import models


class AlexNet(nn.Module):
    def __init__(self, hash_bit, alex_model="AlexNet", pretrained=True):
        super(AlexNet, self).__init__()
        # 将 AlexNet 模型的特征提取部分(卷积层和池化层)保存到 features 中
        model_alexnet = alexnet_dict[alex_model](pretrained=pretrained)
        self.features = model_alexnet.features
        # 创建两个全连接层 fc1,fc2 复制预训练模型的权重和偏置
        fc1 = nn.Linear(256 * 6 * 6, 4096)
        fc1.weight = model_alexnet.classifier[1].weight
        fc1.bias = model_alexnet.classifier[1].bias

        fc2 = nn.Linear(4096, 4096)
        fc2.weight = model_alexnet.classifier[4].weight
        fc2.bias = model_alexnet.classifier[4].bias

        self.hash_layer = nn.Sequential(
            # 随机失活,防止过拟合
            nn.Dropout(),
            fc1,
            # ReLU激活,直接修改tensor节约内存
            nn.ReLU(inplace=True),
            nn.Dropout(),
            fc2,
            nn.ReLU(inplace=True),
            # 全连接层,将张亮映射到bit位的哈希码输出
            nn.Linear(4096, hash_bit),
        )

    # 前向传播
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.hash_layer(x)
        return x


alexnet_dict = {"AlexNet": models.alexnet}
