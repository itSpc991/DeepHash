import torch.nn as nn
from torchvision import models


class ResNet(nn.Module):
    def __init__(self, hash_bit, res_model="ResNet50", pretrained=True):
        super(ResNet, self).__init__()
        # 选择预训练的 resnet 模型
        model_resnet = resnet_dict[res_model](pretrained=pretrained)
        # 初始卷积层
        self.conv1 = model_resnet.conv1
        # BatchNorm 层
        self.bn1 = model_resnet.bn1
        # ReLU
        self.relu = model_resnet.relu
        # 最大池化层
        self.maxPool = model_resnet.maxPool
        # 四层残差块
        self.layer1 = model_resnet.layer1
        self.layer2 = model_resnet.layer2
        self.layer3 = model_resnet.layer3
        self.layer4 = model_resnet.layer4
        # 平均池化层
        self.avgPool = model_resnet.avgPool
        # 定义特征提取的网络结构
        self.feature_layers = nn.Sequential(self.conv1, self.bn1, self.relu, self.maxpool,
                                            self.layer1, self.layer2, self.layer3, self.layer4, self.avgpool)
        # 创建全连接层作为哈希码层,输入维度从预训练的 ResNet 模型中获取
        self.hash_layer = nn.Linear(model_resnet.fc.in_features, hash_bit)
        # 初始化全连接层的权重与偏置
        self.hash_layer.weight.data.normal_(0, 0.01)
        self.hash_layer.bias.data.fill_(0.0)

    # 前向传播
    def forward(self, x):
        x = self.feature_layers(x)
        # 将特征向量 x 展平为一维向量
        x = x.view(x.size(0), -1)
        y = self.hash_layer(x)
        return y


# ResNet50表示有50层卷积神经网络
resnet_dict = {"ResNet18": models.resnet18, "ResNet34": models.resnet34, "ResNet50": models.resnet50,
               "ResNet101": models.resnet101, "ResNet152": models.resnet152}
