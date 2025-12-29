"""
模型定义：ResNet18, VGG11, MobileNetV2, AlexNet
支持模型分割和协同推理
"""
import torch
import torch.nn as nn
import torchvision.models as models


class SplitModel(nn.Module):
    """可分割的模型基类"""
    
    def __init__(self, model, num_classes=101):
        super(SplitModel, self).__init__()
        self.model = model
        self.num_classes = num_classes
        self.split_points = []  # 可分割点列表
        
    def get_split_points(self):
        """获取所有可分割点"""
        return self.split_points
    
    def forward_to_split(self, x, split_point):
        """前向传播到分割点"""
        raise NotImplementedError
    
    def forward_from_split(self, x, split_point):
        """从分割点继续前向传播"""
        raise NotImplementedError
    
    def forward(self, x):
        """完整的前向传播"""
        return self.model(x)


class SplitResNet18(SplitModel):
    """可分割的ResNet18"""
    
    def __init__(self, num_classes=101, pretrained=True):
        # 加载预训练模型
        base_model = models.resnet18(pretrained=pretrained)
        
        # 修改最后的全连接层
        base_model.fc = nn.Linear(base_model.fc.in_features, num_classes)
        
        super(SplitResNet18, self).__init__(base_model, num_classes)
        
        # 定义分割点：0-6
        # 0: 输入层之前（全云端）
        # 1: conv1之后
        # 2: layer1之后
        # 3: layer2之后
        # 4: layer3之后
        # 5: layer4之后
        # 6: avgpool之后（全边缘）
        self.split_points = list(range(7))
        
    def forward_to_split(self, x, split_point):
        """前向传播到分割点"""
        if split_point == 0:
            return x
        
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        
        if split_point == 1:
            return x
        
        x = self.model.layer1(x)
        if split_point == 2:
            return x
        
        x = self.model.layer2(x)
        if split_point == 3:
            return x
        
        x = self.model.layer3(x)
        if split_point == 4:
            return x
        
        x = self.model.layer4(x)
        if split_point == 5:
            return x
        
        x = self.model.avgpool(x)
        x = torch.flatten(x, 1)
        if split_point == 6:
            return x
        
        return x
    
    def forward_from_split(self, x, split_point):
        """从分割点继续前向传播"""
        if split_point == 0:
            return self.forward(x)
        
        if split_point == 1:
            x = self.model.layer1(x)
            x = self.model.layer2(x)
            x = self.model.layer3(x)
            x = self.model.layer4(x)
            x = self.model.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.model.fc(x)
            return x
        
        if split_point == 2:
            x = self.model.layer2(x)
            x = self.model.layer3(x)
            x = self.model.layer4(x)
            x = self.model.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.model.fc(x)
            return x
        
        if split_point == 3:
            x = self.model.layer3(x)
            x = self.model.layer4(x)
            x = self.model.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.model.fc(x)
            return x
        
        if split_point == 4:
            x = self.model.layer4(x)
            x = self.model.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.model.fc(x)
            return x
        
        if split_point == 5:
            x = self.model.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.model.fc(x)
            return x
        
        if split_point == 6:
            x = self.model.fc(x)
            return x
        
        return x


class SplitVGG11(SplitModel):
    """可分割的VGG11"""
    
    def __init__(self, num_classes=101, pretrained=True):
        base_model = models.vgg11(pretrained=pretrained)
        
        # 修改分类器
        base_model.classifier[6] = nn.Linear(4096, num_classes)
        
        super(SplitVGG11, self).__init__(base_model, num_classes)
        
        # 定义分割点：0-6
        # 0: 输入层之前
        # 1-5: features的不同阶段
        # 6: avgpool之后
        self.split_points = list(range(7))
        
    def forward_to_split(self, x, split_point):
        """前向传播到分割点"""
        if split_point == 0:
            return x
        
        # VGG11的features分为多个阶段
        features = self.model.features
        
        # 分割点1: 前2层
        if split_point == 1:
            for i in range(2):
                x = features[i](x)
            return x
        
        # 分割点2: 前5层
        if split_point == 2:
            for i in range(5):
                x = features[i](x)
            return x
        
        # 分割点3: 前10层
        if split_point == 3:
            for i in range(10):
                x = features[i](x)
            return x
        
        # 分割点4: 前15层
        if split_point == 4:
            for i in range(15):
                x = features[i](x)
            return x
        
        # 分割点5: 所有features层
        if split_point >= 5:
            x = features(x)
            x = self.model.avgpool(x)
            x = torch.flatten(x, 1)
            if split_point == 6:
                return x
        
        return x
    
    def forward_from_split(self, x, split_point):
        """从分割点继续前向传播"""
        if split_point == 0:
            return self.forward(x)
        
        features = self.model.features
        
        if split_point == 1:
            for i in range(2, len(features)):
                x = features[i](x)
            x = self.model.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.model.classifier(x)
            return x
        
        if split_point == 2:
            for i in range(5, len(features)):
                x = features[i](x)
            x = self.model.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.model.classifier(x)
            return x
        
        if split_point == 3:
            for i in range(10, len(features)):
                x = features[i](x)
            x = self.model.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.model.classifier(x)
            return x
        
        if split_point == 4:
            for i in range(15, len(features)):
                x = features[i](x)
            x = self.model.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.model.classifier(x)
            return x
        
        if split_point >= 5:
            x = self.model.classifier(x)
            return x
        
        return x


class SplitMobileNetV2(SplitModel):
    """可分割的MobileNetV2"""
    
    def __init__(self, num_classes=101, pretrained=True):
        base_model = models.mobilenet_v2(pretrained=pretrained)
        
        # 修改分类器
        base_model.classifier[1] = nn.Linear(base_model.last_channel, num_classes)
        
        super(SplitMobileNetV2, self).__init__(base_model, num_classes)
        
        # 定义分割点
        self.split_points = list(range(7))
        
    def forward_to_split(self, x, split_point):
        """前向传播到分割点"""
        if split_point == 0:
            return x
        
        features = self.model.features
        
        # 分割点1: 前3层
        if split_point == 1:
            for i in range(3):
                x = features[i](x)
            return x
        
        # 分割点2: 前6层
        if split_point == 2:
            for i in range(6):
                x = features[i](x)
            return x
        
        # 分割点3: 前9层
        if split_point == 3:
            for i in range(9):
                x = features[i](x)
            return x
        
        # 分割点4: 前12层
        if split_point == 4:
            for i in range(12):
                x = features[i](x)
            return x
        
        # 分割点5: 前15层
        if split_point == 5:
            for i in range(15):
                x = features[i](x)
            return x
        
        # 分割点6: 所有features层
        if split_point >= 6:
            x = features(x)
            x = nn.functional.adaptive_avg_pool2d(x, (1, 1))
            x = torch.flatten(x, 1)
            return x
        
        return x
    
    def forward_from_split(self, x, split_point):
        """从分割点继续前向传播"""
        if split_point == 0:
            return self.forward(x)
        
        features = self.model.features
        
        if split_point == 1:
            for i in range(3, len(features)):
                x = features[i](x)
            x = nn.functional.adaptive_avg_pool2d(x, (1, 1))
            x = torch.flatten(x, 1)
            x = self.model.classifier(x)
            return x
        
        if split_point == 2:
            for i in range(6, len(features)):
                x = features[i](x)
            x = nn.functional.adaptive_avg_pool2d(x, (1, 1))
            x = torch.flatten(x, 1)
            x = self.model.classifier(x)
            return x
        
        if split_point == 3:
            for i in range(9, len(features)):
                x = features[i](x)
            x = nn.functional.adaptive_avg_pool2d(x, (1, 1))
            x = torch.flatten(x, 1)
            x = self.model.classifier(x)
            return x
        
        if split_point == 4:
            for i in range(12, len(features)):
                x = features[i](x)
            x = nn.functional.adaptive_avg_pool2d(x, (1, 1))
            x = torch.flatten(x, 1)
            x = self.model.classifier(x)
            return x
        
        if split_point == 5:
            for i in range(15, len(features)):
                x = features[i](x)
            x = nn.functional.adaptive_avg_pool2d(x, (1, 1))
            x = torch.flatten(x, 1)
            x = self.model.classifier(x)
            return x
        
        if split_point >= 6:
            x = self.model.classifier(x)
            return x
        
        return x


class SplitAlexNet(SplitModel):
    """可分割的AlexNet"""
    
    def __init__(self, num_classes=101, pretrained=True):
        base_model = models.alexnet(pretrained=pretrained)
        
        # 修改分类器
        base_model.classifier[6] = nn.Linear(4096, num_classes)
        
        super(SplitAlexNet, self).__init__(base_model, num_classes)
        
        # 定义分割点
        self.split_points = list(range(6))
        
    def forward_to_split(self, x, split_point):
        """前向传播到分割点"""
        if split_point == 0:
            return x
        
        features = self.model.features
        
        # 分割点1-4: features的不同阶段
        if split_point == 1:
            for i in range(3):
                x = features[i](x)
            return x
        
        if split_point == 2:
            for i in range(6):
                x = features[i](x)
            return x
        
        if split_point == 3:
            for i in range(8):
                x = features[i](x)
            return x
        
        if split_point == 4:
            for i in range(10):
                x = features[i](x)
            return x
        
        # 分割点5: 所有features层
        if split_point >= 5:
            x = features(x)
            x = self.model.avgpool(x)
            x = torch.flatten(x, 1)
            return x
        
        return x
    
    def forward_from_split(self, x, split_point):
        """从分割点继续前向传播"""
        if split_point == 0:
            return self.forward(x)
        
        features = self.model.features
        
        if split_point == 1:
            for i in range(3, len(features)):
                x = features[i](x)
            x = self.model.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.model.classifier(x)
            return x
        
        if split_point == 2:
            for i in range(6, len(features)):
                x = features[i](x)
            x = self.model.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.model.classifier(x)
            return x
        
        if split_point == 3:
            for i in range(8, len(features)):
                x = features[i](x)
            x = self.model.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.model.classifier(x)
            return x
        
        if split_point == 4:
            for i in range(10, len(features)):
                x = features[i](x)
            x = self.model.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.model.classifier(x)
            return x
        
        if split_point >= 5:
            x = self.model.classifier(x)
            return x
        
        return x


def get_model(model_name, num_classes=101, pretrained=True):
    """
    获取指定的模型
    
    Args:
        model_name: 模型名称 ('resnet18', 'vgg11', 'mobilenetv2', 'alexnet')
        num_classes: 类别数量
        pretrained: 是否使用预训练权重
    
    Returns:
        模型实例
    """
    model_dict = {
        'resnet18': SplitResNet18,
        'vgg11': SplitVGG11,
        'mobilenetv2': SplitMobileNetV2,
        'alexnet': SplitAlexNet
    }
    
    if model_name.lower() not in model_dict:
        raise ValueError(f"不支持的模型: {model_name}")
    
    return model_dict[model_name.lower()](num_classes=num_classes, pretrained=pretrained)


if __name__ == '__main__':
    # 测试模型
    print("测试模型定义...")
    
    for model_name in ['resnet18', 'vgg11', 'mobilenetv2', 'alexnet']:
        print(f"\n测试 {model_name}:")
        model = get_model(model_name, num_classes=101, pretrained=False)
        
        # 测试完整前向传播
        x = torch.randn(2, 3, 224, 224)
        output = model(x)
        print(f"  完整输出形状: {output.shape}")
        
        # 测试分割推理
        split_points = model.get_split_points()
        print(f"  分割点: {split_points}")
        
        for sp in [1, 3, len(split_points)-1]:
            if sp < len(split_points):
                edge_output = model.forward_to_split(x, sp)
                cloud_output = model.forward_from_split(edge_output, sp)
                print(f"  分割点{sp} - 中间输出: {edge_output.shape}, 最终输出: {cloud_output.shape}")

