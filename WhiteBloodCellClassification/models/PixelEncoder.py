import torchvision.models as models
import torch
import torch.nn as nn


class PixelEncoder(torch.nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()

        # Sửa warning: 'pretrained' bị deprecated từ torchvision 0.13.
        # Dùng weights enum mới: ResNet50_Weights.IMAGENET1K_V1 thay vì pretrained=True,
        # None thay vì pretrained=False.
        if pretrained:
            weights = models.ResNet50_Weights.IMAGENET1K_V1
        else:
            weights = None

        resnet = models.resnet50(weights=weights)

        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool

        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

        # Đóng băng BatchNorm để ổn định huấn luyện khi batch size nhỏ
        # (running_mean/var đã học từ ImageNet được giữ nguyên)
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
                m.requires_grad_(False)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        c1 = self.layer1(x)
        c2 = self.layer2(c1)
        c3 = self.layer3(c2)
        c4 = self.layer4(c3)
        return [c1, c2, c3, c4]



# import torchvision.models as models
# import torch
# import torch.nn as nn

# class PixelEncoder(torch.nn.Module):
#     def __init__(self, pretrained=True):
#         super().__init__()
#         resnet = models.resnet50(pretrained=pretrained)
        
#         self.conv1 = resnet.conv1
#         self.bn1 = resnet.bn1
#         self.relu = resnet.relu
#         self.maxpool = resnet.maxpool

#         self.layer1 = resnet.layer1
#         self.layer2 = resnet.layer2
#         self.layer3 = resnet.layer3
#         self.layer4 = resnet.layer4

#         for m in self.modules():
#             if isinstance(m, nn.BatchNorm2d):
#                 m.eval()
#                 m.requires_grad_(False)
    
#     def forward(self, x):
#         x = self.conv1(x)
#         x = self.bn1(x)
#         x = self.relu(x)
#         x = self.maxpool(x)
#         c1 = self.layer1(x)
#         c2 = self.layer2(c1)
#         c3 = self.layer3(c2)
#         c4 = self.layer4(c3)
#         return [c1, c2, c3, c4]
    

