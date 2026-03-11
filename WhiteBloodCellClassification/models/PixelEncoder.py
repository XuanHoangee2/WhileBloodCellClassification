import torchvision.models as models
import torch

class PixelEncoder(torch.nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        resnet = models.resnet50(pretrained=pretrained)
        
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool

        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
    
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
    

# encoder = PixelEncoder()
# x = torch.randn(1, 3, 256, 256)
# feature = encoder(x)
# for f in feature:
#     print(f.shape)