import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo

__all__ = ['ResNet', 'resnet18']

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth'
}


# ------------------------------------------
# 基础卷积
# ------------------------------------------
def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(
        in_planes, out_planes, kernel_size=3, stride=stride,
        padding=1, bias=False
    )

def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(
        in_planes, out_planes, kernel_size=1, stride=stride, bias=False
    )


# ------------------------------------------
# ResNet 基本残差块
# ------------------------------------------
class BasicBlock(nn.Module):
    expansion = 1
    
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.relu(out + identity)
        return out


# ------------------------------------------
# ACMP-FL 专用 ResNet18（但不改名字）
# ------------------------------------------
class ResNet(nn.Module):
    """
    保持与原版架构一致，但新增 prototype head。
    forward 输出 (logits, proto_vector)
    """
    def __init__(self, args, block, layers, num_classes=1000, zero_init_residual=False):

        super().__init__()
        self.args = args
        self.inplanes = 64

        # ----------- Stem ----------
        self.conv1 = nn.Conv2d(
            3, 64, kernel_size=7, stride=args.stride[0], padding=3, bias=False
        )
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(
            kernel_size=3, stride=args.stride[1], padding=1
        )

        # ----------- ResNet layers ----------
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        # ----------- Pooling ----------
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # --------------------------------------------------
        # 🔥 新增 prototype head（原型 512 维）
        # --------------------------------------------------
        self.proto_head = nn.Linear(512, 512)

        # ----------- Final classifier ----------
        self.fc = nn.Linear(512, num_classes)

        # ----------- 初始化 ----------
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                        nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    # --------------------------------------------------------
    # 构建 layer
    # --------------------------------------------------------
    def _make_layer(self, block, planes, blocks, stride=1):

        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion)
            )
        
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion

        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    # --------------------------------------------------------
    # forward：返回 logits, proto_vector
    # --------------------------------------------------------
    # def forward(self, x):
    #     x = self.relu(self.bn1(self.conv1(x)))
    #     x = self.maxpool(x)

    #     x = self.layer1(x)
    #     x = self.layer2(x)
    #     x = self.layer3(x)
    #     feat_map = self.layer4(x)              # [B,512,H,W]

    #     # Global feature
    #     feat = torch.flatten(self.avgpool(feat_map), 1)  # [B,512]

    #     # 原型向量（用于 ACMP-FL）
    #     proto = self.proto_head(feat)                   # [B,512]

    #     # 分类 logits
    #     logits = self.fc(proto)

    #     return F.log_softmax(logits, dim=1), proto
    def forward(self, x, return_feat=False):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        feat_map = self.layer4(x)                       # [B,512,H,W]

        feat = torch.flatten(self.avgpool(feat_map), 1)  # [B,512]

        # 原型向量（用于原型通信 / 对齐）
        proto = self.proto_head(feat)                   # [B,512]

        # 分类 logits（注意：这里返回 raw logits，不做 softmax/log_softmax）
        logits = self.fc(proto)

        if return_feat:
            return logits, proto
        return logits



# =========================================================
# 🔥 保持函数名不变：resnet18(...)
# main.py 无需修改！
# =========================================================
def resnet18(args, pretrained=True, num_classes=10):
    model = ResNet(args, BasicBlock, [2, 2, 2, 2], num_classes=num_classes)
    if pretrained:
        sd = model_zoo.load_url(model_urls['resnet18'])
        model.load_state_dict(sd, strict=False)
    return model
