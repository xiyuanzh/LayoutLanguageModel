import torch
import torch.nn as nn
import torch.nn.functional as func
import torch.utils.model_zoo as model_zoo

import linklink as link
from models.modules.rnn_helper import LSTM

__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152', 'resnext50_32x4d', 'resnext101_32x8d']

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
    'resnext50_32x4d': 'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
    'resnext101_32x8d': 'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth',
}


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BlockChannelAttention(nn.Module):

    def __init__(self, channel_dim, hidden_dim):
        super(BlockChannelAttention, self).__init__()

        self.channel_num = channel_dim

        self.w_0 = nn.Linear(channel_dim, hidden_dim, bias=False)
        self.w_1 = nn.Linear(hidden_dim, channel_dim, bias=False)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.maxpool = nn.AdaptiveMaxPool2d((1, 1))

        self.relu = nn.ReLU()

    def forward(self, x):
        assert self.channel_num == x.size(1)

        avg_x = self.avgpool(x).squeeze()
        max_x = self.maxpool(x).squeeze()

        avg_x = self.w_1(self.relu(self.w_0(avg_x)))
        max_x = self.w_1(self.relu(self.w_0(max_x)))

        attn = func.sigmoid(avg_x + max_x).unsqueeze(-1).unsqueeze(-1)

        return x * attn.expand_as(x)


class BlockSpatialAttention(nn.Module):

    def __init__(self):
        super(BlockSpatialAttention, self).__init__()

        self.conv = nn.Conv2d(2, 1, kernel_size=7, stride=1, padding=3, bias=False)

    def forward(self, x):
        avg_x = torch.mean(x, dim=1, keepdim=True)
        max_x, _ = torch.max(x, dim=1, keepdim=True)

        conv_x = self.conv(torch.cat([avg_x, max_x], dim=1))
        conv_x = func.sigmoid(conv_x)

        return x * conv_x.expand_as(x)


class BlockAttention(nn.Module):

    def __init__(self, channel_dim, hidden_dim):
        super(BlockAttention, self).__init__()

        self.channel_attention = BlockChannelAttention(channel_dim, hidden_dim)
        self.spatial_attention = BlockSpatialAttention()

    def forward(self, x):

        x = self.channel_attention(x)
        x = self.spatial_attention(x)

        return x


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None, attention=False):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

        self.attention = attention
        if self.attention:
            self.attn1 = BlockAttention(planes, int(planes // 2))
            self.attn2 = BlockAttention(planes, int(planes // 2))

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        if self.attention:
            out = self.attn1(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.attention:
            out = self.attn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None, attention=False):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

        self.attention = attention
        if attention:
            self.attn1 = BlockAttention(width, int(width // 2))
            self.attn2 = BlockAttention(width, int(width // 2))
            self.attn3 = BlockAttention(planes * self.expansion, int(planes * self.expansion // 2))

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        if self.attention:
            out = self.attn1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        if self.attention:
            out = self.attn2(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.attention:
            out = self.attn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, config, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None, dropout=0.5):
        super(ResNet, self).__init__()

        self.config = config
        # self.output_dim = 512
        # self.output_dim = config.model.vision.output_dim

        if config.model.fusion == 'ConcatFusion':
            self.output_dim = config.model.vision.output_dim
        else:
            self.output_dim = config.model.nlp.bert_config.hidden_size + config.model.position.fc_dim

        self.attention = config.model.vision.attention

        if config.model.vision.use_sync_bn:
            norm_layer = link.nn.SyncBatchNorm2d
        else:
            norm_layer = nn.BatchNorm2d

        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(dropout, inplace=True)
        self.fc = nn.Linear(512 * block.expansion, self.output_dim)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

        # # note: from CRNN, RNN parts
        # hidden_size = self.output_dim // 2
        # self.rnn = nn.Sequential(
        #     LSTM(512 * block.expansion, hidden_size, bidirectional=True),
        #     LSTM(2 * hidden_size, hidden_size, bidirectional=True)
        #     )
        # self.dropout = nn.Dropout2d(p=0.25)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer, attention=self.attention))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer, attention=self.attention))

        return nn.Sequential(*layers)

    def forward(self, batch):
        x = self.conv1(batch.image)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # conv_feature = x
        #
        # b, c, h, w = conv_feature.size()
        # assert h == 1, "the height of conv must be 1"
        # conv_feature = conv_feature.squeeze(2)
        # # conv_feature = conv_feature.permute(2, 0, 1)  # [w, b, c] # This is not batch_first
        # conv_feature = conv_feature.permute(0, 2, 1)  # [w, b, c]
        # # rnn features
        # rnn_feature = self.rnn(conv_feature)
        # rnn_feature = self.dropout(rnn_feature)
        # # output = self.fc(rnn_feature)
        # # return output
        #
        # rnn_feature, _ = rnn_feature.max(1)
        #
        # return rnn_feature

        x = self.avgpool(x)
        x = x.reshape(x.size(0), -1)
        x = self.fc(self.dropout(x))

        x = func.normalize(x)

        # batch_x = list()
        # for char_num in batch.char_num:
        #     batch_x.append(x[:char_num, :])
        #     x = x[char_num:]
        # x = nn.utils.rnn.pad_sequence(batch_x, batch_first=True, padding_value=0)

        return x


def _resnet(arch, block, layers, pretrained, progress, config, **kwargs):
    model = ResNet(block, layers, config, **kwargs)
    if config.model.vision.pre_trained:
        state_dict = model_zoo.load_url(model_urls[arch])
        state_dict.pop('fc.weight')
        state_dict.pop('fc.bias')
        model.load_state_dict(state_dict, strict=False)
    return model


def resnet18(config, pretrained=False, progress=True, **kwargs):
    """Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet18', BasicBlock, [2, 2, 2, 2], pretrained, progress, config,
                   **kwargs)


def resnet34(config, pretrained=False, progress=True, **kwargs):
    """Constructs a ResNet-34 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet34', BasicBlock, [3, 4, 6, 3], pretrained, progress, config,
                   **kwargs)


def resnet50(config, pretrained=False, progress=True, **kwargs):
    """Constructs a ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet50', Bottleneck, [3, 4, 6, 3], pretrained, progress, config,
                   **kwargs)


def resnet101(config, pretrained=False, progress=True, **kwargs):
    """Constructs a ResNet-101 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet101', Bottleneck, [3, 4, 23, 3], pretrained, progress, config,
                   **kwargs)


def resnet152(config, pretrained=False, progress=True, **kwargs):
    """Constructs a ResNet-152 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet152', Bottleneck, [3, 8, 36, 3], pretrained, progress, config,
                   **kwargs)


def resnext50_32x4d(config, pretrained=False, progress=True, **kwargs):
    """Constructs a ResNeXt-50 32x4d model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 4
    return _resnet('resnext50_32x4d', Bottleneck, [3, 4, 6, 3],
                   pretrained, progress, config,  **kwargs)


def resnext101_32x8d(config, pretrained=False, progress=True, **kwargs):
    """Constructs a ResNeXt-101 32x8d model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 8
    return _resnet('resnext101_32x8d', Bottleneck, [3, 4, 23, 3],
                   pretrained, progress, config, **kwargs)
