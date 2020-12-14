import torch.nn as nn
import linklink as link

# from models.modules.bn_helper import BNFunc
from models.modules.rnn_helper import LSTM


class CRNN(nn.Module):

    def __init__(self, config, normalize_type='sync_bn', bn_group_size=1, to_caffe=False):
        super(CRNN, self).__init__()
        assert normalize_type == 'sync_bn'
        assert not to_caffe

        in_channels = config.model.vision.in_channel
        self.output_dim = 512

        normalize_func = link.nn.SyncBatchNorm2d  # nn.BatchNorm2d  # BNFunc(group_size=bn_group_size)
        self.to_caffe = to_caffe
        kernels = [3, 3, 3, 3, 3, 3, 2]
        pads = [1, 1, 1, 1, 1, 1, 0]
        strides = [1, 1, 1, 1, 1, 1, 1]
        channels = [64, 128, 256, 256, 512, 512, 512]
        hidden_size = 256
        cnn = nn.Sequential()

        def conv_relu_bn(cnn, i, bn=False):
            channel_in = in_channels if i == 0 else channels[i - 1]
            channel_out = channels[i]
            cnn.add_module('conv{0}'.format(i),
                           nn.Conv2d(channel_in, channel_out, kernels[i], strides[i], pads[i]))
            if bn:
                cnn.add_module('batchnorm{0}'.format(i), normalize_func(channel_out))
            cnn.add_module('relu{0}'.format(i), nn.ReLU(True))

        conv_relu_bn(cnn, 0)
        cnn.add_module('pooling{0}'.format(0), nn.MaxPool2d(2, 2))  # 64x16x64
        conv_relu_bn(cnn, 1)
        cnn.add_module('pooling{0}'.format(1), nn.MaxPool2d(2, 2))  # 128x8x32
        conv_relu_bn(cnn, 2)
        conv_relu_bn(cnn, 3)
        cnn.add_module('pooling{0}'.format(2),
                       nn.MaxPool2d((2, 1), (2, 1)))  # 256x4x16
        conv_relu_bn(cnn, 4, True)
        conv_relu_bn(cnn, 5, True)
        cnn.add_module('pooling{0}'.format(3),
                       nn.MaxPool2d((2, 1), (2, 1)))  # 512x2x16
        conv_relu_bn(cnn, 6)  # 512x1x16

        self.cnn = cnn
        self.rnn = nn.Sequential(
            LSTM(512, hidden_size, bidirectional=True),
            LSTM(2 * hidden_size, hidden_size, bidirectional=True))
        self.dropout = nn.Dropout2d(p=0.5)
        # self.fc = nn.Linear(512, class_num)
        # self.log_softmax = nn.LogSoftmax(dim=-1)
        # for m in self.modules():
        #     weight_init(m)

    def forward(self, batch):
        inputs = batch.image
        # conv features
        conv_feature = self.cnn(inputs)
        b, c, h, w = conv_feature.size()
        assert h == 1, "the height of conv must be 1"
        conv_feature = conv_feature.squeeze(2)
        # conv_feature = conv_feature.permute(2, 0, 1)  # [w, b, c] # This is not batch_first
        conv_feature = conv_feature.permute(0, 2, 1)  # [w, b, c]
        # rnn features
        rnn_feature = self.rnn(conv_feature)
        rnn_feature = self.dropout(rnn_feature)
        # output = self.fc(rnn_feature)
        # return output

        rnn_feature, _ = rnn_feature.max(1)

        return rnn_feature
