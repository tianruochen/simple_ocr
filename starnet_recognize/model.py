"""
Copyright (c) 2019-present NAVER Corp.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import torch.nn as nn

from starnet_recognize.modules.transformation import TPS_SpatialTransformerNetwork
from starnet_recognize.modules.feature_extraction import ResNet_FeatureExtractor
from starnet_recognize.modules.sequence_modeling import BidirectionalLSTM


class Model(nn.Module):

    def __init__(self, num_class):
        # num_class = 7035
        super(Model, self).__init__()

        """ Transformation """
        # 基于TPS的空间变换网络  https://zhuanlan.zhihu.com/p/43054073
        # 其forword函数返回 修正后的图片
        # 输入通道是1 转灰度图了
        self.Transformation = TPS_SpatialTransformerNetwork(
            F=20, I_size=(32, 200), I_r_size=(32, 200), I_channel_num=1)

        """ FeatureExtraction """


        # 经过特征提取后 输出shape： torch.Size([19, 512, 1, 51])
        self.FeatureExtraction = ResNet_FeatureExtractor(1, 512)
        self.FeatureExtraction_output = 512  # int(imgH/16-1) * 512
        self.AdaptiveAvgPool = nn.AdaptiveAvgPool2d((None, 1))  # Transform final (imgH/16-1) -> 1

        """ Sequence modeling"""

        # 输出shape： batch_size x T x output_size
        self.SequenceModeling = nn.Sequential(
            BidirectionalLSTM(self.FeatureExtraction_output, 256, 256),
            BidirectionalLSTM(256, 256, 256))
        self.SequenceModeling_output = 256

        """ Prediction """
        self.Prediction = nn.Linear(self.SequenceModeling_output, num_class)

    def forward(self, input, text, is_train=True):
        """ Transformation stage """
        # input:Batch x1 x h x w  text：B x 26
        input = self.Transformation(input)
        # print(input.size())    [19, 1, 32, 200]
        """ Feature extraction stage """
        visual_feature = self.FeatureExtraction(input)
        # print(visual_feature.size())   [19, 512, 1, 51]
        visual_feature = self.AdaptiveAvgPool(visual_feature.permute(0, 3, 1, 2))  # [b, c, h, w] -> [b, w, c, h]
        # print(visual_feature.size())   [19, 51, 512, 1]
        visual_feature = visual_feature.squeeze(3)    #[19, 51, 512]

        """ Sequence modeling stage """
        # batch_size x T x input_size -> batch_size x T x(2 * hidden_size) -> batch_size x T x output_size
        contextual_feature = self.SequenceModeling(visual_feature)
        print(contextual_feature.size())         # [19, 51, 256]

        """ Prediction stage """
        # batch_size x T x output_size(256)--> batch_size x T x num_class(7036)
        # 19 x 51 x 256-->19 x 51 x 7035
        prediction = self.Prediction(contextual_feature.contiguous())
        print(prediction.shape)                #[19, 51, 7035]
        return prediction


if __name__ == "__main__":
    print("sth")
