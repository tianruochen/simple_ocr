# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2
import numpy as np
import torch
from ocr_detector.models.model import create_model, load_model
from ocr_detector.utils.image import get_affine_transform


class BaseDetector(object):
    def __init__(self, opt):
        opt.device = torch.device('cuda')

        # opt.arch = "res_18"
        # opt.heads = opt.heads = {'hm': 1,
        #                          'wh': 2, "reg": 2}
        # opt.head_conv = 64
        # 返回带翻卷积结构的resent18网络
        self.model = create_model(opt.arch, opt.heads, opt.head_conv)
        # 加载模型参数，未将模型放进GPU前，所有的张量都应先加载进CPU，否则回报错
        self.model = load_model(self.model, opt.load_model)
        self.model = self.model.to(opt.device)
        self.model.eval()

        self.mean = np.array(opt.mean, dtype=np.float32).reshape(1, 1, 3)
        self.std = np.array(opt.std, dtype=np.float32).reshape(1, 1, 3)
        self.max_per_image = 100
        self.num_classes = opt.num_classes      #1
        self.scales = opt.test_scales       # 1 用于多尺度检测
        self.opt = opt
        self.pause = True

    def pre_process(self, image, scale, meta=None):
        height, width = image.shape[0:2]
        new_height = int(height * scale)
        new_width = int(width * scale)
        if self.opt.fix_res:   #ture：测试时取一个固定的分辨率   false；测试时保留原始图像的分辨率
            inp_height, inp_width = self.opt.input_h, self.opt.input_w    # 512，512
            c = np.array([new_width / 2., new_height / 2.], dtype=np.float32)   #c是center的坐标
            s = max(height, width) * 1.0
        else:
            inp_height = (new_height | self.opt.pad) + 1     #opt.pad = 31 ??  高度是32的整数倍
            inp_width = (new_width | self.opt.pad) + 1
            c = np.array([new_width // 2, new_height // 2], dtype=np.float32)
            s = np.array([inp_width, inp_height], dtype=np.float32)

        # 获得图像仿射变换矩阵
        trans_input = get_affine_transform(c, s, 0, [inp_width, inp_height])
        resized_image = cv2.resize(image, (new_width, new_height))
        # 根据输入图像以及仿射变换矩阵获得变换后的图片
        inp_image = cv2.warpAffine(
            resized_image, trans_input, (inp_width, inp_height),
            flags=cv2.INTER_LINEAR)
        inp_image = ((inp_image / 255. - self.mean) / self.std).astype(np.float32)

        #h,w,c-->c,h,w-->1,c,h,w
        images = inp_image.transpose(2, 0, 1).reshape(1, 3, inp_height, inp_width)
        if self.opt.flip_test:
            images = np.concatenate((images, images[:, :, :, ::-1]), axis=0)
        images = torch.from_numpy(images)
        meta = {'c': c, 's': s,
                'out_height': inp_height // self.opt.down_ratio,
                'out_width': inp_width // self.opt.down_ratio}
        return images, meta

    def process(self, images, return_time=False):
        raise NotImplementedError

    def post_process(self, dets, meta, scale=1):
        raise NotImplementedError

    def merge_outputs(self, detections):
        raise NotImplementedError

    def debug(self, debugger, images, dets, output, scale=1):
        raise NotImplementedError

    def show_results(self, debugger, image, results):
        raise NotImplementedError

    def run(self, image_or_path_or_tensor, meta=None):
        ##################################
        # d读取图像为ndarray格式
        if isinstance(image_or_path_or_tensor, np.ndarray):
            image = image_or_path_or_tensor
        else:
            image = cv2.imread(image_or_path_or_tensor)
        ##################################

        detections = []
        for scale in self.scales:    # self.scales=【1】
            images, meta = self.pre_process(image, scale, meta)  # 对图像进行预处理仿射变换和变tensor等
            images = images.to(self.opt.device)  # 图像放入gpu
            # output：网络输出
            # det：根据预测的热度图，中心偏移量，宽高 获得Topk的预测结果  （batch，K，6）
            # 6维的向量包括：4维的坐标，1维的得分，1维的类别
            output, dets = self.process(images, return_time=False)

            # dets一个字典（因为测试是batch是1），字典的key是类别编号（从1开始），values是该类别检测结果的集合
            # 检测结果是一个五维度的向量   左上+右下+得分
            dets = self.post_process(dets, meta, scale)
            detections.append(dets)

        # detections 装有字典的列表 字典的key是类别编号（从1开始），values是该类别检测结果的集合
        # merge_outputs 对detections进行过滤 返回格式与detections一致
        results = self.merge_outputs(detections)

        return {'results': results}


if __name__ == "__main__":
    print("sth")


