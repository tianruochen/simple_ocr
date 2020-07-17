# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import torch
from ocr_detector.models.decode import ctdet_decode
from ocr_detector.models.utils import flip_tensor
from ocr_detector.utils.post_process import ctdet_post_process
from ocr_detector.detectors.base_detector import BaseDetector


class CtdetDetector(BaseDetector):
    def __init__(self, opt):
        super(CtdetDetector, self).__init__(opt)

    def process(self, images, return_time=False):
        with torch.no_grad():
            # 返回一个列表，列表中只存了一个字典，字典中包括三个key: hm，wh，reg
            output = self.model(images)[-1]  # 得到结果
            hm = output['hm'].sigmoid_()
            # print(hm.shape)     [1,1,128,128]
            wh = output['wh']
            reg = output['reg'] if self.opt.reg_offset else None
            # print(self.opt.flip_test)    false
            if self.opt.flip_test:
                hm = (hm[0:1] + flip_tensor(hm[1:2])) / 2
                wh = (wh[0:1] + flip_tensor(wh[1:2])) / 2
                reg = reg[0:1] if reg is not None else None
            # output：网络输出
            # det：根据预测的热度图，中心偏移量，宽高 获得Topk的预测结果  （batch，K，6）
            # 6维的向量包括：4维的坐标，1维的得分，1维的类别
            dets = ctdet_decode(hm, wh, reg=reg, cat_spec_wh=self.opt.cat_spec_wh, K=self.opt.K)
        return output, dets

    def post_process(self, dets, meta, scale=1):
        dets = dets.detach().cpu().numpy()
        #det：根据预测的热度图，中心偏移量，宽高获得Topk的预测结果  （batch，K，6) 默认检测时batch=1
        dets = dets.reshape(1, -1, dets.shape[2])

        # ctdet_post_process返回一个列表，列表中存放一个字典（因为测试是batch是1），字典的key是类别编号，values是该类别检测结果的集合
        # 检测结果是一个五维度的向量   左上+右下+得分
        dets = ctdet_post_process(
            dets.copy(), [meta['c']], [meta['s']],
            meta['out_height'], meta['out_width'], self.opt.num_classes)
        for j in range(1, self.num_classes + 1):
            dets[0][j] = np.array(dets[0][j], dtype=np.float32).reshape(-1, 5)
            dets[0][j][:, :4] /= scale

        # dets[0]一个字典（因为测试是batch是1），字典的key是类别编号，values是该类别检测结果的集合
        # 检测结果是一个五维度的向量   左上+右下+得分
        return dets[0]

    def merge_outputs(self, detections):
        # detections 字典的列表 字典的key是类别编号（从1开始），values是该类别检测结果的集合
        results = {}
        for j in range(1, self.num_classes + 1):
            results[j] = np.concatenate(
                [detection[j] for detection in detections], axis=0).astype(np.float32)
        scores = np.hstack(
            [results[j][:, 4] for j in range(1, self.num_classes + 1)])
        if len(scores) > self.max_per_image:   #100
            kth = len(scores) - self.max_per_image
            thresh = np.partition(scores, kth)[kth]
            for j in range(1, self.num_classes + 1):
                keep_inds = (results[j][:, 4] >= thresh)
                results[j] = results[j][keep_inds]
        # print(results)
        return results


if __name__ == "__main__":
    print("sth")

