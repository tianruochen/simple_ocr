from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import sys


class opts:
    def __init__(self, load_model):

        #load_model = '/data1/zhaoshiyu/temp/model_files/ocr_v1024/model_last.pth'
        self.parser = argparse.ArgumentParser()
        # basic experiment setting
        self.parser.add_argument('--task', default='ctdet',
                                 help='ctdet | ddd | multi_pose | exdet')
        self.parser.add_argument('--dataset', default='word',
                                 help='coco | kitti | coco_hp | pascal')
        # self.parser.add_argument('--exp_id', default='default')
        self.parser.add_argument('--exp_id', default='word_test_res18')  # gai
        self.parser.add_argument('--test', action='store_true')
        self.parser.add_argument('--debug', type=int, default=0,
                                 help='level of visualization.'
                                      '1: only show the final detection results'
                                      '2: show the network output features'
                                      '3: use matplot to display'  # useful when lunching training with ipython notebook
                                      '4: save all visualizations to disk')
        self.parser.add_argument('--demo', default='',
                                 help='path to image/ image folders/ video. '
                                      'or "webcam"')
        self.parser.add_argument('--load_model', default=load_model,
                                 help='path to pretrained model')
        self.parser.add_argument('--resume', action='store_true',
                                 help='resume an experiment. '
                                      'Reloaded the optimizer parameter and '
                                      'set load_model to model_last.pth '
                                      'in the exp dir if load_model is empty.')

        # system
        self.parser.add_argument('--gpus', default='2',
                                 help='-1 for CPU, use comma for multiple gpus')

        # model
        self.parser.add_argument('--arch', default='res_18',
                                 help='model architecture. Currently tested'
                                      'res_18 | res_101 | resdcn_18 | resdcn_101 |'
                                      'dlav0_34 | dla_34 | hourglass')
        self.parser.add_argument('--head_conv', type=int, default=-1,
                                 help='conv layer channels for output head'
                                      '0 for no conv layer'
                                      '-1 for default setting: '
                                      '64 for resnets and 256 for dla.')
        self.parser.add_argument('--down_ratio', type=int, default=4,
                                 help='output stride. Currently only supports 4.')

        # input
        self.parser.add_argument('--input_res', type=int, default=-1,
                                 help='input height and width. -1 for default from '
                                      'dataset. Will be overriden by input_h | input_w')
        self.parser.add_argument('--input_h', type=int, default=-1,
                                 help='input height. -1 for default from dataset.')
        self.parser.add_argument('--input_w', type=int, default=-1,
                                 help='input width. -1 for default from dataset.')

        # test
        self.parser.add_argument('--flip_test', action='store_true',
                                 help='flip data augmentation.')
        self.parser.add_argument('--test_scales', type=str, default='1',
                                 help='multi scale test augmentation.')
        self.parser.add_argument('--nms', action='store_true',
                                 help='run nms in testing.')
        self.parser.add_argument('--K', type=int, default=100,
                                 help='max number of output objects.')
        self.parser.add_argument('--not_prefetch_test', action='store_true',
                                 help='not use parallal data pre-processing.')
        self.parser.add_argument('--fix_res', action='store_true',
                                 help='fix testing resolution or keep '
                                      'the original resolution')
        self.parser.add_argument('--keep_res', action='store_true',
                                 help='keep the original resolution'
                                      ' during validation.')

        # loss
        self.parser.add_argument('--mse_loss', action='store_true',
                                 help='use mse loss or focal loss to train '
                                      'keypoint heatmaps.')
        # ctdet
        self.parser.add_argument('--reg_loss', default='l1',
                                 help='regression loss: sl1 | l1 | l2')
        self.parser.add_argument('--hm_weight', type=float, default=1,
                                 help='loss weight for keypoint heatmaps.')
        self.parser.add_argument('--off_weight', type=float, default=1,
                                 help='loss weight for keypoint local offsets.')
        self.parser.add_argument('--wh_weight', type=float, default=0.1,
                                 help='loss weight for bounding box size.')
        # multi_pose
        self.parser.add_argument('--hp_weight', type=float, default=1,
                                 help='loss weight for human pose offset.')
        self.parser.add_argument('--hm_hp_weight', type=float, default=1,
                                 help='loss weight for human keypoint heatmap.')

        # task
        # ctdet
        self.parser.add_argument('--norm_wh', action='store_true',
                                 help='L1(\hat(y) / y, 1) or L1(\hat(y), y)')
        self.parser.add_argument('--dense_wh', action='store_true',
                                 help='apply weighted regression near center or '
                                      'just apply regression on center point.')
        self.parser.add_argument('--cat_spec_wh', action='store_true',
                                 help='category specific bounding box size.')
        self.parser.add_argument('--not_reg_offset', action='store_true',
                                 help='not regress local offset.')

        # multi_pose
        self.parser.add_argument('--dense_hp', action='store_true',
                                 help='apply weighted pose regression near center '
                                      'or just apply regression on center point.')
        self.parser.add_argument('--not_hm_hp', action='store_true',
                                 help='not estimate human joint heatmap, '
                                      'directly use the joint offset from center.')
        self.parser.add_argument('--not_reg_hp_offset', action='store_true',
                                 help='not regress local offset for '
                                      'human joint heatmaps.')
        self.parser.add_argument('--not_reg_bbox', action='store_true',
                                 help='not regression bounding box size.')

    def parse(self, args=''):
        if args == '':
            opt = self.parser.parse_args()
        else:
            opt = self.parser.parse_args(args)

        opt.gpus_str = opt.gpus
        opt.gpus = [int(gpu) for gpu in opt.gpus.split(',')]
        opt.test_scales = [float(i) for i in opt.test_scales.split(',')]

        opt.fix_res = not opt.keep_res
        print('Fix size testing.' if opt.fix_res else 'Keep resolution testing.')
        opt.reg_offset = not opt.not_reg_offset
        opt.reg_bbox = not opt.not_reg_bbox
        opt.hm_hp = not opt.not_hm_hp
        opt.reg_hp_offset = (not opt.not_reg_hp_offset) and opt.hm_hp

        if opt.head_conv == -1:  # init default head_conv
            opt.head_conv = 256 if 'dla' in opt.arch else 64
        opt.pad = 127 if 'hourglass' in opt.arch else 31
        opt.num_stacks = 2 if opt.arch == 'hourglass' else 1
        opt.root_dir = os.path.join(os.path.dirname(__file__), '..', '..')

        return opt

    def update_dataset_info_and_set_heads(self, opt, dataset):
        input_h, input_w = dataset.default_resolution    #[512,512]
        #'mean': [0.408, 0.447, 0.470], 'std': [0.289, 0.274, 0.278]
        opt.mean, opt.std = dataset.mean, dataset.std
        opt.num_classes = dataset.num_classes    #1

        # input_h(w): opt.input_h overrides opt.input_res overrides dataset default
        input_h = opt.input_res if opt.input_res > 0 else input_h
        input_w = opt.input_res if opt.input_res > 0 else input_w
        opt.input_h = opt.input_h if opt.input_h > 0 else input_h
        opt.input_w = opt.input_w if opt.input_w > 0 else input_w
        opt.output_h = opt.input_h // opt.down_ratio
        opt.output_w = opt.input_w // opt.down_ratio
        opt.input_res = max(opt.input_h, opt.input_w)
        opt.output_res = max(opt.output_h, opt.output_w)
        if opt.task == 'ctdet':
            # assert opt.dataset in ['pascal', 'coco']
            opt.heads = {'hm': opt.num_classes,
                         'wh': 2 if not opt.cat_spec_wh else 2 * opt.num_classes}
            if opt.reg_offset:
                opt.heads.update({'reg': 2})
        else:
            assert 0, 'task not defined!'
        print('heads', opt.heads)
        return opt

    def init(self, args=''):

        default_dataset_info = {
            'ctdet': {'default_resolution': [512, 512], 'num_classes': 1,
                      'mean': [0.408, 0.447, 0.470], 'std': [0.289, 0.274, 0.278],
                      'dataset': 'word'}}

        class Struct:
            def __init__(self, entries):
                for k, v in entries.items():
                    self.__dict__[k] = v

        opt = self.parse(args)
        dataset = Struct(default_dataset_info[opt.task])

        # dataset.dataset = "word"
        opt.dataset = dataset.dataset
        opt = self.update_dataset_info_and_set_heads(opt, dataset)
        return opt


if __name__ == "__main__":
    print("sth")

