# encoding=utf8
import math
import torch
import numpy as np
import cv2


class NormalizePAD(object):

    def __init__(self, max_size, PAD_type='right'):
        self.max_size = max_size
        self.max_width_half = math.floor(max_size[2] / 2)
        self.PAD_type = PAD_type

    def __call__(self, img):
        img = img / 255.0
        img = torch.from_numpy(np.expand_dims(img, axis=0))
        img.sub_(0.5).div_(0.5)
        c, h, w = img.size()
        Pad_img = torch.FloatTensor(*self.max_size).fill_(0)
        Pad_img[:, :, :w] = img  # right pad
        if self.max_size[2] != w:  # add border Pad
            Pad_img[:, :, w:] = img[:, :, w - 1].unsqueeze(2).expand(c, h, self.max_size[2] - w)
        return Pad_img


class AlignCollate(object):

    def __init__(self, imgH=32, imgW=200, keep_ratio_with_pad=False):
        self.imgH = imgH
        self.imgW = imgW
        self.keep_ratio_with_pad = keep_ratio_with_pad

    def __call__(self, batch):
        #batch:[(图片，图片id)]
        batch = filter(lambda x: x is not None, batch)
        images, labels = zip(*batch)
        resized_max_w = self.imgW
        input_channel = 1
        transform = NormalizePAD((input_channel, self.imgH, resized_max_w))

        resized_images = []
        for image in images:
            h, w = image.shape
            ratio = w / float(h)
            if math.ceil(self.imgH * ratio) > self.imgW:
                resized_w = self.imgW
            else:
                resized_w = math.ceil(self.imgH * ratio)

            resized_image = cv2.resize(image, (int(resized_w), int(self.imgH)))
            resized_images.append(transform(resized_image))
            # resized_image.save('./image_test/%d_test.jpg' % w)

        image_tensors = torch.cat([t.unsqueeze(0) for t in resized_images], 0)
        #image_tensors: N x 1 x H x w  labels:图片id （N，）
        return image_tensors, labels


if __name__ == "__main__":
    print("sth")
