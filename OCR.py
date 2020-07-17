# coding=utf-8

import cv2
from ocr_detector.opts import opts
from ocr_detector.detectors.ctdet import CtdetDetector
from starnet_recognize.recognize import StarnetRecognize
from model_config import ALL_CONFIG


class OCR:
    def __init__(self, config_name):
        # config_name = 'ocr_v1024'
        self.config = ALL_CONFIG[config_name]

        #self.config['detect_file'] = '/data1/zhaoshiyu/temp/model_files/ocr_v1024/model_last.pth'
        opt = opts(self.config['detect_file']).init()
        # 创建检测模型并加载模型参数
        self.detector = CtdetDetector(opt)
        # self.config['recg_file'] = '/data1/zhaoshiyu/temp/model_files/ocr_v1024/starnet_recognize.pth'
        self.recognize = StarnetRecognize(self.config['recg_file'], self.config['num2str_total'])
        self.edge_limit = self.config['edge_limit']   #[4, 4, 4, 4]
        self.join_split = self.config['join_split']   #''

    def image_cut(self, im):  # 对图像进行切分
        h, w, c = im.shape
        if h / (w * 1.0) < 2.5:
            return [im]
        list_image = []
        begin_location = 0
        end_location = 0
        while end_location < h:
            end_location = int(min(begin_location + (w * 2), h))
            list_image.append(im[begin_location:end_location, :, :])
            begin_location = end_location - 40
        # 对于长图，相邻图片切片之间重叠40个像素
        return list_image

    def calculateIoU(self, candidateBound, groundTruthBound):
        cx1 = candidateBound[0]
        cy1 = candidateBound[1]
        cx2 = candidateBound[2]
        cy2 = candidateBound[3]
        gx1 = groundTruthBound[0]
        gy1 = groundTruthBound[1]
        gx2 = groundTruthBound[2]
        gy2 = groundTruthBound[3]
        carea = (cx2 - cx1) * (cy2 - cy1)  # C的面积
        x1 = max(cx1, gx1)
        y1 = max(cy1, gy1)
        x2 = min(cx2, gx2)
        y2 = min(cy2, gy2)
        w = max(0, x2 - x1)
        h = max(0, y2 - y1)
        area = w * h  # C∩G的面积
        iou = area / (carea * 1.0)
        return iou

    def dect_filter(self, ret, height):
        boxes = []
        for i in range(1, 2):
            if len(ret['results'][i]) != 0:
                for box in ret['results'][i]:
                    if box[4] > 0.28:
                        if int(box[1]) < 0:
                            box[1] = 0
                        if box[3] > height:
                            box[3] = height
                        boxes.append(box)

        return boxes

    def bboxes_combine(self, list_bbox, w):  # 对图像的bboxes进行合并
        list_bbox_total = []  # 用来装总的list
        count = 0
        for l_ in list_bbox:
            for l in l_:
                l[1] += count
                l[3] += count
                list_bbox_total.append(l)
            count += w * 2 - 40
        return list_bbox_total

    def dect_sort(self, l):
        l_len = len(l)
        l1 = [[l[i][1], l[i][0], i] for i in range(l_len)]
        l2 = sorted(l1)
        l3 = [l[j[-1]] for j in l2]
        bbox = []
        len_ = len(l3)
        for i in range(len_):
            exit_flag = False
            l = list(range(max(i - 3, 0), i)) + list(range(i + 1, min(i + 3, len_)))
            for j in l:
                iou = self.calculateIoU(l3[i], l3[j])
                if iou > 0.8:
                    exit_flag = True
                    break

            if exit_flag:
                continue
            bbox.append(l3[i])
        return l3

    def cut_recognize_image(self, image, list1, h, w):
        le = int(max(list1[0] - self.edge_limit[0], 0))
        up = int(max(list1[1] - self.edge_limit[1], 0))
        re = int(min(list1[2] + self.edge_limit[2], w))
        do = int(min(list1[3] + self.edge_limit[3], h))
        img = image[up:do, le:re, :]
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return img_gray

    def ocr_rgb(self, image):   #rgb格式的图片   真正的ocr检测函数
        image = image[:, :, (2, 1, 0)]    # rgb->bgr
        h, w, c = image.shape  # 获得其形状

        #==================================检  测===================================
        l = []
        list_image = self.image_cut(image)      #返回一个图片切分后的列表
        for i in list_image:  # 首先根据把长图切分成小图
            height, width = i.shape[:2]

            #预处理+检测+后处理
            #返回一个字典，key为result，value为一个列表，列表中存放一个字典
            #该字典的key为类别编号（从1开始），值为属于该类的bbox集合（列表）
            ret = self.detector.run(i)

            l.append(self.dect_filter(ret, height))  # 对得分小于0.28的框进行过滤，得到一个图片块上的bboexes
        l1 = self.bboxes_combine(l, w)  # 根据切分进行合并，得到整张图片上的bboxes集合
        l2 = self.dect_sort(l1)  # bboxes排序 并根据IOU进行过滤



        #==================================对检测结果进行识别========================================
        l3 = {}  # key是id，values为抠取出来的bboxes
        count = 0
        for item in l2:
            # 从原图中截取bboxes，并将其转换为灰度图
            l3[str(count)] = self.cut_recognize_image(image, item, h, w)
            count += 1
        l4 = self.recognize.recog(l3, 10, 2)
        l5 = []
        for item in l4:
            d = {
                'index': int(item),
                'content': l4[item],
                'bboxes': l2[int(item)].tolist()
            }
            l5.append(d)
        l5.sort(key=lambda x: x['index'])
        return l5, self.join_split.join([item['content'] for item in l5])


if __name__ == "__main__":
    print("sth")

