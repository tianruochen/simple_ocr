# encoding=utf8

import torch
import torch.utils.data
import torch.nn.functional as F
from starnet_recognize.utils import CTCLabelConverter
from starnet_recognize.dataset import AlignCollate
from starnet_recognize.model import Model


class StarnetRecognize:
    def __init__(self, modelpath, num2str_total):
        # 将一长串字符串转换为了列表  和字典结构
        self.converter = CTCLabelConverter(num2str_total)
        num_class = len(self.converter.character)     # 7035
        # 识别网络：转换网络+双向LSTM + 最后的分类层  前向传播输出shape：batch x T（51）x num_class
        model = Model(num_class)
        self.device = torch.device('cuda')
        self.model = torch.nn.DataParallel(model).to(self.device)
        print('loading pretrained model from %s' % modelpath)
        self.model.load_state_dict(torch.load(modelpath, map_location=self.device))
        self.AlignCollate_demo = AlignCollate(imgH=32, imgW=200, keep_ratio_with_pad=True)

    def data_loader(self, image_name_list, batch_size):
        nSamples = len(image_name_list)
        for i in range(0, nSamples, batch_size):
            yield image_name_list[i:i + batch_size]

    def recognize_unit(self, images_dict):
        '''返回一个字典，字典的key是图片id，value是从图片中识别出来的字符串'''
        image_name_list = [i for i in images_dict]
        # print("image count: ",len(image_name_list))     19
        load_data = self.data_loader(image_name_list, 64)
        results = {}
        self.model.eval()
        with torch.no_grad():
            for batch_name in load_data:
                #[(图片，图片id)...]
                batch = [(images_dict[i], i) for i in batch_name]
                # print(batch)
                # image_tensors: N x 1 x H x W  labels:图片id，元祖 （N，）
                image_tensors, image_path_list = self.AlignCollate_demo(batch)
                batch_size = image_tensors.size(0)
                image = image_tensors.to(self.device)  # [18,1,32,100]resize的

                # For max length prediction
                text_for_pred = torch.LongTensor(batch_size, 26).fill_(0).to(self.device)
                """ 过model，产生输出 """
                # preds: B x T x cla_nums 例如：[19, 51, 7035]
                preds = self.model(image, text_for_pred).log_softmax(2)

                """ tensor转化为str """
                preds_size = torch.IntTensor([preds.size(1)] * batch_size)
                _, preds_index = preds.max(2)
                print(preds_index.shape)    # [19, 51]
                preds_index = preds_index.view(-1)  #(19 x 51,)

                # print(preds_index.data)
                preds_prob = F.softmax(preds, dim=2)   # 19 x 51 x 7035
                preds_max_prob, _ = preds_prob.max(dim=2)    # 19 x 51
                print(preds_max_prob.data[0])        # (51,)

                #参数(19 x 51,)   (51,)   (19,)--[51]*19    返回一个列表，列表中存放batch个字符串
                preds_str = self.converter.decode(preds_index.data, preds_max_prob.data[0], preds_size.data)
                for img_name, pred in zip(image_path_list, preds_str):
                    results[img_name] = pred

        # 返回一个字典，字典的key是图片id，value是从图片中识别出来的字符串
        return results  # ,confidence_score

    def combine(self, list_result_text):
        # l1  0  contents: ['d', 'd']
        # after  combine: {'0': 'd'}
        # l1  1  contents: ['al4gnad']
        # after  combine: {'0': 'd', '1': 'al4gnad'}
        l1 = []
        for i in range(len(list_result_text)):
            if i > 0 and len(list_result_text[i]) > 0 and len(list_result_text[i - 1]) > 0:
                list_last = list_result_text[i - 1][-1] if len(list_result_text[i - 1]) < 2 else \
                    list_result_text[i - 1][-2:] if len(list_result_text[i - 1]) < 3 else \
                    list_result_text[i - 1][-3:] if len(list_result_text[i - 1]) < 4 else \
                    list_result_text[i - 1][-4:]
                list_this = list_result_text[i][0] if len(list_result_text[i]) < 2 else \
                    list_result_text[i][:2] if len(list_result_text[i]) < 3 else \
                    list_result_text[i][:3] if len(list_result_text[i]) < 4 else \
                    list_result_text[i][:4]
                index_this = 0
                index_last = -1
                for j in range(len(list_this)):
                    log = 0
                    for t in range(len(list_last)):
                        if list_this[j] == list_last[t]:
                            log = 1
                            break
                    if log == 1:
                        index_this = j
                        index_last = len(list_last) - t
                        break
                    # if list_this[j] in list_last:
                    #     index_ = j
                    #     break

                sentence = list_result_text[i][index_this:]

                sentence1 = l1[-1][:0 - index_last]
                l1[-1] = l1[-1][:0 - index_last]
            else:
                sentence = list_result_text[i]
            l1.append(sentence)
        return ''.join(l1)

    def long_image_cut(self, image_num, image, cut_ratio, move_ratio):
        '''

        :param image: the input numpy image
        :param cut_ratio: max w/h util cut
        :param move_ratio: for best accuracy the next image must contain h*move_ratio of last image
        :return:
        '''
        h, w = image.shape
        ratio = w / (1.0 * h)
        if ratio < cut_ratio:
            return {image_num + '_0': image}, 1
        elif ratio >= cut_ratio:
            list_image = {}
            begin_location = 0
            end_location = 0
            count = 0
            while end_location < int(w):
                end_location = int(min(begin_location + (h * cut_ratio), int(w)))
                list_image[image_num + '_' + str(count)] = image[:, begin_location:end_location]
                begin_location = end_location - int(h * move_ratio)
                count += 1
            return list_image, len(list_image)
        else:
            print('wrong image size')

    def recog(self, imgs_dict, cut_ratio, move_ratio):
        #imgs_dict：l3, cut_ratio：10, move_ratio：2
        images_dicts = {}
        images_number_record = {}  # 1->1_0、1_1、1_2则1:3
        for i in imgs_dict:  # 这个循环的目的在于将batch中的长图进行分解并进行编号例如1->1_0、1_1、1_2
            image_list, num = self.long_image_cut(i, imgs_dict[i], cut_ratio, move_ratio)
            images_number_record[i] = num
            list_result_text = []
            for j in image_list:
                images_dicts[j] = image_list[j]
        #OCR识别核心部分： 返回一个字典，字典的key是图片id，value是从图片中识别出来的字符串
        preds_results = self.recognize_unit(images_dicts)
        # print(preds_results)   {'0_0': 'd', '0_1': 'd', '1_0': 'al4gnad', '2_0': '傍晚6:09', '3_0': '嘿嘿',...
        images_results = {}
        print("image count: ",images_number_record)
        for i in images_number_record:
            l1 = []   #l1记录属于同一张长图的图像片上识别出来的字符串列表
            for j in range(images_number_record[i]):
                l1.append(preds_results[str(i) + '_' + str(j)])

            print("l1 {} contents: {}".format(i,l1))
            # 将所有属于同一张图片的识别结果进行整合
            images_results[i] = self.combine(l1)    #
            print("after combine:",images_results)
        return images_results


if __name__ == "__main__":
    print("sth")
