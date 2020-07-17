# encoding=utf8

import torch


class CTCLabelConverter(object):
    """ Convert between text-label and text-index """
    def __init__(self, num2str_total):
        # character (str): set of the possible characters.
        dict_character = list(num2str_total)

        # print(len(dict_character))    # 7034
        self.dict = {}
        for i, char in enumerate(dict_character):
            # NOTE: 0 is reserved for 'blank' token required by CTCLoss
            self.dict[char] = i + 1
        # print(len(self.dict))         # 7033   '-' 出现了两次
        self.character = ['[blank]'] + dict_character  # dummy '[blank]' token for CTCLoss (index 0)
        # print(len(self.character))    # 7035
    def encode(self, text, batch_max_length=25):
        """convert text-label into text-index.
        input:
            text: text labels of each image. [batch_size]

        output:
            text: concatenated text index for CTCLoss.
                    [sum(text_lengths)] = [text_index_0 + text_index_1 + ... + text_index_(n - 1)]
            length: length of each text. [batch_size]
        """
        length = [len(s) for s in text]

        text = ''.join(text)
        # text = [self.dict[char] for char in text if char in self.character else 0]
        text1 = []
        for i in text:
            if i in self.character:
                text1.append(self.dict[i])
            else:
                text1.append(0)

        return (torch.IntTensor(text1), torch.IntTensor(length))

    def decode(self, text_index, pre_max, length):
        # (19 x 51,)   (51,)   (19,)--[51]*19
        """ convert text-index into text-label. """
        texts = []
        index = 0
        for l in length:  #51
            t = text_index[index:index + l].cpu().numpy()
            # k = pre_max[index:index + l]
            # k = k.cpu().numpy()
            char_list = []
            for i in range(l):
                if t[i] != 0 and (not (i > 0 and t[i - 1] == t[i])):  # removing repeated characters and blank.
                    char_list.append(self.character[t[i]])
            text = ''.join(char_list)
            texts.append(text)
            index += l
        return texts


if __name__ == "__main__":
    print("sth")
