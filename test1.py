import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from PIL import Image
from OCR import OCR
import numpy as np


if __name__ == "__main__":
    temp_model = OCR('ocr_v1024')
    img_path = "/data1/zhaoshiyu/temp/017108.png"
    pil_im = Image.open(img_path)
    pil_im = np.array(pil_im.convert("RGB"))

    ret = temp_model.ocr_rgb(pil_im)
    print(ret)

