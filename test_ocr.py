
from paddleocr import PaddleOCR
 
import cv2
 
import numpy as np
 
from PIL import Image, ImageDraw, ImageFont
 
from paddleocr import PaddleOCR, draw_ocr
 
font=cv2.FONT_HERSHEY_SIMPLEX
 
 
 
#Paddleocr目前支持中英文、英文、法语、德语、韩语、日语，可以通过修改lang参数进行切换
 
#参数依次为`ch`, `en`, `french`, `german`, `korean`, `japan`。
 
ocr = PaddleOCR(use_angle_cls=True, lang="ch",use_gpu=False,
 
                rec_model_dir='./inference/ch_ppocr_server_v2.0_rec_infer/',
 
                cls_model_dir='./inference/ch_ppocr_mobile_v2.0_cls_infer/',
 
                det_model_dir='./inference/ch_ppocr_server_v2.0_det_infer/') # need to run only once to download and load model into memory
 
 
 
img_path = './doc/imgs_en/2.jpg'
 
result = ocr.ocr(img_path, cls=True)
 
 
 
# 显示结果
 
from PIL import Image
 
image = Image.open(img_path).convert('RGB')
 
boxes = [line[0] for line in result]
 
txts = [line[1][0] for line in result]
 
scores = [line[1][1] for line in result]
 
print(txts)
im_show = draw_ocr(image, boxes, txts, scores, font_path='./simfang.ttf')
 
im_show = Image.fromarray(im_show)
 
 
 
im_show.save('./result2.png')