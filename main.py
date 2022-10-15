import argparse
from scripts.load_model import load_model_vietocr
from scripts.main_edge import edge
from scripts.create_mask import mask
import os
from scripts.translate import read, translate
from scripts.writer import draw_text
from scripts.load_model import load_model_keras


def main(args):

    print('---------start load model---------')
    detector = load_model_keras()
    pred = load_model_vietocr()

    path= args.path
    print("---------crop image and create mask--------")
    org, size = mask(path=path, detector=detector)
    print("---------create new font---------")
    edge(mode=2)

    path_image_crop = os.listdir(args.c)
    
    chars = read(pred, path_image_crop)
    print("-----------finish read vietnamese text----------")
    # translate
    text = translate(chars)
    print('------------finish traslate vietnamese to english text-----------')
    print(text)

    draw_text(text, org, size)
   

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, help='path to the image', default='D:/Python/OCR/Naver/vietnamese/vintext/train_images/im0339.jpg')
    parser.add_argument('--c', type=str, default='./crop')
    args = parser.parse_args()

    main(args)