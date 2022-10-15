from models.vietocr.tool.predictor import Predictor
from models.vietocr.tool.config import Cfg
from models.keras_ocr import detection,tools

def load_model_vietocr():
    
    # for i in path_image_crop:
        config = Cfg.load_config_from_name('vgg_transformer')
        config['weights'] = 'D:/Python/OCR/text-recognition/models/weights/transformerocr.pth'
        config['cnn']['pretrained']=False
        config['device'] = 'cpu'
        config['predictor']['beamsearch']=False

        pred = Predictor(config)

        return pred
def load_model_keras():

    detector = detection.Detector()
    return detector