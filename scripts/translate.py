from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import numpy as np
from PIL import Image 


def read(pred, path_image_crop):

    chars = []
    num_file = np.array([])
    for i in path_image_crop:
        num = i.split('.')[0][11:]
        num_file = np.append(num_file, [int(num)])

    for i in path_image_crop:
        j = f'./crop/image_crop_{int(min(num_file))}.jpg'
        img = Image.open(j)

        character = pred.predict(img, return_prob=True)
        if character[1] > 0.8:
            # print("Cac chu tieng viet trong anh: ", character)
            # save word to array
            chars.append(character[0])
            
        num_file = np.delete(num_file, num_file.argmin())
        if num_file == []:
            continue

    return chars

def translate(chars):

    dict_map = {
    "òa": "oà",
    "Òa": "Oà",
    "ÒA": "OÀ",
    "óa": "oá",
    "Óa": "Oá",
    "ÓA": "OÁ",
    "ỏa": "oả",
    "Ỏa": "Oả",
    "ỎA": "OẢ",
    "õa": "oã",
    "Õa": "Oã",
    "ÕA": "OÃ",
    "ọa": "oạ",
    "Ọa": "Oạ",
    "ỌA": "OẠ",
    "òe": "oè",
    "Òe": "Oè",
    "ÒE": "OÈ",
    "óe": "oé",
    "Óe": "Oé",
    "ÓE": "OÉ",
    "ỏe": "oẻ",
    "Ỏe": "Oẻ",
    "ỎE": "OẺ",
    "õe": "oẽ",
    "Õe": "Oẽ",
    "ÕE": "OẼ",
    "ọe": "oẹ",
    "Ọe": "Oẹ",
    "ỌE": "OẸ",
    "ùy": "uỳ",
    "Ùy": "Uỳ",
    "ÙY": "UỲ",
    "úy": "uý",
    "Úy": "Uý",
    "ÚY": "UÝ",
    "ủy": "uỷ",
    "Ủy": "Uỷ",
    "ỦY": "UỶ",
    "ũy": "uỹ",
    "Ũy": "Uỹ",
    "ŨY": "UỸ",
    "ụy": "uỵ",
    "Ụy": "Uỵ",
    "ỤY": "UỴ",
    }
    text = ''
    for i in range(len(chars)):
        text += ' '+ chars[i]
    print(text)
    tokenizer_vi2en = AutoTokenizer.from_pretrained("vinai/vinai-translate-vi2en", src_lang="vi_VN")
    model_vi2en = AutoModelForSeq2SeqLM.from_pretrained("vinai/vinai-translate-vi2en")

    for i, j in dict_map.items():
        text = text.replace(i, j)
    input_ids = tokenizer_vi2en(text, return_tensors="pt").input_ids
    output_ids = model_vi2en.generate(
        input_ids,
        do_sample=True,
        top_k=100,
        top_p=0.8,
        decoder_start_token_id=tokenizer_vi2en.lang_code_to_id["en_XX"],
        num_return_sequences=1,
    )
    en_text = tokenizer_vi2en.batch_decode(output_ids, skip_special_tokens=True)
    en_text = " ".join(en_text)
    return en_text


    