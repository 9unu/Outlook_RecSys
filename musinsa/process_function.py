import requests
from PIL import Image
from io import BytesIO
import requests
from PIL import Image
from io import BytesIO
from transformers import BartTokenizer, BartForConditionalGeneration
import time
from googletrans import Translator
import torch
import random
"""1. 분산 청크 설정"""
def create_chunks(data_list, chunk_num):
    chunk_size = len(data_list) // chunk_num
    chunks = [data_list[i * chunk_size:(i + 1) * chunk_size] for i in range(chunk_num)]
    if len(data_list) % chunk_num != 0:
        chunks.append(data_list[chunk_num * chunk_size:])
    return chunks

"""2. 이미지 전처리 (CPU)"""
def image_process(image_list, processor):
    processed_image_list = []
    for image_link in image_list:
        response = requests.get(image_link)
        processed_image = processor(Image.open(BytesIO(response.content)))
        processed_image_list.append(processed_image)
    return processed_image_list

"""3. 한글 -> 영어 변환"""
def translate(text):
    translator = Translator()  
    translated = translator.translate(text, src='ko', dest='en')
    time.sleep(random.randint(1,3))
    print(translated.text)
    return translated.text

"""4. 텍스트 요약 전처리"""
def summarize_text(text, max_length=77):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bart_tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-cnn')
    bart_model = BartForConditionalGeneration.from_pretrained('facebook/bart-large-cnn').to(device)
    inputs = bart_tokenizer(text, return_tensors='pt', max_length=1024, truncation=True).to(device)
    summary_ids = bart_model.generate(inputs['input_ids'], max_length=max_length, min_length=30, length_penalty=2.0, num_beams=4, early_stopping=True)
    summary = bart_tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary


"""5. 번역 + 요약 병합"""
def translate_and_summarize(text):
    print(text)
    translator = Translator()
    try:
        translated = translator.translate(text, src='ko', dest='en')
        result = summarize_text(translated.text)
        print(result)
        print('\n')
        return result
    except Exception as e:
        print("번역 오류:", e)
        return "nan"