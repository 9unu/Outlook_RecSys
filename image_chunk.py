import multiprocessing as mp
from PIL import Image
import torch
import pandas as pd
import json
import clip
import requests
from io import BytesIO
from googletrans import Translator
import time
def translate(text):
    translator = Translator()  
    translated = translator.translate(text, src='ko', dest='en')
    time.sleep(1)
    return translated.text

def df_translate(df):
    df['caption']=df['caption'].apply(translate)
    return df

def image_process(df, preprocess):
    processed_images = []
    for _, row in df.iterrows():
        response = requests.get(row['image_link'])
        processed_image = preprocess(Image.open(BytesIO(response.content)))
        processed_images.append(processed_image)
    df['processed_image'] = processed_images
    return df


if __name__ == '__main__':
    mp.freeze_support()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device, jit=False)

    with open("musinsa_data.json", 'r', encoding='utf-8') as file:
        data=json.load(file)
    origin=pd.DataFrame(data)

    origin=origin.iloc[:100]

    product_num=[]
    title=[]
    caption=[]
    tags=[]
    image_link=[]

    for index, image_link_list in origin['image_links'].items():
        for image in image_link_list:
            product_num.append(origin.loc[index, 'product_num'])
            title.append(origin.loc[index, 'title'])
            caption.append(origin.loc[index, 'caption'])
            tags.append(origin.loc[index, 'tags'])
            image_link.append(image)

    df = pd.DataFrame({'product_num': product_num, 'title': title, 'caption': caption, 'tags': tags, 'image_link': image_link})

    """제목 + 태그 + caption 을 전부 모델 학습에 사용하려했으나, 최대 input 길이가 77이라서 우선 caption만 input"""
    # df['all_caption'] = df['title'].apply(lambda x: x.strip()) + "<sep>" + df['caption'].apply(lambda x: x.strip())+ "<sep>" + df['tags'].apply(lambda x: ', '.join(x)).apply(lambda x: x.replace("#","")).apply(lambda x: x.strip())


    # 딕셔너리 df이 이미 정의되어 있다고 가정
    chunk_num = mp.cpu_count()
    chunk_size = len(df) // chunk_num
    chunk_list = []
    for num in range(chunk_num-1):
        chunk_list.append(df.iloc[chunk_size * num:chunk_size * (num + 1)])

    if len(df) % chunk_num != 0:
        chunk_list.append(df.iloc[chunk_num*chunk_size:])
    pool = mp.Pool(processes=len(chunk_list))
    dfs = pool.starmap(image_process, [(chunk, preprocess) for chunk in chunk_list])
    pool.close()
    pool.join()
    df = pd.concat(dfs, ignore_index=True)

    # chunk_num = mp.cpu_count()
    # chunk_size = len(df) // chunk_num
    # chunk_list = []
    # for num in range(chunk_num-1):
    #     chunk_list.append(df.iloc[chunk_size * num:chunk_size * (num + 1)])
    # if len(df) % chunk_num != 0:
    #     chunk_list.append(df.iloc[chunk_num*chunk_size:])
    # pool = mp.Pool(processes=len(chunk_list))
    # dfs = pool.map(df_translate,  chunk_list)
    # pool.close()
    # pool.join()
    # df = pd.concat(dfs, ignore_index=True)

    df.to_pickle("image_processed_data.pickle")