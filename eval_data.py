import multiprocessing as mp
from PIL import Image
import torch
import pandas as pd
import json
import clip
import requests
from io import BytesIO

import torch.multiprocessing as mp

def image_process(df, preprocess):
    processed_images = []
    for _, row in df.iterrows():
        response = requests.get(row['image_link'])
        processed_image = preprocess(Image.open(BytesIO(response.content))).unsqueeze(0).cpu() # GPU 텐서를 CPU 텐서로 변환
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

    image_link=[]
    for index, image_link_list in origin['image_links'].items():
        for image in image_link_list:
            image_link.append(image)

    df = pd.DataFrame({'image_link': image_link})

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

    df.to_pickle("eval_data.pickle")