import multiprocessing as mp
import torch
import pandas as pd
# import json
import clip
from process_function import image_process, create_chunks, translate_and_summarize
import time

if __name__ == '__main__':
    mp.freeze_support()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, processor = clip.load("ViT-B/32", device=device, jit=False)

    # with open("musinsa_data.json", 'r', encoding='utf-8') as file:
    #     data=json.load(file)

    # origin=pd.DataFrame(data)
    origin=pd.read_csv("musinsa_df_1.csv")
    # origin=origin.iloc[:40]
    """제목 + 태그 + caption 을 전부 모델 학습에 사용하려했으나, 최대 input 길이가 77이라서 우선 caption만 input"""
    # df['all_caption'] = df['title'].apply(lambda x: x.strip()) + "<sep>" + df['caption'].apply(lambda x: x.strip())+ "<sep>" + df['tags'].apply(lambda x: ', '.join(x)).apply(lambda x: x.replace("#","")).apply(lambda x: x.strip())
    origin['caption'] = origin['caption'].apply(translate_and_summarize)
    origin.dropna(inplace=True)
    product_num=[]
    title=[]
    caption=[]
    tags=[]
    image_link=[]
    
    for index, image_link_list in origin['image_links'].items():
        image_link_list=image_link_list.strip("[]").replace("'","").split(', ')
        for image in image_link_list[:1]:
            product_num.append(origin.loc[index, 'product_num'])
            title.append(origin.loc[index, 'title'])
            caption.append(origin.loc[index, 'caption'])
            tags.append(origin.loc[index, 'tags'])
            image_link.append(image)

    df = pd.DataFrame({'product_num': product_num, 'title': title, 'caption': caption, 'tags': tags, 'image_link': image_link})
    """이미지 인코딩 분산 처리 (멀티 프로세싱)"""
    image_list = df['image_link'].to_list()
    chunk_num = mp.cpu_count()
    if len(df)>chunk_num:
        chunks = create_chunks(image_list, chunk_num)
        
        # 전처리된 데이터 병합
        pool = mp.Pool(processes=chunk_num)
        image_list_pool = pool.starmap(image_process, [(chunk, processor) for chunk in chunks])
        pool.close()
        pool.join()
        processed_image_list = [image for sublist in image_list_pool for image in sublist]
        df['processed_image'] = processed_image_list
    
    else:
        df['processed_image']=image_process(image_list, processor)
    
    try:
        df_last= pd.read_pickle("processed_data.pickle")
        df=pd.concat([df_last, df], axis=0)
        df.to_pickle("processed_data.pickle")
    except:
        df.to_pickle("processed_data.pickle")