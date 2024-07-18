import multiprocessing as mp
import pandas as pd
import pickle
import clip
import torch
from process_function import translate_and_summarize
import requests
from PIL import Image
from io import BytesIO
import time
import matplotlib.pyplot as plt
def calculate_similarility(text, model, image_features, device):
    text_encoded = clip.tokenize([text]).to(device)
    text_features = model.encode_text(text_encoded).float()
    similarities = []
    for img_feat in image_features:
        img_feat = img_feat.to(device)
        similarity = torch.cosine_similarity(text_features, img_feat, dim=-1)
        similarities.append(similarity)
    sorted_indices = sorted(range(len(similarities)), key=lambda i: similarities[i], reverse=True)
    return sorted_indices

def display_image(url):
    response=requests.get(url)
    img=Image.open(BytesIO(response.content))
    # img.show()
    plt.imshow(img)
    plt.axis('off')
    plt.show()
    
    

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device, jit=False)
    model.load_state_dict(torch.load("best_model.pt"))
    model.eval()
    model = model.to(device)

    df = pd.read_pickle(r"C:\Users\KHU\Desktop\무신사\total_processed_data.pickle")
    caption_list = df['caption'].to_list()
    image_url_list = df['image_link'].to_list()
    prod_url_list=df['product_link'].to_list()
    with open("image_features.pkl", "rb") as f:
        image_features = pickle.load(f)

    while True:
        text = input("유저 쿼리 : ")  # 유저 input
        start=time.time()
        text=translate_and_summarize(text)
        image_rank_indices = calculate_similarility(text, model, image_features, device)
        # 관련된 이미지 5개 출력
        end=time.time()
        cnt=1
        for i in image_rank_indices[:5]:
            print(f"{cnt}번째 순위")
            cnt+=1
            print(caption_list[i])
            print("이미지 링크:", image_url_list[i])
            print("제품 링크:", prod_url_list[i])
            display_image(image_url_list[i])
            print('소요시간:', end-start, "초")
            print("-"*30)
            