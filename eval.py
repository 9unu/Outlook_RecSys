import multiprocessing as mp
import pandas as pd
import pickle
import clip
import torch

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

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device, jit=False)
    model.load_state_dict(torch.load("best_model.pt"))
    model.eval()
    model = model.to(device)

    df = pd.read_pickle(r"C:\Users\KHU\Desktop\무신사\processed_data.pickle")
    caption_list = df['caption'].to_list()
    url_list = df['image_link'].to_list()
    with open("image_features.pkl", "rb") as f:
        image_features = pickle.load(f)

    while True:
        text = input("찾으려는 상품: ")  # 유저 input
        image_rank_indices = calculate_similarility(text, model, image_features, device)
        # 관련된 이미지 5개 출력
        for i in image_rank_indices[:5]:
            print(caption_list[i])
            print(url_list[i])