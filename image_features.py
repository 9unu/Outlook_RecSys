import clip
import torch
import pandas as pd
import pickle
from tqdm import tqdm

def image_encode(model, image):
    with torch.no_grad():
        image_feature = model.encode_image(image).float()
    return image_feature

def get_image_feature(processed_image_list, model, device):
    image_features = []
    model.eval()
    with torch.no_grad():
        for image in tqdm(processed_image_list, desc="Encoding image"):
            image = image.to(device)
            image_feature = image_encode(model, image)
            image_features.append(image_feature.cpu())
    return image_features

def unsqueeze_and_convert(tensor):
    return tensor.unsqueeze(0)

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device, jit=False)
    model.load_state_dict(torch.load("best_model.pt"))
    model = model.to(device)
    df = pd.read_pickle(r"C:\Users\KHU\Desktop\무신사\processed_data.pickle")
    processed_image_list = df['processed_image'].apply(unsqueeze_and_convert).to_list()
    image_features = get_image_feature(processed_image_list, model, device)

    # 학습한 모델로 특성 추출한 값을 리스트 (피클)로 저장
    with open("image_features.pkl", "wb") as f:
        pickle.dump(image_features, f)