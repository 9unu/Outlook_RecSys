import clip
import torch
import pandas as pd
from numba import cuda
import torch.multiprocessing as mp
import requests
from PIL import Image
from io import BytesIO

def image_encode(queue1, model, image):
    with torch.no_grad():
        encoded_image = model.encode_image(image).float().cpu()
        queue1.put(encoded_image)

if __name__ == '__main__':
    mp.freeze_support()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device, jit=False)

    df = pd.read_pickle(r"C:\Users\KHU\Desktop\무신사\eval_data.pickle")

    # 모델 로드
    model.load_state_dict(torch.load("best_model.pt"))
    model.eval()
    model = model.to(device) # 모델을 디바이스로 이동

    # 이미지 데이터셋 로드 및 특징 추출
    image_link = df['image_link'].tolist()
    image_processed = [img.to(device) for img in df['processed_image'].tolist()] # 이미지 텐서를 디바이스로 이동
    image_features = []
    
    print(len(image_processed))
    for image in image_processed[:30]:
        queue1 = mp.Queue()
        p1 = mp.Process(target=image_encode, args=(queue1, model, image))
        p1.start()
        encoded_image = queue1.get()
        p1.join()  # Ensure the process completes before moving on
        image_features.append(encoded_image)
    
    # 텍스트 입력
    text = input("텍스트입력:")

    # 텍스트 인코딩
    with torch.no_grad():
        text_encoded = clip.tokenize([text]).to(device)
        text_features = model.encode_text(text_encoded).float().cpu()

    # 유사도 계산 및 최대값 찾기
    similarities = [torch.cosine_similarity(text_features, img_feat, dim=-1).item() for img_feat in image_features]
    max_idx = similarities.index(max(similarities))

    # 가장 관련된 이미지 출력
    response = requests.get(image_link[max_idx])
    image = Image.open(BytesIO(response.content))
    image.show()