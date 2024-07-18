import torch
from torch import nn, optim
import pandas as pd
import numpy as np
import clip
from torch.utils.data import Dataset, DataLoader, BatchSampler
from sklearn.model_selection import train_test_split
from tqdm.notebook import tqdm

class ClipDataset(Dataset):
    def __init__(self, df, preprocess):
        self.preprocess = preprocess
        self.df = df
        self.img_paths = df['image_link'].tolist()
        self.captions = df['caption'].tolist()
        self.processed_images = df['processed_image'].tolist()
        self.path2label = {path: i for i, path in enumerate(self.img_paths)}
        
    def __len__(self):
        return len(self.captions)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        image = self.processed_images[idx]
        caption = self.captions[idx]
        label = self.path2label[img_path]
        return image, caption, label


class BalancedBatchSampler(BatchSampler):
    def __init__(self, labels, n_classes, n_samples):
        self.labels = labels
        self.labels_set = list(set(self.labels.numpy()))
        self.label_to_indices = {label: np.where(self.labels.numpy() == label)[0]
                                 for label in self.labels_set}
        for l in self.labels_set:
            np.random.shuffle(self.label_to_indices[l])
        self.used_label_indices_count = {label: 0 for label in self.labels_set}
        self.count = 0
        self.n_classes = n_classes
        self.n_samples = n_samples
        self.n_dataset = len(self.labels)
        self.batch_size = self.n_samples * self.n_classes

    def __iter__(self):
        self.count = 0
        while self.count + self.batch_size < self.n_dataset:
            classes = np.random.choice(self.labels_set, self.n_classes, replace=False)
            indices = []
            for class_ in classes:
                indices.extend(self.label_to_indices[class_][
                               self.used_label_indices_count[class_]:self.used_label_indices_count[
                                                                         class_] + self.n_samples])
                self.used_label_indices_count[class_] += self.n_samples
                if self.used_label_indices_count[class_] + self.n_samples > len(self.label_to_indices[class_]):
                    np.random.shuffle(self.label_to_indices[class_])
                    self.used_label_indices_count[class_] = 0
            yield indices
            self.count += self.n_classes * self.n_samples

    def __len__(self):
        return self.n_dataset // self.batch_size


def convert_models_to_fp32(model): 
    for p in model.parameters(): 
        p.data = p.data.float() 
        p.grad.data = p.grad.data.float()



if __name__ == "__main__":
    # 모델 초기화
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device, jit=False)
    BATCH_SIZE=8
    EPOCH=30

    # 전처리 마친 데이터셋 : 이미지 (인코딩), 텍스트 (번역 및 요약) 
    df = pd.read_pickle("total_processed_data.pickle")
    # trin, test 셋 분할
    train, test = train_test_split(df, test_size=0.2, random_state=42)
    train_dataset = ClipDataset(train, preprocess)
    test_dataset = ClipDataset(test, preprocess)

    train_labels = torch.tensor([item[2] for item in train_dataset])
    train_sampler = BalancedBatchSampler(train_labels, BATCH_SIZE, 1)
    train_dataloader = DataLoader(train_dataset, batch_sampler=train_sampler)

    test_labels = torch.tensor([item[2] for item in test_dataset])
    test_sampler = BalancedBatchSampler(test_labels, BATCH_SIZE, 1)
    test_dataloader = DataLoader(test_dataset, batch_sampler=test_sampler)

    if device == "cpu":
        model.float()
        
    # 학습 파라미터 설정
    loss_img = nn.CrossEntropyLoss()
    loss_txt = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, len(train_dataloader)*EPOCH)

    # 모델 학습
    best_te_loss = 1e5
    best_ep = -1
    for epoch in range(EPOCH):
        print(f"running epoch {epoch}, best test loss {best_te_loss} after epoch {best_ep}")
        step = 0
        tr_loss = 0
        model.train()
        pbar = tqdm(train_dataloader, leave=False)
        for batch in pbar:
            step += 1
            optimizer.zero_grad()

            images, texts, _ = batch
            images = images.to(device)
            texts = clip.tokenize(texts=texts, truncate=True).to(device)
            logits_per_image, logits_per_text = model(images, texts)
            ground_truth = torch.arange(BATCH_SIZE).to(device)

            total_loss = (loss_img(logits_per_image,ground_truth) + loss_txt(logits_per_text,ground_truth))/2
            total_loss.backward()
            tr_loss += total_loss.item()
            if device == "cpu":
                optimizer.step()
                scheduler.step()
            else:
                convert_models_to_fp32(model)
                optimizer.step()
                scheduler.step()
                clip.model.convert_weights(model)
            pbar.set_description(f"train batchCE: {total_loss.item()}", refresh=True)
        tr_loss /= step
        
        step = 0
        te_loss = 0
        with torch.no_grad():
            model.eval()
            test_pbar = tqdm(test_dataloader, leave=False)
            for batch in test_pbar:
                step += 1
                images, texts, _ = batch
                images = images.to(device)
                texts = clip.tokenize(texts=texts, truncate=True).to(device)
                logits_per_image, logits_per_text = model(images, texts)
                ground_truth = torch.arange(BATCH_SIZE).to(device)

                total_loss = (loss_img(logits_per_image,ground_truth) + loss_txt(logits_per_text,ground_truth))/2
                te_loss += total_loss.item()
                test_pbar.set_description(f"test batchCE: {total_loss.item()}", refresh=True)
            te_loss /= step
            
        if te_loss < best_te_loss:
            best_te_loss = te_loss
            best_ep = epoch
            torch.save(model.state_dict(), "best_model.pt")
        print(f"epoch {epoch}, tr_loss {tr_loss}, te_loss {te_loss}")
    # torch.save(model.state_dict(), "last_model.pt")
