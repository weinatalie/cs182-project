import pandas as pd
import re
import math
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from torch.nn.utils.rnn import pad_sequence

import sentencepiece as spm

from tqdm import tqdm

file = "amazon_review.csv"

data = pd.read_csv(file).dropna(ignore_index=True)

data["reviewText"].to_csv("amazon_reviews.txt", index=False, header=False)
data["overall"] = data["overall"] -1

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

vocab_size = 16000

'''
spm.SentencePieceTrainer.train(
    input='amazon_reviews.txt',
    model_prefix='amazon_reviews',
    vocab_size=vocab_size,
    model_type='unigram',
    character_coverage=1.0
)
'''

filter_ = 1


df_shuffled = data.sample(frac=filter_, random_state=42).reset_index(drop=True)

# Split the shuffled DataFrame
train_size = 0.8
train_df = df_shuffled.sample(frac=train_size, random_state=42).reset_index(drop=True)
test_df = df_shuffled.drop(train_df.index).reset_index(drop=True)

print("Train Size: ", len(train_df))
print("Test Size: ", len(test_df))

tok = spm.SentencePieceProcessor(model_file='amazon_reviews.model')

print(tok.encode("This book is amazing!", out_type=int))

class GPTBlock(nn.Module):
    def __init__(self, emb_dim, num_heads, ff_dim, dropout=0.1):
        super().__init__()

        self.ln1 = nn.LayerNorm(emb_dim)
        self.attn = nn.MultiheadAttention(
            emb_dim, num_heads, dropout=dropout, batch_first=True
        )

        self.ln2 = nn.LayerNorm(emb_dim)
        self.ff = nn.Sequential(
            nn.Linear(emb_dim, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, emb_dim),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, attn_mask=None, key_padding_mask=None):
        # self-attention only
        h = self.ln1(x)
        h, _ = self.attn(
            h, h, h,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask
        )
        x = x + self.dropout(h)

        # feedforward
        h2 = self.ln2(x)
        h2 = self.ff(h2)
        x = x + self.dropout(h2)

        return x

class DecoderOnlyTransformer(nn.Module):
    def __init__(self, vocab_size, emb_dim=128, num_heads=4,
                 num_layers=4, ff_dim=256, max_len=512, dropout=0.1):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.pos_embed = nn.Embedding(max_len, emb_dim)

        self.blocks = nn.ModuleList([
            GPTBlock(emb_dim, num_heads, ff_dim, dropout)
            for _ in range(num_layers)
        ])

        self.ln_f = nn.LayerNorm(emb_dim)
        self.fc = nn.Linear(emb_dim, 5)

    def forward(self, x, lengths):
        B, L = x.shape

        pos = torch.arange(L, device=x.device).unsqueeze(0)
        h = self.embedding(x) + self.pos_embed(pos)

        causal_mask = torch.triu(
            torch.ones(L, L, device=x.device), diagonal=1
        ).bool()

        pad_mask = (x == 0)

        for block in self.blocks:
            h = block(h, attn_mask=causal_mask, key_padding_mask=pad_mask)

        h = self.ln_f(h)

        idx = (lengths - 1).view(-1, 1, 1).expand(-1, 1, h.size(-1))
        last_hidden = h.gather(1, idx).squeeze(1)

        return self.fc(last_hidden)


def collate_fn(batch):
    sequences, labels = zip(*batch)

    lengths = torch.tensor([len(seq) for seq in sequences])

    padded = pad_sequence(
        sequences,
        batch_first=True, 
        padding_value=0
    )

    labels = torch.stack([l for l in labels])
    return padded, lengths, labels


class ReviewDataset(Dataset):
    def __init__(self, train_dframe, spm_model):
        self.df = train_dframe
        self.sp = spm.SentencePieceProcessor()
        self.sp.load(spm_model)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        text = self.df.loc[idx, "reviewText"]
        label = torch.tensor(self.df.loc[idx, "overall"], dtype = torch.long)
        ids = self.sp.encode(text, out_type=int)
        ids = torch.tensor(ids, dtype=torch.long)

        return ids, label

train_dataset = ReviewDataset(train_df, "amazon_reviews.model")
train_loader = DataLoader(
    train_dataset,
    batch_size=8,
    shuffle=True,
    collate_fn=collate_fn
)

max_len = 0
for padded, lengths, labels in train_loader:
    batch_max = lengths.max().item()
    max_len = max(max_len, batch_max)
print("max_len", max_len)


# Loading Data

model = DecoderOnlyTransformer(vocab_size, max_len = max_len)
model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
criterion = nn.CrossEntropyLoss()
epochs = 10

total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Trainable parameters: {total_params}")


def train(model, dataloader, optimizer, criterion, epochs):
    model.train()
    track_loss = []
    total_loss = 0
    for i in range(epochs):
        total_loss = 0
        
        train_data_loader = tqdm(dataloader)
        
        for padded, lengths, labels in train_data_loader:
            padded = padded.to(device)
            lengths = lengths.to(device)
            labels = labels.to(device)
    
            optimizer.zero_grad()
            outputs = model(padded, lengths)


            loss = criterion(outputs, labels)
            track_loss.append(loss.item())
            avg = sum(track_loss[-10:])/10
    
            loss.backward()
            optimizer.step()
    
            total_loss += loss.item()
            train_data_loader.set_postfix(loss=avg)
            del outputs, loss
            torch.cuda.empty_cache()

        print("Epoch", i+1, total_loss / len(dataloader))
    return total_loss, track_loss


loss, tracked_loss = train(model, train_loader, optimizer, criterion, epochs)



def predict(text):
    ids = torch.tensor(tok.encode(text, out_type=int)).unsqueeze(0)
    lengths = torch.tensor([len(ids)])
    ids = ids.to(device)
    lengths = lengths.to(device)
    with torch.no_grad():
        out = model(ids, lengths)
    return float(out)

#print("Prediction for <I love this book>", predict("I love this book"))

pred = []
for i in range(len(test_df["reviewText"])):
    v = predict(test_df["reviewText"][i])
    pred.append(v)

gt = test_df["overall"].tolist()

gt_th = torch.tensor(gt)
pred_th = torch.tensor(pred)


print("loss", criterion(pred_th,gt_th))

plt.plot(tracked_loss)
