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
import random

file = "amazon_review.csv"

data = pd.read_csv(file).dropna(ignore_index=True)

data["reviewText"].to_csv("amazon_reviews.txt", index=False, header=False)
data["overall"] = data["overall"] -1

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

vocab_size = 16000
start_token = 16000
end_token = 16001

'''
spm.SentencePieceTrainer.train(
    input='amazon_reviews.txt',
    model_prefix='amazon_reviews',
    vocab_size=vocab_size,
    model_type='unigram',
    character_coverage=1.0
)
'''

tok = spm.SentencePieceProcessor(model_file='amazon_reviews.model')

print(tok.encode("This book is amazing!", out_type=int))

ls  = [len(tok.encode(i, out_type=int)) for i in data["reviewText"]]

data["lengths"] = ls

data_trunc = data[data["lengths"]<=80]

v = data_trunc["overall"].value_counts()

balanced_data = (
    data_trunc.groupby("overall")
      .sample(n=min(v), random_state=42)
      .reset_index(drop=True)
)

filter_ = 1


df_shuffled = balanced_data.sample(frac=filter_, random_state=42).reset_index(drop=True)

# Split the shuffled DataFrame
train_size = 0.8
train_df = df_shuffled.sample(frac=train_size, random_state=42).reset_index(drop=True)
test_df = df_shuffled.drop(train_df.index).reset_index(drop=True)

len_train = len(train_df)
len_test = len(test_df)

print("Train Size: ", len(train_df))
print("Test Size: ", len(test_df))

num_rows = 2000

n = [0, 1, 2, 3, 5, 7, 10]
def getEncoding(df, i):
    row_input = [16000] + tok.encode(df["reviewText"].iloc[i], out_type=int) + [16001]
    row_output = [int(df["overall"].iloc[i])]
    return row_input, row_output

annos_x = {}
for i in n:
    annos_x[i]=[]
annos_y = []
for i in tqdm(range(num_rows)):
    k = random.randint(0,len(test_df) - 1)
    #input_fin = test_df["reviewText"][k]
    #output_fin = test_df["overall"][k]
    input_fin, output_fin = getEncoding(test_df, i)
    icl_df = test_df.drop(test_df.index[k])
    for j in n:
        s= icl_df.sample(n=j)
        in_ = []
        for shot in range(j):
            inp, outp = getEncoding(s, shot)
            in_.extend(inp)
            in_.extend(outp)
        in_.extend(input_fin)
        annos_x[j].append(in_)
    annos_y.extend(output_fin)



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
    def __init__(self, vocab_size, emb_dim=512, num_heads=8,
                 num_layers=12, ff_dim=512, max_len=100, dropout=0.5):
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
        ids_encoded = [start_token] + ids + [end_token]
        ids_encoded = torch.tensor(ids_encoded, dtype=torch.long)
        return ids_encoded, label

train_dataset = ReviewDataset(train_df, "amazon_reviews.model")
train_loader = DataLoader(
    train_dataset,
    batch_size=50,
    shuffle=True,
    collate_fn=collate_fn
)

test_dataset = ReviewDataset(test_df, "amazon_reviews.model")
test_loader = DataLoader(
    test_dataset,
    batch_size=50,
    shuffle=True,
    collate_fn=collate_fn
)



max_len = 0
for padded, lengths, labels in train_loader:
    batch_max = lengths.max().item()
    max_len = max(max_len, batch_max)

for padded, lengths, labels in test_loader:
    batch_max = lengths.max().item()
    max_len = max(max_len, batch_max)
print("max_len", max_len)


# Loading Data

model = DecoderOnlyTransformer(vocab_size+2, max_len = max_len)
model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
criterion = nn.CrossEntropyLoss()
epochs = 15

total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Trainable parameters: {total_params}")


def train(model, dataloader, optimizer, criterion, epochs):
    model.train()
    track_loss = []
    total_loss = 0
    for i in range(epochs):
        model.train()
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
        model.eval()
        total_val_loss = 0
        corr =0
        ll = 0

        with torch.no_grad():
            for padded, lengths, labels in test_loader:
                padded = padded.to(device)
                lengths = lengths.to(device)
                labels = labels.to(device)
                outputs = model(padded, lengths)


                loss = criterion(outputs, labels)

                v = torch.argmax(outputs, dim =1)
                delt = torch.sum(v==labels)
                ll+= len(v)
                corr += delt

                total_val_loss += loss.item()

        print("Epoch", i+1, "track loss: ", total_loss / len(train_data_loader), " val loss: ", total_val_loss / len(test_loader), "val acc: ", corr / len_test )
    return total_loss, track_loss


loss, tracked_loss = train(model, train_loader, optimizer, criterion, epochs)

model.eval()

def predict(text):
    ids = torch.tensor(tok.encode(text, out_type=int)).unsqueeze(0)
    lengths = torch.tensor([len(ids)])
    ids = ids.to(device)
    lengths = lengths.to(device)
    with torch.no_grad():
        out = model(ids, lengths)
        ao = torch.argmax(out)
    return ao

#print("Prediction for <I love this book>", predict("I love this book"))

model.eval()
total_val_loss = 0
corr =0

with torch.no_grad():
    for padded, lengths, labels in test_loader:
        padded = padded.to(device)
        lengths = lengths.to(device)
        labels = labels.to(device)
        outputs = model(padded, lengths)


        loss = criterion(outputs, labels)

        v = torch.argmax(outputs, dim =1)
        delt = torch.sum(v==labels)
        corr += delt

        total_val_loss += loss.item()

print("val acc: ", corr / len_test )

total_train_loss = 0
corr =0

with torch.no_grad():
    for padded, lengths, labels in train_loader:
        padded = padded.to(device)
        lengths = lengths.to(device)
        labels = labels.to(device)
        outputs = model(padded, lengths)


        loss = criterion(outputs, labels)

        v = torch.argmax(outputs, dim =1)
        delt = torch.sum(v==labels)
        corr += delt

        total_train_loss += loss.item()

print("train acc: ", corr / len_train)



class ICLDataset(Dataset):
    def __init__(self, shot_dict, y, shots):
        self.ICL_shot = shot_dict[shots]
        self.y = y

    def __len__(self):
        return len(self.ICL_shot)

    def __getitem__(self, idx):
        text = self.ICL_shot[idx]
        label = torch.tensor(self.y[idx], dtype=torch.long)
        
        #label = self.df.loc[idx, "overall"].item()
        #one_hot_encoded_label = F.one_hot(label, num_classes=5).squeeze(0)
        #ids = self.sp.encode(text, out_type=int)
        ids = torch.tensor(text, dtype=torch.long)

        return ids, label


for shot_ex in n:
    icl_dataset = ICLDataset(annos_x, annos_y, shot_ex)
    icl_loader = DataLoader(
        icl_dataset,
        batch_size=50,
        shuffle=True,
        collate_fn=collate_fn
    )

    ll = 0
    corr = 0
    total_icl_loss = 0
    with torch.no_grad():
        for padded, lengths, labels in icl_loader:
            #padded, lengths, labels = padded.to(device), lengths.to(device), labels.to(device)

            outputs = model(padded, lengths)
            v = torch.argmax(outputs, dim = 1)
            corr += sum(v==labels)
            ll += len(v)
            loss = criterion(outputs, labels)
            total_icl_loss += loss.item()


    avg_icl = total_icl_loss / len(icl_loader)
    icl_acc = corr / ll
    print(shot_ex, avg_val, icl_acc)

plt.plot(tracked_loss)
