import pandas as pd
import re
import math
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import numpy as np

from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F

import sentencepiece as spm

from tqdm import tqdm
import random

file = "amazon_review.csv"

data = pd.read_csv(file).dropna(ignore_index=True)
data['overall'] = data["overall"] - 1

vocab_size = 16000
seq_len = 100

spm.SentencePieceTrainer.train(
    input='amazon_reviews.txt',
    model_prefix='amazon_reviews',
    vocab_size=vocab_size,
    model_type='unigram',
    character_coverage=1.0
)

tok = spm.SentencePieceProcessor(model_file='amazon_reviews.model')

filter_ = 1

ls  = [len(tok.encode(i, out_type=int)) for i in data["reviewText"]]

data["lengths"] = ls

data_trunc = data[data["lengths"]<=80]

v = data_trunc["overall"].value_counts()

balanced_data = (
    data_trunc.groupby("overall")
      .sample(n=min(v), random_state=42)
      .reset_index(drop=True)
)

len(balanced_data)

df_shuffled = balanced_data.sample(frac=filter_, random_state=42).reset_index(drop=True)

# Split the shuffled DataFrame
train_size = 0.8
train_df = df_shuffled.sample(frac=train_size, random_state=42).reset_index(drop=True)
test_df = df_shuffled.drop(train_df.index).reset_index(drop=True)

# get one input output pair with special tokens for later concatenation with other pairs for one sequence under max sequence length
def getEncoding(df, i):
    reviewtext = "Review: "+ df["reviewText"].iloc[i]
    rating = "Rating: " + str(int(df["overall"].iloc[i]))
    #row = [16000] + tok.encode(reviewText, out_type=int) + [16001] + tok.encode([int(df["overall"].iloc[i])], out_type = int)
    row = [16000] + tok.encode(reviewtext, out_type = int) + tok.encode(rating, out_type = int) + [16001]
    return row

def getShiftSeq(df_t, max_seq=1024):
    seqs_x = []
    seqs_y = []
    seqs = []
    c = []
    for i in range(len(df_t)):
        row = getEncoding(train_df, i)
        if len(c) + len(row) > max_seq +1:
            seqs_x.append(c[:-1])
            seqs_y.append(c[1:])
            seqs.append(c)
            c = []
        c.extend(row)
    return seqs_x, seqs_y, seqs

train_seqs_x, train_seqs_y, train_seqs = getShiftSeq(train_df, max_seq=seq_len)
test_seqs_x, test_seqs_y, test_seqs = getShiftSeq(test_df, max_seq=seq_len)

class ReviewDataset(Dataset):
    def __init__(self, seqs_x, seqs_y):
        self.seqs_x = seqs_x
        self.seqs_y = seqs_y

    def __len__(self):
        assert len(self.seqs_x) == len(self.seqs_y)
        return len(self.seqs_x)

    def __getitem__(self, idx):
        text = self.seqs_x
        labels = self.seqs_y
        
        #label = self.df.loc[idx, "overall"].item()
        #one_hot_encoded_label = F.one_hot(label, num_classes=5).squeeze(0)
        text_tensor = torch.tensor(text, dtype=torch.long)
        label_tensor = torch.tensor(labels, dtype=torch.long)
        return text_tensor, label_tensor

train_dataset = ReviewDataset(train_seqs_x, train_seqs_y)
train_loader = DataLoader(
    train_dataset,
    batch_size=50,
    shuffle=True,
    collate_fn=collate_fn
)


print(tok.encode("This book is amazing!", out_type=int))

train_df

ex = [1,2,3,4,5, 10, 15, 2005]

# Dataset class for your sequences
def causal_seq(ex, seq_len):
    ex_x = [ex[:i] + [0] * (seq_len-i) for i in range(1,len(ex))]
    ex_y = [ex[i+1] for i in range(len(ex)-1)]
    lens = [i for i in range(1,len(ex))]
    return torch.LongTensor(ex_x), torch.LongTensor(ex_y), torch.LongTensor(lens)

class TokenDatasetA(Dataset):
    def __init__(self, seqs, seq_len=seq_len):
        self.seqs = seqs
        self.seq_len = seq_len
    
    def __len__(self):
        return len(self.seqs)
    
    def __getitem__(self, idx):
        seq, label, lens = causal_seq(self.seqs[idx], self.seq_len)
        return seq, label, lens

def collate_fnA(batch):
    seq_list = [b[0] for b in batch]
    x = torch.cat(seq_list, dim=0)
    
    lab_list = [b[1] for b in batch]
    y = torch.cat(lab_list, dim=0)
    
    le_list = [b[2] for b in batch]
    lens = torch.cat(le_list, dim=0)
    
    return x, y, lens


class TokenDatasetB(Dataset):
    def __init__(self, seqs_x, seqs_y):
        self.seqs_x = seqs_x
        self.seqs_y = seqs_y
    
    def __len__(self):
        return len(self.seqs_x)
    
    def __getitem__(self, idx):
        return torch.LongTensor(self.seqs_x[idx]), torch.LongTensor(self.seqs_y[idx])

# Padding collate function for variable length sequences
def collate_fnB(batch):
    seqs_x, seqs_y = zip(*batch)
    lens = [len(s) for s in seqs_x]
    max_len = max(lens)
    
    # Pad sequences
    padded_x = torch.zeros(len(seqs_x), max_len, dtype=torch.long)
    padded_y = torch.zeros(len(seqs_y), max_len, dtype=torch.long)
    
    for i, (x, y) in enumerate(zip(seqs_x, seqs_y)):
        padded_x[i, :len(x)] = x
        padded_y[i, :len(y)] = y
    
    return padded_x, padded_y, torch.LongTensor(lens)

dataset = TokenDatasetB(train_seqs_x, train_seqs_y)
train_loader = DataLoader(dataset, batch_size=5, shuffle=True, collate_fn=collate_fnB)

for batch_idx, (x, y, lengths) in enumerate(train_loader):
    print("bi")
    print(batch_idx)
    print("x")
    print(x)
    print(x.shape)
    print("y")
    print(y)
    print(y.shape)
    print("lengths")
    print(lengths.shape)
    break

lengths[:10]


# LSTM Model
class LSTMNextToken(nn.Module):
    def __init__(self, vocab_size, embed_dim=64, hidden_dim=128, num_layers=2, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers, 
                           batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, vocab_size)
    
    def forward(self, x, lengths):
        embedded = self.embedding(x)
        
        # Pack padded sequence
        packed = nn.utils.rnn.pack_padded_sequence(
            embedded, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        
        packed_out, _ = self.lstm(packed)
        
        # Unpack
        output, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)
        
        output = self.dropout(output)
        logits = self.fc(output)
        
        return logits

# Training function
def train_model(model, train_loader, epochs=10, lr=0.001, device='cuda'):
    model = model.to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # Ignore padding
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    track_loss = []
    for epoch in range(epochs):
        model.train()
        total_loss = 0

        loader = tqdm(train_loader)
        
        for x, y, lengths in loader:
            x, y = x.to(device), y.to(device)
            lengths = lengths.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            logits = model(x, lengths)
            
            # Reshape for loss calculation
            logits = logits.view(-1, logits.size(-1))
            y = y.view(-1)
            
            loss = criterion(logits, y)
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item()
            track_loss.append(loss.item())
            avg_loss = sum(track_loss[-10:]) / 10
            loader.set_postfix(loss=avg_loss)
        
        avg_loss = total_loss / len(train_loader)
        print(f'Epoch {epoch+1}/{epochs}, Average Loss: {avg_loss:.4f}')
    
    return model

dataset = TokenDatasetB(train_seqs_x, train_seqs_y)
train_loader = DataLoader(dataset, batch_size=5, shuffle=True, collate_fn=collate_fnB)


# Initialize model (set vocab_size to your tokenizer's vocab size + special tokens)
vocab_size = 16000+2  # Adjust based on your tokenizer
model = LSTMNextToken(vocab_size=vocab_size, embed_dim=256, hidden_dim=512, num_layers=2)



# Train
model = train_model(model, train_loader, epochs=10, lr=1e-3, device = 'cpu')

# Save model
torch.save(model.state_dict(), 'lstm_next_token.pth')

# Generate text example
def generate(model, start_tokens, max_length=100, temperature=1.0):
    model.eval()
    tokens = start_tokens.copy()
    
    with torch.no_grad():
        for _ in range(max_length):
            x = torch.LongTensor([tokens]).to(device)
            lengths = torch.LongTensor([len(tokens)])
            
            logits = model(x, lengths)
            next_token_logits = logits[0, -1, :] / temperature
            probs = torch.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, 1).item()
            
            tokens.append(next_token)
            
            if next_token == 16001:  # Stop at end token
                break
    
    return tokens

# Loading Data

model = RNNClassifier(vocab_size+2)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()
epochs = 15

def train(model, dataloader, optimizer, criterion, epochs):
    model.train()
    track_loss = []
    total_loss = 0
    for i in range(epochs):
        model.train()
        total_loss = 0
        
        train_data_loader = tqdm(dataloader)
        
        for padded, lengths, labels in train_data_loader:
            #padded = padded.to(device)
            #lengths = lengths.to(device)
            #labels = labels.to(device)
    
            optimizer.zero_grad()
            outputs = model(padded, lengths)
            
            loss = criterion(outputs, labels)
            track_loss.append(loss.item())
            avg = sum(track_loss[-10:])/10
    
            loss.backward()
            optimizer.step()
    
            total_loss += loss.item()
            train_data_loader.set_postfix(loss=avg)

        avg_train = total_loss / len(train_loader)
        ## ---- VALIDATION ---- ##
        model.eval()
        total_val_loss = 0

        with torch.no_grad():
            for padded, lengths, labels in test_loader:
                #padded, lengths, labels = padded.to(device), lengths.to(device), labels.to(device)

                outputs = model(padded, lengths)
                loss = criterion(outputs, labels)
                total_val_loss += loss.item()

        avg_val = total_val_loss / len(test_loader)


        print(f"Epoch {i+1}: train_loss={avg_train:.4f}, val_loss={avg_val:.4f}")
        
    return total_loss, track_loss


loss,tracked_loss = train(model, train_loader, optimizer, criterion, epochs)

import matplotlib.pyplot as plt

plt.plot(tracked_loss)

ll = 0
corr = 0
total_val_loss = 0
with torch.no_grad():
    for padded, lengths, labels in test_loader:
        #padded, lengths, labels = padded.to(device), lengths.to(device), labels.to(device)

        outputs = model(padded, lengths)
        v = torch.argmax(outputs, dim = 1)
        corr += sum(v==labels)
        ll += len(v)
        loss = criterion(outputs, labels)
        total_val_loss += loss.item()
        

avg_val = total_val_loss / len(test_loader)
val_acc = corr / ll
print(avg_val, val_acc)

def collate_fn_icl(batch):
    sequences, labels = zip(*batch)

    lengths = torch.tensor([len(seq) for seq in sequences])

    padded = pad_sequence(
        sequences,
        batch_first=True, 
        padding_value=0
    )

    labels = torch.stack([l for l in labels])
    return padded, lengths, labels


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

test_df

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
    print(avg_val, icl_acc)

def predict(text):
    ids = torch.tensor(tok.encode(text, out_type=int)).unsqueeze(0)
    lengths = torch.tensor([len(ids)])
    with torch.no_grad():
        out = model(ids, lengths)
        ao = torch.argmax(out).item()
    return ao

predict("best ")

pred = []
for i in range(len(test_df["reviewText"])):
    v = predict(test_df["reviewText"][i])
    pred.append(v)

gt = test_df["overall"].tolist()

gt_th = torch.tensor(gt).int()
pred_th = torch.tensor(pred).int()

sum(gt_th==pred_th)/len(gt_th)

