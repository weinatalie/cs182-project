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

tok = spm.SentencePieceProcessor(model_file='amazon_reviews.model')


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



print(tok.encode("This book is amazing!", out_type=int))

class RNNClassifier(nn.Module):
    def __init__(self, vocab_size, emb_dim=512, hidden_size=1024, num_layers=12, dropout=0.2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.rnn = nn.LSTM(
            emb_dim, hidden_size, num_layers,
            batch_first=True, dropout=dropout
        )
        self.fc = nn.Linear(hidden_size, 5)

    def forward(self, x, lengths):
        emb = self.embedding(x)
        #print('A')
        packed = nn.utils.rnn.pack_padded_sequence(
            emb, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        #print('B')
        _, (h_n, _) = self.rnn(packed)
        #print('C')
        last_hidden = h_n[-1]  # last layer hidden state
        pred = self.fc(last_hidden)
        #print('D')
        return pred

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

# Loading Data

model = RNNClassifier(vocab_size)
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



def predict(text):
    ids = torch.tensor(tok.encode(text, out_type=int)).unsqueeze(0)
    lengths = torch.tensor([len(ids)])
    ids = ids.to(device)
    lengths = lengths.to(device)
    with torch.no_grad():
        out = model(ids, lengths)
        ao = torch.argmax(out).item()
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


plt.plot(tracked_loss)
