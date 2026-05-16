from engine import fen_to_tensor
import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from chess_cnn import ChessCNN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_and_preprocess(csv_path, sample_size=200000):
    df = pd.read_csv(csv_path)
    df = df[~df['Evaluation'].str.contains('#')]
    df['Evaluation'] = df['Evaluation'].astype(int)
    df['Evaluation'] = df['Evaluation'].clip(-5000, 5000)
    df['Evaluation'] = df['Evaluation'] / 100
    df_sample = df.sample(n=sample_size, random_state=42)
    return df_sample

class ChessDataset(Dataset):
    def __init__(self, dataframe):
        self.fens = dataframe['FEN']
        self.evals = dataframe['Evaluation']

    def __len__(self):
        return len(self.fens)

    def __getitem__(self, idx):
        tensor = fen_to_tensor(self.fens.iloc[idx])
        score = self.evals.iloc[idx]
        return tensor, score


# 1. Veriyi yükle
df = load_and_preprocess("chessData.csv", sample_size=1000000)

# 2. Train/test böl (%80 eğitim, %20 test)
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# 3. Dataset oluştur
train_dataset = ChessDataset(train_df)
test_dataset = ChessDataset(test_df)

# 4. DataLoader oluştur
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# 5. Model, loss, optimizer
model = ChessCNN()
model.to(device)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 6. Eğitim döngüsü
num_epochs = 10

for epoch in range(num_epochs):
    model.train()
    total_loss = 0

    for batch_tensors, batch_scores in train_loader:
        batch_tensors = batch_tensors.float().to(device)
        batch_scores = batch_scores.float().to(device).unsqueeze(1)

        predictions = model(batch_tensors)
        loss = criterion(predictions, batch_scores)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")


# Test loss'u ölç
model.eval()
test_loss = 0
with torch.no_grad():
    for batch_tensors, batch_scores in test_loader:
        batch_tensors = batch_tensors.float().to(device)
        batch_scores = batch_scores.float().to(device).unsqueeze(1)
        predictions = model(batch_tensors)
        loss = criterion(predictions, batch_scores)
        test_loss += loss.item()

avg_test_loss = test_loss / len(test_loader)
print(f"Test Loss: {avg_test_loss:.4f}")

# 7. Modeli kaydet
torch.save(model.state_dict(), "model-1m.pth")
print("Model kaydedildi.")
