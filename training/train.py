import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from engine.chess_cnn import ChessCNN, fen_to_tensor

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_and_preprocess(csv_path, sample_size=1000000):
    df = pd.read_csv(csv_path)
    df = df[~df['Evaluation'].str.contains('#')]
    df['Evaluation'] = df['Evaluation'].astype(int)
    df['Evaluation'] = df['Evaluation'].clip(-1500, 1500)

    # balanced sampling: take an equal number of positions from each eval range
    # fixes the model collapsing to the dataset average
    df['bin'] = pd.cut(df['Evaluation'], bins=[-1501, -1000, -500, -100, 100, 500, 1000, 1501], labels=False)
    min_count = df['bin'].value_counts().min()
    per_bin = min(min_count, sample_size // 7)
    df_balanced = df.groupby('bin').apply(lambda x: x.sample(n=per_bin, random_state=42)).reset_index(drop=True)
    df_balanced = df_balanced.drop(columns=['bin'])

    df_balanced['Evaluation'] = df_balanced['Evaluation'] / 100

    print(f"balanced dataset size: {len(df_balanced)}")
    print(f"per bin: {per_bin}")

    return df_balanced

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


df = load_and_preprocess("chessData.csv", sample_size=1000000)

train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

train_dataset = ChessDataset(train_df)
test_dataset = ChessDataset(test_df)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

model = ChessCNN()
model.to(device)

criterion = nn.HuberLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

num_epochs = 20

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

torch.save(model.state_dict(), "models/model-balanced-huber-1m.pth")
print("Model saved: model-balanced-huber.pth")