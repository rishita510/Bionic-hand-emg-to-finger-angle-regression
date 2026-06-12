import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt

# ─────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────
FILE_PATH = r'C:\Users\isros\OneDrive\Pictures\emg_dataset_rishita_t2f.csv'

EMG_COLS   = ['emg_ch0', 'emg_ch1', 'emg_ch2', 'emg_ch3', 'emg_ch4', 'emg_ch5']
ANGLE_COLS = ['thumb_flex', 'index_mcp', 'middle_mcp', 'ring_mcp', 'pinkie_mcp']

WINDOW_SIZE  = 200      # ~200ms at 954 Hz
STEP_SIZE    = 10       # 75% overlap
SMOOTH_WIN   = 7        # rolling median window for angle smoothing
BATCH_SIZE   = 64
EPOCHS       = 50
LR           = 1e-3
DEVICE       = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(f"Using device: {DEVICE}")

# ─────────────────────────────────────────
# 1. LOAD DATA
# ─────────────────────────────────────────
df = pd.read_csv(FILE_PATH)
print(f"Data shape: {df.shape}")

# ─────────────────────────────────────────
# 2. ANGLE SMOOTHING (rolling median)
# ─────────────────────────────────────────
for col in ANGLE_COLS:
    df[col] = df[col].rolling(window=SMOOTH_WIN, center=True, min_periods=1).median()

print("Angle smoothing done.")

# ─────────────────────────────────────────
# 3. NORMALIZE EMG
# ─────────────────────────────────────────
scaler = StandardScaler()
df[EMG_COLS] = scaler.fit_transform(df[EMG_COLS])

# ─────────────────────────────────────────
# 4. SLIDING WINDOW
# ─────────────────────────────────────────
def create_windows(df, window_size, step_size):
    X, y = [], []
    emg   = df[EMG_COLS].values
    angle = df[ANGLE_COLS].values

    for start in range(0, len(df) - window_size, step_size):
        end = start + window_size
        X.append(emg[start:end])           # (window_size, 6)
        y.append(angle[end - 1])           # label = last timestep angles

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

X, y = create_windows(df, WINDOW_SIZE, STEP_SIZE)
print(f"Windows: X={X.shape}, y={y.shape}")

# ─────────────────────────────────────────
# 5. TRAIN/TEST SPLIT
# ─────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, shuffle=True
)
print(f"Train: {X_train.shape} | Test: {X_test.shape}")

# ─────────────────────────────────────────
# 6. DATASET & DATALOADER
# ─────────────────────────────────────────
class EMGDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X)  # (N, window, 6)
        self.y = torch.tensor(y)  # (N, 5)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_loader = DataLoader(EMGDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
test_loader  = DataLoader(EMGDataset(X_test,  y_test),  batch_size=BATCH_SIZE, shuffle=False)

# ─────────────────────────────────────────
# 7. CNN-LSTM MODEL
# ─────────────────────────────────────────
class CNN_LSTM(nn.Module):
    def __init__(self, n_channels=6, n_outputs=5):
        super().__init__()

        # CNN block — extract local temporal features
        self.cnn = nn.Sequential(
            nn.Conv1d(n_channels, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)   # (window/2, 64)
        )

        # LSTM block — capture sequential dependencies
        self.lstm = nn.LSTM(
            input_size=64,
            hidden_size=128,
            num_layers=2,
            batch_first=True,
            dropout=0.3
        )

        # Fully connected output
        self.fc = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, n_outputs)
        )

    def forward(self, x):
        # x: (batch, window, channels)
        x = x.permute(0, 2, 1)          # → (batch, channels, window) for Conv1d
        x = self.cnn(x)                  # → (batch, 64, window/2)
        x = x.permute(0, 2, 1)          # → (batch, window/2, 64) for LSTM
        out, _ = self.lstm(x)            # → (batch, window/2, 128)
        out = out[:, -1, :]              # last timestep → (batch, 128)
        out = self.fc(out)               # → (batch, 5)
        return out

model = CNN_LSTM().to(DEVICE)
print(model)
print(f"Total params: {sum(p.numel() for p in model.parameters()):,}")

# ─────────────────────────────────────────
# 8. TRAIN
# ─────────────────────────────────────────
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

train_losses, test_losses = [], []

for epoch in range(EPOCHS):
    # --- Train ---
    model.train()
    train_loss = 0
    for xb, yb in train_loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        optimizer.zero_grad()
        pred = model(xb)
        loss = criterion(pred, yb)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    train_loss /= len(train_loader)

    # --- Eval ---
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for xb, yb in test_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            pred = model(xb)
            test_loss += criterion(pred, yb).item()

    test_loss /= len(test_loader)
    scheduler.step(test_loss)

    train_losses.append(train_loss)
    test_losses.append(test_loss)

    if (epoch + 1) % 5 == 0:
        print(f"Epoch [{epoch+1}/{EPOCHS}] | Train Loss: {train_loss:.6f} | Test Loss: {test_loss:.6f}")

# ─────────────────────────────────────────
# 9. EVALUATE
# ─────────────────────────────────────────
model.eval()
all_preds, all_true = [], []

with torch.no_grad():
    for xb, yb in test_loader:
        xb = xb.to(DEVICE)
        pred = model(xb).cpu().numpy()
        all_preds.append(pred)
        all_true.append(yb.numpy())

all_preds = np.concatenate(all_preds)
all_true  = np.concatenate(all_true)

print("\n── Per-finger MAE ──")
for i, col in enumerate(ANGLE_COLS):
    mae = mean_absolute_error(all_true[:, i], all_preds[:, i])
    r2  = r2_score(all_true[:, i], all_preds[:, i])
    print(f"  {col:15s} | MAE: {mae:.4f} | R²: {r2:.4f}")

# ─────────────────────────────────────────
# 10. PLOT
# ─────────────────────────────────────────
plt.figure(figsize=(10, 4))
plt.plot(train_losses, label='Train Loss')
plt.plot(test_losses,  label='Test Loss')
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.title('CNN-LSTM Training Curve')
plt.legend()
plt.tight_layout()
plt.savefig('training_curve.png')
plt.show()
print("\nTraining curve saved as training_curve.png")

# Save model
torch.save(model.state_dict(), 'cnn_lstm_angle_model.pth')
print("Model saved as cnn_lstm_angle_model.pth")