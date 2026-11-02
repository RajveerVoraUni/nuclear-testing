import os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) if 'scripts' in __file__ else os.path.abspath('.')
DATA_DIR = os.path.join(BASE_DIR, 'data') if os.path.exists(os.path.join(BASE_DIR, 'data')) else BASE_DIR

def get_data_path(fname):
    p = os.path.join(DATA_DIR, fname)
    return p if os.path.exists(p) else fname
"""
train_cnn.py
3-Channel 2D Convolutional Neural Network (PyTorch) trained on synchronized
(3, 128, 128) STFT spectrograms:
  - Channel 0: Vertical (Z) Spectrogram
  - Channel 1: North-South / Radial (N) Spectrogram
  - Channel 2: East-West / Transverse (E) Spectrogram
Strictly zero-padded along the time axis (locking dt and df without resizing/distortion).
Evaluated using 5-Fold GroupKFold strictly by seismic event ID (evid).
Saves model weights to 'seismic_cnn_128x128.pt'.
"""

import numpy as np
import scipy.signal as signal
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import GroupKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, balanced_accuracy_score
)

def compute_128x128_spec(sig, fs=20.0, target_freq=128, target_time=128):
    sig = sig - np.mean(sig)
    std_val = np.std(sig)
    if std_val > 1e-8:
        sig = sig / std_val

    nperseg = 256
    noverlap = 236
    if len(sig) < nperseg:
        sig = np.pad(sig, (0, nperseg - len(sig)))

    f, t, Zxx = signal.stft(sig, fs=fs, window="hann", nperseg=nperseg, noverlap=noverlap)
    spec = np.log10(np.abs(Zxx[:target_freq, :])**2 + 1e-6)

    # Zero-padding along time axis (preserve physics)
    if spec.shape[1] < target_time:
        spec = np.pad(spec, ((0, 0), (0, target_time - spec.shape[1])), mode="constant", constant_values=np.min(spec))
    else:
        spec = spec[:, :target_time]

    spec = (spec - np.mean(spec)) / (np.std(spec) + 1e-6)
    return spec.astype(np.float32)

class Dataset3C(Dataset):
    def __init__(self, specs_3c, labels):
        self.specs_3c = specs_3c
        self.labels = labels

    def __len__(self):
        return len(self.specs_3c)

    def __getitem__(self, idx):
        # Shape: (3, 128, 128)
        x = torch.tensor(self.specs_3c[idx], dtype=torch.float32)
        y = torch.tensor(self.labels[idx], dtype=torch.float32)
        return x, y

class SeismicCNN(nn.Module):
    def __init__(self):
        super(SeismicCNN, self).__init__()
        self.features = nn.Sequential(
            # 3 Channels (Z, N, E) in
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2), # (32, 64, 64)

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2), # (64, 32, 32)

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2), # (128, 16, 16)

            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)) # (128, 4, 4)
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(128 * 4 * 4, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        feat = self.features(x)
        feat = feat.view(feat.size(0), -1)
        return self.classifier(feat).squeeze(1)

def main():
    print("Loading 3-Component datasets...")
    eq = np.load(get_data_path("earthquake_data.npz"))
    exp = np.load(get_data_path("explosion_data.npz"))

    specs = []
    labels = []
    groups = []

    print("Generating 3-Channel (Z, N, E) 128x128 spectrogram tensors for earthquakes...")
    for i in range(len(eq["z"])):
        sz = compute_128x128_spec(eq["z"][i])
        sn = compute_128x128_spec(eq["n"][i])
        se = compute_128x128_spec(eq["e"][i])
        s_3c = np.stack([sz, sn, se], axis=0)
        specs.append(s_3c)
        labels.append(0)
        groups.append(str(eq["evid"][i]))

    print("Generating 3-Channel (Z, N, E) 128x128 spectrogram tensors for nuclear explosions...")
    for i in range(len(exp["z"])):
        sz = compute_128x128_spec(exp["z"][i])
        sn = compute_128x128_spec(exp["n"][i])
        se = compute_128x128_spec(exp["e"][i])
        s_3c = np.stack([sz, sn, se], axis=0)
        specs.append(s_3c)
        labels.append(1)
        groups.append(str(exp["evid"][i]))

    specs = np.array(specs)
    labels = np.array(labels)
    groups = np.array(groups)

    print(f"\nConstructed 3C Spectrogram Tensor: {specs.shape}")
    print(f"Total: {len(labels)} | Explosions: {np.sum(labels == 1)} | Earthquakes: {np.sum(labels == 0)}")
    print(f"Unique Events: {len(np.unique(groups))}")

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}\n")

    gkf = GroupKFold(n_splits=5)
    accs, b_accs, precs, recs, f1s, roc_aucs, pr_aucs = [], [], [], [], [], [], []

    print("Beginning 5-Fold Group-KFold 3C CNN Training...")
    for fold, (tr_idx, te_idx) in enumerate(gkf.split(specs, labels, groups=groups)):
        train_ds = Dataset3C(specs[tr_idx], labels[tr_idx])
        test_ds = Dataset3C(specs[te_idx], labels[te_idx])

        train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
        test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)

        model = SeismicCNN().to(device)
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=15)

        for epoch in range(15):
            model.train()
            for x_b, y_b in train_loader:
                x_b, y_b = x_b.to(device), y_b.to(device)
                optimizer.zero_grad()
                loss = criterion(model(x_b), y_b)
                loss.backward()
                optimizer.step()
            scheduler.step()

        model.eval()
        y_true_fold, y_prob_fold = [], []
        with torch.no_grad():
            for x_b, y_b in test_loader:
                x_b = x_b.to(device)
                probs = torch.sigmoid(model(x_b)).cpu().numpy()
                y_prob_fold.extend(probs)
                y_true_fold.extend(y_b.numpy())

        y_true_fold = np.array(y_true_fold)
        y_prob_fold = np.array(y_prob_fold)
        y_pred_fold = (y_prob_fold >= 0.5).astype(int)

        acc = accuracy_score(y_true_fold, y_pred_fold)
        b_acc = balanced_accuracy_score(y_true_fold, y_pred_fold)
        prec = precision_score(y_true_fold, y_pred_fold, zero_division=0)
        rec = recall_score(y_true_fold, y_pred_fold, zero_division=0)
        f1 = f1_score(y_true_fold, y_pred_fold, zero_division=0)
        auc_r = roc_auc_score(y_true_fold, y_prob_fold)
        auc_pr = average_precision_score(y_true_fold, y_prob_fold)

        accs.append(acc)
        b_accs.append(b_acc)
        precs.append(prec)
        recs.append(rec)
        f1s.append(f1)
        roc_aucs.append(auc_r)
        pr_aucs.append(auc_pr)

        print(f"Fold {fold+1}: Acc={acc*100:.1f}%, BalAcc={b_acc*100:.1f}%, Prec={prec*100:.1f}%, Rec={rec*100:.1f}%, ROC-AUC={auc_r:.4f}, PR-AUC={auc_pr:.4f}")

    print("\n" + "="*80)
    print("3-CHANNEL 2D CNN (Z, N, E) PERFORMANCE (5-Fold Event-Grouped CV)")
    print("="*80)
    print(f"Accuracy:          {np.mean(accs)*100:.2f}% +/- {np.std(accs)*100:.2f}%")
    print(f"Balanced Accuracy: {np.mean(b_accs)*100:.2f}% +/- {np.std(b_accs)*100:.2f}%")
    print(f"Precision:         {np.mean(precs)*100:.2f}% +/- {np.std(precs)*100:.2f}%")
    print(f"Recall:            {np.mean(recs)*100:.2f}% +/- {np.std(recs)*100:.2f}%")
    print(f"F1-Score:          {np.mean(f1s):.4f} +/- {np.std(f1s):.4f}")
    print(f"ROC-AUC:           {np.mean(roc_aucs):.4f} +/- {np.std(roc_aucs):.4f}")
    print(f"PR-AUC:            {np.mean(pr_aucs):.4f} +/- {np.std(pr_aucs):.4f}")
    print("="*80)

    # Train final CNN model on all data
    full_ds = Dataset3C(specs, labels)
    full_loader = DataLoader(full_ds, batch_size=32, shuffle=True)
    full_model = SeismicCNN().to(device)
    optimizer = optim.AdamW(full_model.parameters(), lr=1e-3, weight_decay=1e-2)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=15)

    for epoch in range(15):
        full_model.train()
        for x_b, y_b in full_loader:
            x_b, y_b = x_b.to(device), y_b.to(device)
            optimizer.zero_grad()
            loss = criterion(full_model(x_b), y_b)
            loss.backward()
            optimizer.step()
        scheduler.step()

    torch.save(full_model.state_dict(), "seismic_cnn_128x128.pt")
    print("Saved 3-channel CNN weights to seismic_cnn_128x128.pt successfully!")

if __name__ == "__main__":
    main()

