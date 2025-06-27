import time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt

# ───────────────────────────────────────────────
#                Dataset definition
# ───────────────────────────────────────────────
class EEGDenoiseDataset(Dataset):
    def __init__(self, root_dir: str | Path):
        self.samples = []
        root_dir = Path(root_dir)

        for subj in sorted(root_dir.glob("*")):
            if not subj.is_dir():
                continue
            try:
                clean = np.load(subj / "clean.npy")    # shape: (N, ch, T)
                noise = np.load(subj / "noise.npy")
                dirty = np.load(subj / "dirty.npy")
            except FileNotFoundError as e:
                print(f"Skipping {subj.name}: {e}")
                continue

            for d_ep, n_ep, c_ep in zip(dirty, noise, clean):
                for ch in range(d_ep.shape[0]):
                    x = np.stack([c_ep[ch]+n_ep[ch], n_ep[ch]], axis=0)  # (2, T)
                    y = c_ep[ch][None, ...]                    # (1, T)
                    self.samples.append((x.astype(np.float32),
                                          y.astype(np.float32)))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x, y = self.samples[idx]

        # Normalize per-sample (z-score)
        def normalize(z):
            mean = z.mean(axis=1, keepdims=True)
            std  = z.std(axis=1, keepdims=True) + 1e-8
            return (z - mean) / std

        x = normalize(x)
        #y = normalize(y)

        return torch.from_numpy(x), torch.from_numpy(y)


# ───────────────────────────────────────────────
#                   Training
# ───────────────────────────────────────────────
def train(
    root_dir="/content/drive/MyDrive/data_segmented/data_segmented",
    batch_size=32,
    epochs=4,
    lr=2e-4,
    log_dir="runs/denoise",
    val_split=0.1,
    model_save_path="best_model.pt",
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device used:", device)

    # Load dataset and split by sample
    full_dataset = EEGDenoiseDataset(root_dir)
    val_size = int(len(full_dataset) * val_split)
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    print(f"Train size: {train_size} | Validation size: {val_size}")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, drop_last=True, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2, drop_last=False, pin_memory=True)

    # Model, loss, optimizer
    model = DeepDSP_UNetRes().to(device)
    criterion = nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # TensorBoard
    ts = time.strftime("%Y%m%d-%H%M%S")
    writer = SummaryWriter(Path(log_dir) / ts)

    best_val_loss = float("inf")  # initialize
    global_step = 0
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0

        for batch_idx, (x, y) in enumerate(train_loader, 1):
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            y_hat = model(x)
            loss = criterion(y_hat, y)
            loss.backward()
            optimizer.step()

            writer.add_scalar("Loss/batch", loss.item(), global_step)
            global_step += 1
            running_loss += loss.item()

            if batch_idx % 500 == 0:
                print(
                    f"Epoch {epoch:02d}/{epochs}  "
                    f"Batch {batch_idx:04d}/{len(train_loader)}  "
                    f"Loss {loss.item():.6f}"
                )

        # Epoch summary
        train_epoch_loss = running_loss / len(train_loader)
        writer.add_scalar("Loss/epoch", train_epoch_loss, epoch)
        print(f"[{epoch:02d}/{epochs}]  Train avg_loss={train_epoch_loss:.6f}")

        # ───── Validation Pass ─────
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                y_hat = model(x)
                loss = criterion(y_hat, y)
                val_loss += loss.item()

        val_epoch_loss = val_loss / len(val_loader)
        writer.add_scalar("Loss/val_epoch", val_epoch_loss, epoch)
        print(f"[{epoch:02d}/{epochs}]  Validation avg_loss={val_epoch_loss:.6f}")

        # Save best model
        if val_epoch_loss < best_val_loss:
            best_val_loss = val_epoch_loss
            torch.save(model.state_dict(), model_save_path)
            print(f">>> Saved new best model with val_loss={best_val_loss:.6f}")

    writer.close()

    # ───── Final Visual Inspection ─────
    print("\nVisualizing predictions from best model...")
    model.load_state_dict(torch.load(model_save_path))
    model.eval()

    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            y_hat = model(x)

            # Pick first sample from batch
            i = 0
            t = np.arange(y.shape[-1])

            plt.figure(figsize=(10, 4))
            plt.plot(t, y[i, 0].cpu(), label="Ground Truth")
            plt.plot(t, y_hat[i, 0].cpu(), label="Prediction")
            plt.title("EEG Denoising (1 Sample)")
            plt.xlabel("Time (samples)")
            plt.ylabel("µV")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.show()
            break  # only show one batch

if __name__ == "__main__":
    train()