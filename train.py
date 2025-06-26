# train.py
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter

from resunet import DeepDSP_UNetRes


# ───────────────────────────────────────────────
#                Dataset definition
# ───────────────────────────────────────────────
class EEGDenoiseDataset(Dataset):
    def __init__(self, root_dir: str | Path):
        self.samples = []
        root_dir = Path(root_dir)

        for subj in root_dir.iterdir():
            if not subj.is_dir():
                continue

            clean = np.load(subj / "clean.npy")   # (N_epochs, 26, 52)
            noise = np.load(subj / "noise.npy")
            dirty = np.load(subj / "dirty.npy")

            # one sample = one electrode’s 0.1-s window
            for d_ep, n_ep, c_ep in zip(dirty, noise, clean):     # epoch loop
                for ch in range(d_ep.shape[0]):                   # channel loop
                    x = np.stack([d_ep[ch], n_ep[ch]], axis=0)    # (2, 52)
                    y = c_ep[ch][None, ...]                       # (1, 52)
                    self.samples.append((x.astype(np.float32),
                                          y.astype(np.float32)))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x, y = self.samples[idx]
        return torch.from_numpy(x), torch.from_numpy(y)


# ───────────────────────────────────────────────
#                   Training
# ───────────────────────────────────────────────
def train(
    root_dir="data_segmented/",
    batch_size=64,
    epochs=2,
    lr=2e-4,
    log_dir="runs/denoise",
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('Device used is: ', device)

    # Data
    dataset = EEGDenoiseDataset(root_dir)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    # Model / loss / opt
    model = DeepDSP_UNetRes(in_channels=2, out_channels=1).to(device)
    criterion = nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # TensorBoard
    ts = time.strftime("%Y%m%d-%H%M%S")
    writer = SummaryWriter(Path(log_dir) / ts)

    global_step = 0
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0

        for x, y in loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()

            y_hat = model(x)
            loss = criterion(y_hat, y)
            loss.backward()
            optimizer.step()

            # TensorBoard step logging
            writer.add_scalar("Loss/step", loss.item(), global_step)
            global_step += 1
            running_loss += loss.item()

        epoch_loss = running_loss / len(loader)
        writer.add_scalar("Loss/epoch", epoch_loss, epoch)
        print(f"[{epoch:02d}/{epochs}]  loss={epoch_loss:.6f}")

    # Save final model
    # Path("checkpoints").mkdir(exist_ok=True)
    # ckpt_path = Path("checkpoints") / f"dsp_model_{ts}.pth"
    # torch.save(model.state_dict(), ckpt_path)
    # print(f"Model saved → {ckpt_path}")

    writer.close()


if __name__ == "__main__":
    train()
