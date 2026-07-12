import io
import urllib.request
from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms as T
from torchvision.datasets import ImageFolder
from PIL import Image

BASE_DIR = Path(__file__).parent.parent
MODEL_DIR = BASE_DIR / "models"
MODEL_PATH = MODEL_DIR / "autoencoder.pt"
IMAGE_SIZE = 128
EMBED_DIM = 256


def _ensure_model_dir() -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)


def image_transform(image_size: int = IMAGE_SIZE) -> T.Compose:
    return T.Compose([
        T.Resize((image_size, image_size)),
        T.CenterCrop(image_size),
        T.ToTensor(),
    ])


class FashionAutoencoder(nn.Module):
    def __init__(self, embedding_dim: int = EMBED_DIM, image_size: int = IMAGE_SIZE) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(128 * (image_size // 8) * (image_size // 8), embedding_dim),
        )

        self.decoder = nn.Sequential(
            nn.Linear(embedding_dim, 128 * (image_size // 8) * (image_size // 8)),
            nn.Unflatten(1, (128, image_size // 8, image_size // 8)),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        latent = self.encoder(x)
        return self.decoder(latent)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_image_encoder(model_path: Optional[str] = None, device: Optional[torch.device] = None) -> Tuple[Optional[FashionAutoencoder], T.Compose]:
    transform = image_transform()
    path = Path(model_path) if model_path else MODEL_PATH

    if not path.exists():
        return None, transform

    device = device or get_device()
    model = FashionAutoencoder().to(device)
    state = torch.load(path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model, transform


def _load_pil_image(path_or_url: str) -> Image.Image:
    if path_or_url.startswith("http://") or path_or_url.startswith("https://"):
        req = urllib.request.Request(path_or_url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req) as response:
            data = response.read()
        return Image.open(io.BytesIO(data)).convert("RGB")
    return Image.open(path_or_url).convert("RGB")


def encode_image(image_path: str, model: FashionAutoencoder, transform: T.Compose, device: Optional[torch.device] = None):
    if model is None:
        return None

    device = device or get_device()
    image = _load_pil_image(image_path)
    tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        embedding = model.encode(tensor)
        embedding = torch.nn.functional.normalize(embedding, dim=1)
    return embedding.cpu().numpy().reshape(-1)


def train_autoencoder(
    data_dir: str,
    model_out: Optional[str] = None,
    epochs: int = 10,
    batch_size: int = 64,
    learning_rate: float = 1e-3,
    image_size: int = IMAGE_SIZE,
    device: Optional[torch.device] = None,
) -> None:
    device = device or get_device()
    model_out_path = Path(model_out) if model_out else MODEL_PATH
    _ensure_model_dir()

    transform = image_transform(image_size)
    dataset = ImageFolder(data_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    model = FashionAutoencoder(image_size=image_size).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    model.train()
    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        for batch in dataloader:
            images, _ = batch
            images = images.to(device)
            optimizer.zero_grad()
            recon = model(images)
            loss = criterion(recon, images)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * images.size(0)

        average_loss = total_loss / max(1, len(dataloader.dataset))
        print(f"Epoch {epoch}/{epochs} - loss: {average_loss:.6f}")

    model_out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), model_out_path)
    print(f"Saved autoencoder model to: {model_out_path}")
