"""
Fashion Transformer Autoencoder — Keras 3 (PyTorch backend)
============================================================

Architecture:
  Encoder:
    - 4× Conv2D (stride-2) downsamples 128×128 → 8×8 feature map
    - Reshape to sequence (64 tokens × 256 channels)
    - Learnable positional embedding
    - 2× Transformer self-attention + FFN blocks
    - Dense → embedding_dim (256-D latent vector)

  Decoder:
    - Dense → 64 × 256 sequence
    - Learnable positional embedding
    - 2× Transformer self-attention + FFN blocks
    - Reshape back to 8×8 spatial
    - 4× Conv2DTranspose (stride-2) upsamples 8×8 → 128×128
    - Sigmoid output

Model is saved / loaded as `.keras` native format.
"""

import io
import os
import urllib.request
from pathlib import Path
from typing import Optional, Tuple

# ── Keras 3 with PyTorch backend ─────────────────────────────────────────────
os.environ.setdefault("KERAS_BACKEND", "torch")

import keras
from keras import layers
import numpy as np

# Keep PyTorch available for DataLoader and GPU checks
import torch
from torchvision import transforms as T
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from PIL import Image

BASE_DIR = Path(__file__).parent.parent
MODEL_DIR = BASE_DIR / "models"
MODEL_PATH = MODEL_DIR / "autoencoder.keras"

IMAGE_SIZE = 128
EMBED_DIM  = 256


# ── Transformer Block ─────────────────────────────────────────────────────────
@keras.saving.register_keras_serializable(package="AiFashion")
class TransformerBlock(keras.layers.Layer):
    """Single Transformer block: Multi-Head Attention + FFN with residuals."""

    def __init__(self, embed_dim: int, num_heads: int, ff_dim: int, dropout: float = 0.1, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads  = num_heads
        self.ff_dim     = ff_dim
        self.dropout_rate = dropout

        self.mha   = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim // num_heads)
        self.norm1 = layers.LayerNormalization(epsilon=1e-6)
        self.norm2 = layers.LayerNormalization(epsilon=1e-6)
        self.ffn   = keras.Sequential([
            layers.Dense(ff_dim, activation="gelu"),
            layers.Dense(embed_dim),
        ])
        self.drop1 = layers.Dropout(dropout)
        self.drop2 = layers.Dropout(dropout)

    def call(self, x, training=False):
        attn = self.mha(x, x, training=training)
        attn = self.drop1(attn, training=training)
        x    = self.norm1(x + attn)
        ffn  = self.ffn(x, training=training)
        ffn  = self.drop2(ffn, training=training)
        return self.norm2(x + ffn)

    def get_config(self):
        cfg = super().get_config()
        cfg.update(
            embed_dim=self.embed_dim,
            num_heads=self.num_heads,
            ff_dim=self.ff_dim,
            dropout=self.dropout_rate,
        )
        return cfg


# ── Main Model ────────────────────────────────────────────────────────────────
@keras.saving.register_keras_serializable(package="AiFashion")
class FashionAutoencoder(keras.Model):
    """
    CNN-Transformer Hybrid Autoencoder for fashion images.

    Input:  (batch, 3, 128, 128)  – PyTorch channels-first format
    Output: (batch, 3, 128, 128)  – Reconstructed image [0, 1]
    encode: Returns (batch, embedding_dim) latent vector
    """

    def __init__(self, embedding_dim: int = EMBED_DIM, image_size: int = IMAGE_SIZE, **kwargs):
        super().__init__(**kwargs)
        self.embedding_dim = embedding_dim
        self.image_size    = image_size
        self._spatial = image_size // 16   # = 8

        # ── Encoder CNN ───────────────────────────────────────────────────
        # Input: (B, 3, 128, 128) — channels first
        self.enc_conv1 = layers.Conv2D(32,  4, strides=2, padding="same", data_format="channels_first")
        self.enc_bn1   = layers.BatchNormalization(axis=1)
        self.enc_conv2 = layers.Conv2D(64,  4, strides=2, padding="same", data_format="channels_first")
        self.enc_bn2   = layers.BatchNormalization(axis=1)
        self.enc_conv3 = layers.Conv2D(128, 4, strides=2, padding="same", data_format="channels_first")
        self.enc_bn3   = layers.BatchNormalization(axis=1)
        self.enc_conv4 = layers.Conv2D(256, 4, strides=2, padding="same", data_format="channels_first")
        self.enc_bn4   = layers.BatchNormalization(axis=1)
        # Shape after encoder CNN: (B, 256, 8, 8)

        # ── Encoder Transformer ───────────────────────────────────────────
        # Permute to (B, 8, 8, 256) then reshape to (B, 64, 256)
        self.enc_perm    = layers.Permute((2, 3, 1))
        self.enc_reshape = layers.Reshape((64, 256))
        self.enc_tb1     = TransformerBlock(256, num_heads=4, ff_dim=512)
        self.enc_tb2     = TransformerBlock(256, num_heads=4, ff_dim=512)
        self.enc_flatten = layers.Flatten()
        self.enc_proj    = layers.Dense(embedding_dim, name="latent_proj")

        # ── Decoder Transformer ───────────────────────────────────────────
        self.dec_proj    = layers.Dense(64 * 256, name="dec_proj")
        self.dec_reshape = layers.Reshape((64, 256))
        self.dec_tb1     = TransformerBlock(256, num_heads=4, ff_dim=512)
        self.dec_tb2     = TransformerBlock(256, num_heads=4, ff_dim=512)

        # ── Decoder CNN ───────────────────────────────────────────────────
        # Reshape back to (B, 8, 8, 256) → permute to channels-first (B, 256, 8, 8)
        sp = self._spatial
        self.dec_reshape2 = layers.Reshape((sp, sp, 256))
        self.dec_perm     = layers.Permute((3, 1, 2))

        self.dec_conv1 = layers.Conv2DTranspose(128, 4, strides=2, padding="same", data_format="channels_first")
        self.dec_bn1   = layers.BatchNormalization(axis=1)
        self.dec_conv2 = layers.Conv2DTranspose(64,  4, strides=2, padding="same", data_format="channels_first")
        self.dec_bn2   = layers.BatchNormalization(axis=1)
        self.dec_conv3 = layers.Conv2DTranspose(32,  4, strides=2, padding="same", data_format="channels_first")
        self.dec_bn3   = layers.BatchNormalization(axis=1)
        self.dec_conv4 = layers.Conv2DTranspose(3,   4, strides=2, padding="same", data_format="channels_first")

        # Positional embeddings (learned)
        self._enc_pos = None   # built lazily so Keras serialisation works
        self._dec_pos = None

    def build(self, input_shape):
        self._enc_pos = self.add_weight(
            shape=(1, 64, 256),
            initializer="random_normal",
            trainable=True,
            name="enc_pos_emb",
        )
        self._dec_pos = self.add_weight(
            shape=(1, 64, 256),
            initializer="random_normal",
            trainable=True,
            name="dec_pos_emb",
        )
        super().build(input_shape)

    # ── Forward helpers ───────────────────────────────────────────────────────
    def encode(self, x, training=False):
        """Return normalised latent embedding of shape (batch, embedding_dim)."""
        x = keras.activations.relu(self.enc_bn1(self.enc_conv1(x), training=training))
        x = keras.activations.relu(self.enc_bn2(self.enc_conv2(x), training=training))
        x = keras.activations.relu(self.enc_bn3(self.enc_conv3(x), training=training))
        x = keras.activations.relu(self.enc_bn4(self.enc_conv4(x), training=training))

        x = self.enc_perm(x)       # (B, 8, 8, 256)
        x = self.enc_reshape(x)    # (B, 64, 256)
        x = x + self._enc_pos
        x = self.enc_tb1(x, training=training)
        x = self.enc_tb2(x, training=training)
        x = self.enc_flatten(x)    # (B, 64*256)
        z = self.enc_proj(x)       # (B, embedding_dim)
        return z

    def decode(self, z, training=False):
        d = self.dec_proj(z)        # (B, 64*256)
        d = self.dec_reshape(d)     # (B, 64, 256)
        d = d + self._dec_pos
        d = self.dec_tb1(d, training=training)
        d = self.dec_tb2(d, training=training)

        d = self.dec_reshape2(d)   # (B, 8, 8, 256)
        d = self.dec_perm(d)       # (B, 256, 8, 8)

        d = keras.activations.relu(self.dec_bn1(self.dec_conv1(d), training=training))
        d = keras.activations.relu(self.dec_bn2(self.dec_conv2(d), training=training))
        d = keras.activations.relu(self.dec_bn3(self.dec_conv3(d), training=training))
        d = keras.activations.sigmoid(self.dec_conv4(d))   # (B, 3, 128, 128)
        return d

    def call(self, x, training=False):
        z = self.encode(x, training=training)
        return self.decode(z, training=training)

    def get_config(self):
        cfg = super().get_config()
        cfg.update(embedding_dim=self.embedding_dim, image_size=self.image_size)
        return cfg


# ── Utilities ─────────────────────────────────────────────────────────────────
def _ensure_model_dir() -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def image_transform(image_size: int = IMAGE_SIZE) -> T.Compose:
    return T.Compose([
        T.Resize((image_size, image_size)),
        T.CenterCrop(image_size),
        T.ToTensor(),              # → (3, H, W) float32 in [0, 1]
    ])


def load_image_encoder(
    model_path: Optional[str] = None,
    device: Optional[torch.device] = None,
) -> Tuple[Optional[FashionAutoencoder], T.Compose]:
    """Load trained autoencoder from .keras file.  Returns (model, transform)."""
    transform = image_transform()
    path = Path(model_path) if model_path else MODEL_PATH

    if not path.exists():
        return None, transform

    model = keras.saving.load_model(str(path))
    model.trainable = False
    return model, transform


def _load_pil_image(path_or_url: str) -> Image.Image:
    if path_or_url.startswith("http://") or path_or_url.startswith("https://"):
        req = urllib.request.Request(path_or_url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req) as response:
            data = response.read()
        return Image.open(io.BytesIO(data)).convert("RGB")
    return Image.open(path_or_url).convert("RGB")


def encode_image(
    image_path: str,
    model: FashionAutoencoder,
    transform: T.Compose,
    device: Optional[torch.device] = None,
) -> Optional[np.ndarray]:
    """Encode a single image to a normalised numpy embedding vector."""
    if model is None:
        return None

    image   = _load_pil_image(image_path)
    tensor  = transform(image).unsqueeze(0)  # (1, 3, H, W) PyTorch tensor

    emb = model.encode(tensor, training=False)

    # Convert Keras/PyTorch tensor → numpy
    if hasattr(emb, "numpy"):
        arr = emb.numpy()
    else:
        arr = np.array(emb)

    arr = arr.reshape(-1)
    norm = np.linalg.norm(arr)
    return arr / norm if norm > 0 else arr


# ── Training ──────────────────────────────────────────────────────────────────
def train_autoencoder(
    data_dir: str,
    model_out: Optional[str] = None,
    epochs: int = 20,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    image_size: int = IMAGE_SIZE,
    device: Optional[torch.device] = None,
) -> None:
    """Train the Transformer Autoencoder and save as .keras file."""
    _ensure_model_dir()
    model_out_path = Path(model_out) if model_out else MODEL_PATH

    device = device or get_device()
    print(f"Device     : {device}")
    print(f"Keras back : {keras.backend.backend()}")

    # ── Data ──────────────────────────────────────────────────────────────
    train_transform = T.Compose([
        T.Resize((image_size, image_size)),
        T.RandomHorizontalFlip(),
        T.RandomRotation(5),
        T.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1),
        T.ToTensor(),
    ])
    dataset    = ImageFolder(data_dir, transform=train_transform)
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True,
        num_workers=0, drop_last=True,
    )
    print(f"Dataset    : {len(dataset)} images across {len(dataset.classes)} classes")
    print(f"Classes    : {dataset.classes}")
    print(f"Batches/ep : {len(dataloader)}")

    # ── Model ─────────────────────────────────────────────────────────────
    model = FashionAutoencoder(embedding_dim=EMBED_DIM, image_size=image_size)

    optimizer = keras.optimizers.Adam(learning_rate=learning_rate, weight_decay=1e-5)
    lr_schedule = keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=learning_rate,
        decay_steps=epochs * len(dataloader),
    )

    # ── Training loop ─────────────────────────────────────────────────────
    best_loss = float("inf")
    print(f"\n{'Epoch':>6}  {'Loss':>10}  {'Best':>10}")
    print("─" * 32)

    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        n_samples  = 0

        for images, _ in dataloader:
            # images: PyTorch tensor (B, 3, H, W) in [0, 1]
            with keras.StatelessScope():
                pass

            # Convert to keras-compatible format (still channels-first torch tensor)
            with keras.backend.StatelessScope():
                pass

            # Train step via Keras GradientTape
            with keras.backend.GradientTape() as tape:
                recon = model(images, training=True)
                loss  = keras.losses.mean_squared_error(
                    keras.ops.reshape(images, (-1,)),
                    keras.ops.reshape(recon,  (-1,)),
                )
                # Structured scalar loss
                loss  = keras.ops.mean((recon - images) ** 2)

            grads = tape.gradient(loss, model.trainable_weights)
            optimizer.apply(grads, model.trainable_weights)

            bs = images.shape[0]
            if hasattr(loss, "numpy"):
                total_loss += float(loss.numpy()) * bs
            else:
                total_loss += float(loss) * bs
            n_samples  += bs

        avg_loss = total_loss / max(1, n_samples)
        marker   = " ◀ best" if avg_loss < best_loss else ""

        if avg_loss < best_loss:
            best_loss = avg_loss
            model_out_path.parent.mkdir(parents=True, exist_ok=True)
            model.save(str(model_out_path))

        print(f"{epoch:>6}  {avg_loss:>10.6f}  {best_loss:>10.6f}{marker}")

    print(f"\nTraining complete.")
    print(f"Best loss  : {best_loss:.6f}")
    print(f"Model saved: {model_out_path}")
