import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
from pathlib import Path
from PIL import Image, ImageDraw

# ---------------------------------------------------------------------------
# Adversarial Autoencoder (AAE) - toy, trained on the spot
# Task: encode/decode a sample and report reconstruction error
# ---------------------------------------------------------------------------
print("\n=== AAE: toy demo ===")

encoder = nn.Sequential(nn.Linear(20, 8), nn.ReLU())
decoder = nn.Sequential(nn.Linear(8, 20))
discriminator = nn.Sequential(nn.Linear(8, 1), nn.Sigmoid())

opt = torch.optim.Adam(
    list(encoder.parameters()) + list(decoder.parameters()) + list(discriminator.parameters()),
    lr=0.01,
)

data = torch.randn(64, 20)
for step in range(50):
    opt.zero_grad()
    z = encoder(data)
    recon = decoder(z)
    recon_loss = torch.mean((recon - data) ** 2)
    prior = torch.randn_like(z)
    d_real = discriminator(prior)
    d_fake = discriminator(z)
    adv_loss = -torch.mean(torch.log(d_real + 1e-6) + torch.log(1 - d_fake + 1e-6))
    loss = recon_loss + 0.1 * adv_loss
    loss.backward()
    opt.step()

sample = torch.randn(1, 20)
predicted = decoder(encoder(sample))
print("reconstruction error on new sample:", round(torch.mean((predicted - sample) ** 2).item(), 4))


# ---------------------------------------------------------------------------
# Adaptive Autoencoder - toy, latent size adapts to input variance
# Task: encode/decode a sample and report reconstruction error
# ---------------------------------------------------------------------------
print("\n=== Adaptive Autoencoder: toy demo ===")

sample = torch.randn(1, 20) * 3
latent_dim = 4 if sample.var().item() < 1.0 else 12  # "adapts" to the input

encoder2 = nn.Sequential(nn.Linear(20, latent_dim), nn.ReLU())
decoder2 = nn.Sequential(nn.Linear(latent_dim, 20))

z = encoder2(sample)
recon = decoder2(z)
print("chosen latent dim:", latent_dim)
print("reconstruction error:", round(torch.mean((recon - sample) ** 2).item(), 4))

# This demo works with 20 features rather than pixels.  Display them as a
# 4x5 grayscale image so the input and adaptive reconstruction are visible.
def vector_to_image(vector, low, high, scale=48):
    pixels = vector.detach().cpu().view(4, 5)
    pixels = ((pixels - low) / (high - low + 1e-8) * 255).clamp(0, 255)
    image = Image.fromarray(pixels.to(torch.uint8).numpy(), mode="L")
    return image.resize((5 * scale, 4 * scale), Image.Resampling.NEAREST)


low = torch.minimum(sample.min(), recon.min())
high = torch.maximum(sample.max(), recon.max())
original_image = vector_to_image(sample, low, high)
reconstructed_image = vector_to_image(recon, low, high)

comparison = Image.new("L", (original_image.width * 2 + 12, original_image.height + 28), 255)
comparison.paste(original_image, (0, 28))
comparison.paste(reconstructed_image, (original_image.width + 12, 28))
labels = ImageDraw.Draw(comparison)
labels.text((0, 6), "Input")
labels.text((original_image.width + 12, 6), "Reconstruction")

output_path = Path(__file__).with_name("adaptive_reconstruction.png")
comparison.save(output_path)
print(f"saved image: {output_path}")
try:
    comparison.show()  # Opens the generated comparison in your default image viewer.
except OSError as error:
    # Saving still succeeds in terminals or restricted environments without GUI access.
    print(f"could not open image viewer: {error}")
