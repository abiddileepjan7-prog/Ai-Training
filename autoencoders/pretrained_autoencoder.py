import argparse
from pathlib import Path

import torch
from diffusers import AutoencoderKL
from PIL import Image, ImageDraw
from torchvision import transforms


MODEL_ID = "stabilityai/sd-vae-ft-mse"
IMAGE_SIZE = 256


def make_demo_image() -> Image.Image:
    """Create an input image when --input is not supplied."""
    image = Image.new("RGB", (IMAGE_SIZE, IMAGE_SIZE), (232, 239, 250))
    draw = ImageDraw.Draw(image)
    draw.rectangle((24, 24, 232, 232), outline=(40, 84, 150), width=5)
    draw.ellipse((65, 55, 195, 185), fill=(31, 130, 210))
    draw.polygon([(55, 210), (128, 125), (201, 210)], fill=(250, 178, 66))
    return image


def load_image(path: Path | None) -> Image.Image:
    if path is None:
        return make_demo_image()
    if not path.is_file():
        raise FileNotFoundError(f"Input image not found: {path}")
    return Image.open(path).convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE))


def comparison_image(input_image: Image.Image, generated_image: Image.Image) -> Image.Image:
    """Return a labeled input-versus-generated image comparison."""
    gap, label_height = 12, 32
    canvas = Image.new("RGB", (IMAGE_SIZE * 2 + gap, IMAGE_SIZE + label_height), "white")
    canvas.paste(input_image, (0, label_height))
    canvas.paste(generated_image, (IMAGE_SIZE + gap, label_height))

    draw = ImageDraw.Draw(canvas)
    draw.text((8, 9), "Input image", fill="black")
    draw.text((IMAGE_SIZE + gap + 8, 9), "Generated reconstruction", fill="black")
    return canvas


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate an autoencoder reconstruction.")
    parser.add_argument("--input", type=Path, help="Image to reconstruct (optional).")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).with_name("pretrained_comparison.png"),
        help="Where to save the input-versus-generated comparison image.",
    )
    parser.add_argument("--show", action="store_true", help="Open the saved output in the default image viewer.")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    image = load_image(args.input)
    image_tensor = transforms.ToTensor()(image).unsqueeze(0).to(device) * 2 - 1

    print(f"Loading pretrained autoencoder: {MODEL_ID}")
    vae = AutoencoderKL.from_pretrained(MODEL_ID).to(device).eval()
    with torch.inference_mode():
        latent = vae.encode(image_tensor).latent_dist.mode()
        predicted = vae.decode(latent).sample

    output_image = transforms.ToPILImage()(
        ((predicted.squeeze(0).cpu().clamp(-1, 1) + 1) / 2)
    )
    comparison = comparison_image(image, output_image)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    comparison.save(args.output)

    mse = torch.mean((image_tensor.cpu() - predicted.cpu()) ** 2).item()
    print(f"device: {device}")
    print(f"latent shape: {tuple(latent.shape)}")
    print(f"reconstruction MSE: {mse:.6f}")
    print(f"input-versus-generated image saved to: {args.output.resolve()}")

    if args.show:
        try:
            comparison.show()
        except OSError as error:
            print(f"Could not open image viewer: {error}")


if __name__ == "__main__":
    main()
