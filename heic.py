from pathlib import Path
from PIL import Image
import pillow_heif

# Enable HEIC support
pillow_heif.register_heif_opener()

def convert_folder(folder_path):
    folder = Path(folder_path)

    if not folder.is_dir():
        raise ValueError("Provided path is not a folder")

    heic_files = list(folder.glob("*.heic")) + list(folder.glob("*.HEIC"))

    if not heic_files:
        print("No HEIC files found.")
        return

    for heic in heic_files:
        png_path = heic.with_suffix(".png")

        try:
            with Image.open(heic) as img:
                img.save(png_path, format="PNG")
            print(f"✔ {heic.name} → {png_path.name}")
        except Exception as e:
            print(f"✖ Failed: {heic.name} ({e})")

if __name__ == "__main__":
    convert_folder(".")  # current folder
