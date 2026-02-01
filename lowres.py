from PIL import Image

# cd /Users/jirnyak/Mirror/spriter

# python3.11 lowres.py


img = Image.open("input.png")
img = img.resize((128, 128), resample=Image.NEAREST)
img.save("output.png")