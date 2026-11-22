import struct

with open("../data/t10k-images.idx3-ubyte", "rb") as f:
    header = f.read(16)
    magic, num_images, rows, cols = struct.unpack(">IIII", header)
    raw_pixels = f.read()

print(f"Magic: {magic} | Immagini: {num_images} | Dimensioni: {rows}x{cols}")
print(f"Byte totali pixel: {len(raw_pixels)}")

img_size = rows * cols
first_img = raw_pixels[0:img_size]

for r in range(rows):
    line = ""
    for c in range(cols):
        val = first_img[r * cols + c]
        line += "# " if val > 128 else ". "
    print(line)