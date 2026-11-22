import struct
from pathlib import Path

DATA_DIR = Path("../data")  # Modifica se i file sono in un'altra cartella


def find_file(candidates):
    for name in candidates:
        p = DATA_DIR / name
        if p.exists():
            return p
    raise FileNotFoundError(f"Nessun file trovato tra: {candidates} in {DATA_DIR.resolve()}")


def load_images(path: Path):
    with open(path, "rb") as f:
        magic, num_images, rows, cols = struct.unpack(">IIII", f.read(16))
        data = f.read(num_images * rows * cols)
    images = [
        data[i * rows * cols : (i + 1) * rows * cols]
        for i in range(num_images)
    ]
    return images, rows, cols


def load_labels(path: Path):
    with open(path, "rb") as f:
        magic, num_items = struct.unpack(">II", f.read(8))
        labels = list(f.read(num_items))
    return labels


def print_ascii_digit(img_bytes, rows, cols):
    chars = " .:-=+*#%@"
    for r in range(rows):
        line = ""
        for c in range(cols):
            val = img_bytes[r * cols + c]
            idx = min(len(chars) - 1, (val * len(chars)) // 256)
            line += chars[idx] * 2  # Doppio carattere per compensare le proporzioni del terminale
        print(line)


def main():
    images_path = find_file(["t10k-images.idx3-ubyte", "t10k-images-idx3-ubyte"])
    labels_path = find_file(["t10k-labels.idx1-ubyte", "t10k-labels-idx1-ubyte"])

    images, rows, cols = load_images(images_path)
    labels = load_labels(labels_path)

    print(f"Caricate {len(images)} immagini ({rows}x{cols}) e {len(labels)} etichette.")

    while True:
        raw = input("\nInserisci l'indice immagine (0..9999) o 'q' per uscire [default 0]: ").strip()
        if raw.lower() == "q":
            break
        idx = 0 if raw == "" else int(raw)
        if not (0 <= idx < len(images)) or idx >= len(labels):
            print("Indice fuori intervallo.")
            continue

        print(f"\n--- Indice {idx} | Etichetta nel file labels: {labels[idx]} ---")
        print_ascii_digit(images[idx], rows, cols)


if __name__ == "__main__":
    main()