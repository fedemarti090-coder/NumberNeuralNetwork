import re
from pathlib import Path
import matplotlib.pyplot as plt

# Metti qui il percorso del tuo file di log
LOG_FILE_PATH = r"C:\Users\utente\Desktop\C++\NumberNeuralNetwork\data\log\60kresponce.log"


def read_log_text(file_path: Path) -> str:
    raw_bytes = file_path.read_bytes()

    # Gestisce UTF-16 LE/BE (comune con il reindirizzamento '>' in Windows PowerShell)
    if raw_bytes.startswith((b"\xff\xfe", b"\xfe\xff")):
        for enc in ("utf-16", "utf-16-le", "utf-16-be"):
            try:
                return raw_bytes.decode(enc)
            except UnicodeDecodeError:
                continue

    text = raw_bytes.decode("utf-8", errors="replace")
    # Se ci sono molti byte nulli, probabilmente è UTF-16 senza BOM
    if text.count("\x00") > len(text) // 4:
        return raw_bytes.decode("utf-16-le", errors="replace")

    return text


def parse_logs(text: str):
    steps = []
    accuracies = []
    mses = []

    cleaned_text = text.replace("\x00", "").replace("\r\n", "\n").replace("\r", "\n")

    acc_pattern = re.compile(
        r"immagin\w*\s*[:=]\s*(\d+).*?accurat\w*\s*[:=]\s*([0-9]+(?:[.,][0-9]+)?)",
        re.IGNORECASE,
    )
    mse_pattern = re.compile(
        r"(?:mse|loss|errore)\s*[:=]\s*([0-9]+(?:[.,][0-9]+)?)",
        re.IGNORECASE,
    )

    for line in cleaned_text.splitlines():
        line = line.strip()
        if not line:
            continue

        acc_match = acc_pattern.search(line)
        if acc_match:
            step = int(acc_match.group(1))
            acc = float(acc_match.group(2).replace(",", "."))
            steps.append(step)
            accuracies.append(acc)

            mse_match = mse_pattern.search(line)
            if mse_match:
                mse_val = float(mse_match.group(1).replace(",", "."))
                mses.append((step, mse_val))

    return steps, accuracies, mses


def print_summary(steps, accuracies, mses, file_path: Path, raw_text: str):
    print("\n" + "=" * 55)
    print("RIEPILOGO COMPLETO RISULTATI RETE NEURALE 60k image")
    print("=" * 55)
    print(f"File log                 : {file_path}")

    if not accuracies:
        print("Nessuna metrica riconosciuta nel file.")
        print("Esempio formato atteso: 'Immagini: 500 | Accuratezza: 98.6%'")
        sample_lines = [line for line in raw_text.replace("\x00", "").splitlines() if line.strip()][:5]
        print("Prime righe lette nel file:")
        for s in sample_lines:
            print(f"  {repr(s)}")
        print("=" * 55 + "\n")
        return

    print(f"Punti rilevati           : {len(accuracies)}")
    print(f"Range immagini/step      : {steps[0]} -> {steps[-1]}")
    print(f"Accuratezza iniziale     : {accuracies[0]:.4f}% (a {steps[0]})")
    print(f"Accuratezza finale       : {accuracies[-1]:.4f}% (a {steps[-1]})")

    max_acc = max(accuracies)
    min_acc = min(accuracies)
    best_step = steps[accuracies.index(max_acc)]
    worst_step = steps[accuracies.index(min_acc)]
    avg_acc = sum(accuracies) / len(accuracies)
    delta_acc = accuracies[-1] - accuracies[0]

    print(f"Accuratezza massima      : {max_acc:.4f}% (a {best_step})")
    print(f"Accuratezza minima       : {min_acc:.4f}% (a {worst_step})")
    print(f"Accuratezza media        : {avg_acc:.4f}%")
    print(f"Variazione (Inizio->Fine): {delta_acc:+.4f}%")

    if mses:
        mse_vals = [m[1] for m in mses]
        print("-" * 55)
        print(f"MSE iniziale             : {mse_vals[0]:.6f}")
        print(f"MSE finale               : {mse_vals[-1]:.6f}")
        print(f"MSE minimo               : {min(mse_vals):.6f}")
        print(f"MSE massimo              : {max(mse_vals):.6f}")
        print(f"MSE medio                : {sum(mse_vals) / len(mse_vals):.6f}")

    print("-" * 55)
    print("Tutti i punti estratti:")
    mse_dict = dict(mses)
    for s, a in zip(steps, accuracies):
        if s in mse_dict:
            print(f"  - Immagini/Step {s:>6}: Accuratezza = {a:.4f}% | MSE = {mse_dict[s]:.6f}")
        else:
            print(f"  - Immagini/Step {s:>6}: Accuratezza = {a:.4f}%")

    print("=" * 55 + "\n")


def plot_results(steps, accuracies, mses, output_path: Path):
    if not accuracies:
        return

    if mses:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    else:
        fig, ax1 = plt.subplots(1, 1, figsize=(10, 5))
        ax2 = None

    ax1.plot(steps, accuracies, marker="o", markersize=3, linewidth=1.8, label="Accuratezza (%)")
    ax1.set_title("Andamento Prestazioni Rete Neurale")
    ax1.set_ylabel("Accuratezza (%)")
    ax1.set_ylim(0, 100)
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend()

    if ax2 is not None:
        mse_steps = [m[0] for m in mses]
        mse_vals = [m[1] for m in mses]
        ax2.plot(mse_steps, mse_vals, marker="s", markersize=3, linewidth=1.8, label="MSE / Loss")
        ax2.set_xlabel("Immagini / Step")
        ax2.set_ylabel("Errore (MSE)")
        ax2.grid(True, linestyle="--", alpha=0.5)
        ax2.legend()
    else:
        ax1.set_xlabel("Immagini / Step")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"Grafico salvato in: {output_path}")
    plt.show()


def main():
    log_path = Path(LOG_FILE_PATH)
    if not log_path.exists() or not log_path.is_file():
        print(f"Errore: file non trovato -> {log_path}")
        return

    text = read_log_text(log_path)
    steps, accuracies, mses = parse_logs(text)

    print_summary(steps, accuracies, mses, log_path, text)
    output_img = log_path.with_suffix(".png")
    plot_results(steps, accuracies, mses, output_img)


if __name__ == "__main__":
    main()