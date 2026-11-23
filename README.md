<div align="center">

# LeNet Neural Network

### Dalle convoluzioni in C++ al riconoscimento di cifre in tempo reale

Una rete neurale **ispirata a LeNet-5**, implementata in C++ senza framework di machine learning, compilata in WebAssembly e integrata in una piccola app desktop con Electron.

**C++17** · **MNIST** · **WebAssembly** · **Electron**

[Architettura](#architettura) · [Avvio rapido](#avvio-rapido) · [Come funziona](#come-funziona) · [Struttura](#struttura-del-progetto)

</div>

---

## Il progetto

Questo progetto esplora l'intero percorso di una rete neurale: operazioni sui tensori e backpropagation in C++, inferenza sui pesi salvati e una UI desktop per disegnare una cifra e vedere come il modello la classifica.

![LeNet Studio: disegno di una cifra e distribuzione delle probabilità](./img/img-4.png)

*Disegna una cifra sulla tavoletta: il modello aggiorna la previsione e le probabilità per le classi da 0 a 9.*

## Architettura

Il flusso di inferenza è:

```text
28×28 input → C1 → S2 → C3 → S4 → C5 → F6 → output (10 classi)
```

![Diagramma dell'architettura della rete](./img/nn.svg)

| Layer | Operazione | Dimensione in uscita |
|---|---|---:|
| Input | Immagine in scala di grigi | $1 \times 28 \times 28$ |
| C1 | Convoluzione $5 \times 5$, 6 filtri | $6 \times 24 \times 24$ |
| S2 | Average pooling $2 \times 2$ | $6 \times 12 \times 12$ |
| C3 | Convoluzione $5 \times 5$, 16 filtri | $16 \times 8 \times 8$ |
| S4 | Average pooling $2 \times 2$ | $16 \times 4 \times 4$ |
| C5 | Convoluzione $4 \times 4$, 120 filtri | $120 \times 1 \times 1$ |
| F6 | Fully connected | $84$ |
| Output | Fully connected, una unità per cifra | $10$ |

Il progetto usa la funzione di attivazione scalata di LeCun et al. (1998):

$$f(x) = 1.7159 \cdot \tanh\left(\frac{2}{3} x\right)$$

Per visualizzare la distribuzione sulle dieci classi, i punteggi di uscita $z_i$ (per $i \in \{0, \dots, 9\}$) vengono convertiti in probabilità tramite softmax:

$$\sigma(z)_i = \frac{e^{z_i}}{\sum_{j=0}^{9} e^{z_j}}$$

## Avvio rapido

### App desktop

Per compilare il bridge C++ in WebAssembly serve **Emscripten**; per avviare l'interfaccia servono **Node.js e npm**.

1. Attiva l'ambiente Emscripten con `emsdk_env.bat` dal tuo SDK.
2. Dalla cartella principale del repository, compila il modulo:

   ```bat
   build-wasm.bat
   ```

3. Installa le dipendenze e avvia Electron:

   ```bat
   cd app
   npm install
   npm start
   ```

Lo script genera `wasm/lenet5.js`, `wasm/lenet5.wasm` e `wasm/lenet5.data`. I pesi `src/model/lenet5_weights.bin` vengono inclusi nel pacchetto di dati e caricati all'avvio.

Nell'app puoi disegnare con il mouse o il touch: l'immagine viene ritagliata, centrata e normalizzata per il modello; la previsione e le probabilità delle cifre si aggiornano mentre scrivi.

### Valutazione nativa C++

Con un compilatore C++17, dalla cartella principale:

```bat
g++ -std=c++17 -O3 -Isrc src/main.cpp -o src/build/main.exe
cd src
build\main.exe
```

Il programma legge il set MNIST di test da `data/train-test/` e carica i pesi da `src/model/lenet5_weights.bin`. Eseguilo da `src/`, perché i percorsi dei dati e dei pesi sono relativi alla directory corrente.

> **Nota:** allo stato attuale `src/main.cpp` esegue la valutazione in modalità inferenza. Il codice di backpropagation è presente, ma il flag `train` di `LayerHandler` è disattivato per impostazione predefinita: l'esecuzione di questo comando non allena la rete né aggiorna i pesi.

## Come funziona

### Convoluzione

Ogni filtro scorre sulle mappe d'ingresso, calcola una somma pesata con bias e applica la funzione di attivazione. Dato un ingresso $X$, un kernel $K$ di dimensione $k_h \times k_w$ e un bias $b$, l'attivazione in posizione $(m, n)$ della feature map è:

$$Y_{m,n} = f\left(\sum_{i=0}^{k_h - 1} \sum_{j=0}^{k_w - 1} X_{m+i, n+j} \cdot K_{i,j} + b\right)$$

I layer C1, C3 e C5 imparano rispettivamente $6$, $16$ e $120$ filtri.

### Subsampling

S2 e S4 riducono la risoluzione con average pooling $2 \times 2$. Ogni finestra $2 \times 2$ viene mediata e passata all'attivazione, riducendo la dimensione spaziale di un fattore $2$ lungo ciascun asse e il numero di valori da elaborare:

$$P_{u,v} = f\left(\frac{1}{4}\sum_{a=0}^{1}\sum_{b=0}^{1} X_{2u+a,\, 2v+b}\right)$$

### Fully connected

Le $120$ attivazioni di C5 vengono collegate a $84$ neuroni nel layer F6, poi ai $10$ neuroni di output (uno per ciascuna cifra), calcolando per ciascun neurone $y_k = f\left(\sum_j w_{kj} x_j + b_k\right)$.

### Pesi

I parametri della rete sono salvati in formato binario nel file `src/model/lenet5_weights.bin` e caricati sia dall'app WASM sia dal programma C++.

## Struttura del progetto

```text
.
├── app/
│   ├── main.js                 # Processo principale Electron e bridge IPC
│   └── public/                 # Interfaccia per disegnare e visualizzare i risultati
├── data/
│   ├── log/                    # Log e analisi delle prestazioni
│   ├── train/                  # Dati MNIST di training
│   └── train-test/             # Dati MNIST di test
├── img/                        # Screenshot e diagramma della rete
├── src/
│   ├── core/
│   │   ├── dataLoader.hpp      # Lettura e normalizzazione MNIST
│   │   └── layer.hpp           # Layer, inferenza e backpropagation
│   ├── model/
│   │   └── lenet5_weights.bin  # Pesi salvati
│   ├── WASM/
│   │   └── wasm_bridge.cpp     # Funzioni esportate verso WebAssembly
│   └── main.cpp                # Runner C++ per il set di test
└── build-wasm.bat              # Build Emscripten per Windows
```

## Riferimento

Architettura e funzione di attivazione prendono ispirazione dal lavoro:

> Y. LeCun, L. Bottou, Y. Bengio, P. Haffner, *Gradient-Based Learning Applied to Document Recognition*, Proceedings of the IEEE, 1998.

Il PDF è disponibile in `data/lecun-98.pdf`.

## Test di efficienza

Elaborando il dataset da $10.000$ immagini di cifre manoscritte (*hand-written digits*), l'accuracy del modello si attesta intorno al **97% – 97.5%**.

<div align="center">
  <img src="./img/img-2.png" alt="Risultati del test su 10.000 immagini" />
</div>

Testando invece il dataset da $60.000$ immagini, l'accuracy va dal **97.5%** fino a un massimo del **97.9%**.

<div align="center">
  <img src="./img/img-3.png" alt="Risultati del test su 60.000 immagini" />
</div>

| Dataset | Campioni | Accuracy |
|---|---:|---:|
| Test cifre manoscritte | $10.000$ | $97.0\% – 97.5\%$ |
| Valutazione estesa | $60.000$ | $97.5\% – 97.9\%$ |

Per analisi dettagliate sulle prestazioni è possibile consultare i log in `data/log/`.
