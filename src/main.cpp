#include <iostream>
#include <vector>
#include <algorithm>
#include <random>

#include "core/dataLoader.hpp"
#include "core/layer.hpp"

int main()
{

    DataLoader dataLoader;
    dataLoader.init(false);

    std::vector<float> dataRaw = dataLoader.getNormalizedData();
    std::vector<uint8_t> labels = dataLoader.getLabels();

    // dataLoader.stampaData(dataRaw);

    LayerHandler net;
    std::string weightsFile = "model/lenet5_weights.bin";

    if (net.loadWeights(weightsFile))
    {
        std::cout << "Pesi caricati da " << weightsFile << '\n';
    }
    else
    {
        std::cout << "Nessun file trovato, inizializzo pesi casuali...\n";
        net.randomizeWeight(0.1f);
    }

    int correct = 0;
    const int totalImages = std::min({60000, static_cast<int>(dataRaw.size() / 784), static_cast<int>(labels.size())});

    for (int i = 0; i < totalImages; ++i)
    {
        std::vector<float> Input(dataRaw.begin() + 784 * i, dataRaw.begin() + 784 * (i + 1));
        int ans = static_cast<int>(labels[i]);

        if (net.elabImg(Input, ans))
        {
            ++correct;
        }

        if ((i + 1) % 500 == 0)
        {
            float acc = (100.0f * correct) / static_cast<float>(i + 1);
            std::cout << "Immagini: " << (i + 1) << " | Accuratezza: " << acc << "%\n";
        }
    }

    if (net.saveWeights(weightsFile))
    {
        std::cout << "Pesi salvati in " << weightsFile << '\n';
    }
    else
    {
        std::cerr << "Errore nel salvataggio dei pesi!\n";
    }

    return 0;
}