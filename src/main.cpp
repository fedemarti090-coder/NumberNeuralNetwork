#include <iostream>
#include <vector>
#include <algorithm>

#include "core/dataLoader.hpp"
#include "core/layer.hpp"

int argmax(const std::vector<float> &v)
{
    if (v.empty())
        return -1;
    auto it = std::max_element(v.begin(), v.end());
    return static_cast<int>(std::distance(v.begin(), it));
}

std::vector<float> getAnswerVector(int ans)
{
    std::vector<float> correctAns(10, 0.0f);
    for (int i = 0; i < 10; i++)
    {
        if (i == ans)
        {
            correctAns[i] = 1.0f;
        }
        else
        {
            correctAns[i] = -1.0f;
        }
    }

    return correctAns;
}

bool elabImg(std::vector<float> Input, LayerHandler &layerHandler, int ans)
{
    /*
    C1 -> S2 -> C3 -> S4 -> C5 -> F6 -> out
    */
    std::vector<float> C1 = layerHandler.C(Input, 1, 28, 5, 6);
    std::vector<float> S2 = layerHandler.S(C1, 0, 6, 24, 2);
    std::vector<float> C3 = layerHandler.C(S2, 6, 12, 5, 16);
    std::vector<float> S4 = layerHandler.S(C3, 1, 16, 8, 2);
    std::vector<float> C5 = layerHandler.C(S4, 16, 4, 4, 120);
    std::vector<float> F6 = layerHandler.F6(C5);
    std::vector<float> Out = layerHandler.Output(F6);

    std::cout << "Cifra prevista (pesi random): " << argmax(Out) << " Cifra effettiva: " << ans << std::endl;

    std::vector<float> ansVector = getAnswerVector(ans);

    std::vector<float> delta;
    for (int i = 0; i < 10; i++)
    {
        delta[i] = Out[i] - ansVector[i];
    }

    return false;
}

int main()
{

    DataLoader dataLoader;

    std::vector<float> dataRaw = dataLoader.getNormalizedData();
    std::vector<uint8_t> labels = dataLoader.getLabels();

    // dataLoader.stampaData(dataRaw);

    std::vector<float> Input(dataRaw.begin(), dataRaw.begin() + 784);

    LayerHandler layerHandler;
    layerHandler.randomizeWeight(); // randomizzazione dei weight

    elabImg(Input, layerHandler, labels[0]);

    return 0;
}