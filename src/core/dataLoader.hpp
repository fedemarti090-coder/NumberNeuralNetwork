#include <algorithm>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

class DataLoader
{
private:


    std::string dataPath;
    std::string labelsPath;

    static uint32_t readBigEndianU32(std::ifstream &file)
    {
        uint8_t b[4];
        file.read(reinterpret_cast<char *>(b), 4);

        return (static_cast<uint32_t>(b[0]) << 24) | (static_cast<uint32_t>(b[1]) << 16) | (static_cast<uint32_t>(b[2]) << 8) | static_cast<uint32_t>(b[3]);
    }

public:
    void init(bool large)
    {
        if (!large)
        {
            dataPath = "../data/train-test/t10k-images.idx3-ubyte";
            labelsPath = "../data/train-test/t10k-labels.idx1-ubyte";
        }
        else
        {
            dataPath = "../data/train/train-images.idx3-ubyte";
            labelsPath = "../data/train/train-labels.idx1-ubyte";
        }
    }

    std::vector<float> getNormalizedData()
    {
        std::ifstream file(dataPath, std::ios::binary);
        if (!file.is_open())
        {
            std::cout << "File vuoto \n";
            return {};
        }

        uint32_t magic = readBigEndianU32(file);
        uint32_t numImage = readBigEndianU32(file);
        uint32_t rows = readBigEndianU32(file);
        uint32_t cols = readBigEndianU32(file);

        size_t totalPixels = static_cast<size_t>(numImage) * rows * cols;
        std::vector<uint8_t> rawPixel(totalPixels);
        file.read(reinterpret_cast<char *>(rawPixel.data()), totalPixels);

        std::vector<float> normalizedData(totalPixels);
        for (size_t i = 0; i < totalPixels; ++i)
        {
            normalizedData[i] = static_cast<float>(rawPixel[i]) / 255.0f; // Normalizzo i pixel
        }

        return normalizedData;
    }

    std::vector<uint8_t> getLabels()
    {
        std::ifstream file(labelsPath, std::ios::binary);
        if (!file.is_open())
        {
            std::cout << "File etichette non aperto\n";
            return {};
        }

        uint32_t magic = readBigEndianU32(file);
        uint32_t numItems = readBigEndianU32(file);
        (void)magic;

        if (!file || numItems == 0)
        {
            std::cout << "Header etichette non valido\n";
            return {};
        }

        std::vector<uint8_t> labels(numItems);
        file.read(reinterpret_cast<char *>(labels.data()), numItems);

        if (file.gcount() != static_cast<std::streamsize>(numItems))
        {
            std::cout << "Lettura etichette incompleta\n";
            return {};
        }

        return labels;
    }

    void stampaData(const std::vector<float> &f, std::size_t start = 0, std::size_t count = 784, std::size_t cols = 28)
    {
        if (start >= f.size())
        {
            return;
        }

        std::size_t end = std::min(f.size(), start + count);
        for (std::size_t i = start; i < end; ++i)
        {
            std::cout << f[i] << " ";
            if (cols > 0 && ((i - start + 1) % cols == 0))
            {
                std::cout << "\n";
            }
        }
        std::cout << "\n";
    }
};