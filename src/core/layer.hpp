#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <algorithm>

class LayerHandler
{
private:
    // C
    static constexpr int c_filter_size = 120 * 16 * 4 * 4;
    std::vector<float> c_filters;

    static constexpr int c_bias_size = 120;
    std::vector<float> c_biases;

    // S
    static constexpr int s_bias_size = 16;
    std::vector<float> s_bias;

    // F
    static constexpr int f6_in = 120;
    static constexpr int f6_out = 84;
    std::vector<float> f6_weight;
    std::vector<float> f6_bias;

    // Out
    static constexpr int out_in = 84;
    static constexpr int out_size = 10;
    std::vector<float> out_weight;
    std::vector<float> out_bias;

    static inline float activate(float x)
    {
        return 1.7159f * std::tanh((2.0f / 3.0f) * x);
    }

    void fillRandom(std::vector<float> &vec, std::mt19937 &gen, float limit)
    {
        std::uniform_real_distribution<float> dist(-limit, limit);
        for (float &val : vec)
        {
            val = dist(gen);
        }
    }

public:
    LayerHandler()
        : c_filters(c_filter_size, 0.0f),
          c_biases(c_bias_size, 0.0f),
          s_bias(s_bias_size, 0.0f),
          f6_weight(f6_in * f6_out, 0.0f),
          f6_bias(f6_out, 0.0f),
          out_weight(out_in * out_size, 0.0f),
          out_bias(out_size, 0.0f) {}

    void randomizeWeight(float limit = 0.1f)
    {
        if (limit < 0.0f)
        {
            limit = -limit;
        }

        std::random_device rd;
        std::mt19937 gen(rd());

        fillRandom(c_filters, gen, limit);
        fillRandom(c_biases, gen, limit);
        fillRandom(s_bias, gen, limit);
        fillRandom(f6_weight, gen, limit);
        fillRandom(f6_bias, gen, limit);
        fillRandom(out_weight, gen, limit);
        fillRandom(out_bias, gen, limit);
    }

    // Convoluzione
    std::vector<float> C(const std::vector<float> &In, int in_channels, int W, int size, int num_filters)
    {
        int kernel_elems = size * size;
        int out_W = W - size + 1;
        int in_map_cells = W * W;
        int out_map_cells = out_W * out_W;
        int required_weights = num_filters * in_channels * kernel_elems;

        if (in_channels <= 0 || W < size || size <= 0 || num_filters <= 0 || out_W <= 0 ||
            static_cast<int>(In.size()) < in_channels * in_map_cells ||
            static_cast<int>(c_biases.size()) < num_filters ||
            static_cast<int>(c_filters.size()) < required_weights)
        {
            return {};
        }

        std::vector<float> Out(num_filters * out_map_cells, 0.0f);

        for (int f = 0; f < num_filters; ++f)
        {
            int out_offset = f * out_map_cells;

            for (int cell = 0; cell < out_map_cells; ++cell)
            {
                int base_r = cell / out_W;
                int base_c = cell % out_W;
                float sum = 0.0f;

                for (int ch = 0; ch < in_channels; ++ch)
                {
                    int in_ch_offset = ch * in_map_cells;
                    int w_ch_offset = ((f * in_channels) + ch) * kernel_elems;

                    for (int i = 0; i < kernel_elems; ++i)
                    {
                        int r = base_r + (i / size);
                        int c = base_c + (i % size);
                        sum += In[in_ch_offset + r * W + c] * c_filters[w_ch_offset + i];
                    }
                }

                sum += c_biases[f];
                Out[out_offset + cell] = activate(sum);
            }
        }

        return Out;
    }

    // Sub Sampling
    std::vector<float> S(const std::vector<float> &In, int layer, int channels, int W, int size)
    {
        if (size <= 0 || channels <= 0 || W < size || (W % size) != 0 || layer < 0 ||
            static_cast<int>(In.size()) < channels * W * W ||
            layer >= static_cast<int>(s_bias.size()))
        {
            return {};
        }

        int out_W = W / size;
        int in_map_cells = W * W;
        int out_map_cells = out_W * out_W;
        int pool_area = size * size;

        std::vector<float> Out(channels * out_map_cells, 0.0f);

        for (int ch = 0; ch < channels; ++ch)
        {
            int in_ch_offset = ch * in_map_cells;
            int out_ch_offset = ch * out_map_cells;

            for (int cell = 0; cell < out_map_cells; ++cell)
            {
                int base_r = (cell / out_W) * size;
                int base_c = (cell % out_W) * size;
                float sum = 0.0f;

                for (int i = 0; i < pool_area; ++i)
                {
                    int r = base_r + (i / size);
                    int c = base_c + (i % size);
                    sum += In[in_ch_offset + r * W + c];
                }

                float avg = sum / static_cast<float>(pool_area);
                Out[out_ch_offset + cell] = activate(avg + s_bias[layer]);
            }
        }

        return Out;
    }

    // Fully connected classico
    std::vector<float> FC(const std::vector<float> &In, const std::vector<float> &weights, const std::vector<float> &biases, int in_dim, int out_dim)
    {
        if (in_dim <= 0 || out_dim <= 0 ||
            static_cast<int>(In.size()) < in_dim ||
            static_cast<int>(biases.size()) < out_dim ||
            static_cast<int>(weights.size()) < in_dim * out_dim)
        {
            return {};
        }

        std::vector<float> Out(out_dim, 0.0f);
        for (int o = 0; o < out_dim; ++o)
        {
            float sum = 0.0f;
            int w_offset = o * in_dim;
            for (int i = 0; i < in_dim; ++i)
            {
                sum += In[i] * weights[w_offset + i];
            }
            sum += biases[o];
            Out[o] = activate(sum);
        }
        return Out;
    }

    // F6
    std::vector<float> F6(const std::vector<float> &In)
    {
        return FC(In, f6_weight, f6_bias, f6_in, f6_out);
    }

    // output
    std::vector<float> Output(const std::vector<float> &In)
    {
        return FC(In, out_weight, out_bias, out_in, out_size);
    }

    void stampaLayer(std::vector<float> &Layer, int size)
    {
        int limite = std::min(size, static_cast<int>(Layer.size()));

        for (int i = 0; i < limite; ++i)
        {
            std::cout << Layer[i] << " ";
        }
        std::cout << '\n';
    }
};
