#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <algorithm>
#include <fstream>

class LayerHandler
{
private:
    bool train = false;

    // C
    static constexpr int c1_w_size = 6 * 1 * 5 * 5;    // 150
    static constexpr int c3_w_size = 16 * 6 * 5 * 5;   // 2400
    static constexpr int c5_w_size = 120 * 16 * 4 * 4; // 30720

    static constexpr int c1_w_offset = 0;
    static constexpr int c3_w_offset = c1_w_offset + c1_w_size;   // 150
    static constexpr int c5_w_offset = c3_w_offset + c3_w_size;   // 2550
    static constexpr int c_filter_size = c5_w_offset + c5_w_size; // 33270

    static constexpr int c1_b_size = 6;
    static constexpr int c3_b_size = 16;
    static constexpr int c5_b_size = 120;

    static constexpr int c1_b_offset = 0;
    static constexpr int c3_b_offset = c1_b_offset + c1_b_size; // 6
    static constexpr int c5_b_offset = c3_b_offset + c3_b_size; // 22
    static constexpr int c_bias_size = c5_b_offset + c5_b_size; // 142

    std::vector<float> c_filters;
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
    std::vector<float> C(const std::vector<float> &In, int in_channels, int W, int size, int num_filters, int w_offset, int b_offset)
    {
        int kernel_elems = size * size;
        int out_W = W - size + 1;
        int in_map_cells = W * W;
        int out_map_cells = out_W * out_W;
        int required_weights = num_filters * in_channels * kernel_elems;

        if (in_channels <= 0 || W < size || size <= 0 || num_filters <= 0 || out_W <= 0 || w_offset < 0 || b_offset < 0 || static_cast<int>(In.size()) < in_channels * in_map_cells || static_cast<int>(c_biases.size()) < b_offset + num_filters || static_cast<int>(c_filters.size()) < w_offset + required_weights)
        {
            return {};
        }

        std::vector<float> Out(num_filters * out_map_cells, 0.0f);

        for (int f = 0; f < num_filters; ++f)
        {
            int out_offset = f * out_map_cells;
            float bias_val = c_biases[b_offset + f];

            for (int cell = 0; cell < out_map_cells; ++cell)
            {
                int base_r = cell / out_W;
                int base_c = cell % out_W;
                float sum = 0.0f;

                for (int ch = 0; ch < in_channels; ++ch)
                {
                    int in_ch_offset = ch * in_map_cells;
                    int filter_ch_offset = w_offset + ((f * in_channels) + ch) * kernel_elems;

                    for (int i = 0; i < kernel_elems; ++i)
                    {
                        int r = base_r + (i / size);
                        int c = base_c + (i % size);
                        sum += In[in_ch_offset + r * W + c] * c_filters[filter_ch_offset + i];
                    }
                }

                sum += bias_val;
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

    // Fully connected classic
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
    std::vector<float> F6_(const std::vector<float> &In)
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

private:
    float n = 0.005f;

    static int argmax(const std::vector<float> &v)
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

    static inline float deActivateFromY(float y)
    {
        constexpr float A = 1.7159f;
        constexpr float B = 2.0f / 3.0f;
        float ratio = y / A;
        return A * B * (1.0f - ratio * ratio);
    }

    std::vector<float> backpropOutput(const std::vector<float> &F6, const std::vector<float> &Out, const std::vector<float> &target)
    {
        std::vector<float> delta_out(out_size, 0.0f);
        std::vector<float> grad_f6(out_in, 0.0f);

        for (int o = 0; o < out_size; ++o)
        {
            float err = Out[o] - target[o];
            delta_out[o] = err * deActivateFromY(Out[o]);

            int w_offset = o * out_in;
            for (int i = 0; i < out_in; ++i)
            {
                int idx = w_offset + i;
                grad_f6[i] += delta_out[o] * out_weight[idx];
                out_weight[idx] -= n * delta_out[o] * F6[i];
            }

            out_bias[o] -= n * delta_out[o];
        }

        return grad_f6;
    }

    std::vector<float> backpropF6(const std::vector<float> &C5, const std::vector<float> &F6, const std::vector<float> &grad_f6)
    {
        std::vector<float> grad_c5(f6_in, 0.0f);

        for (int j = 0; j < f6_out; ++j)
        {
            float delta_f6 = grad_f6[j] * deActivateFromY(F6[j]);

            int w_offset = j * f6_in;
            for (int c = 0; c < f6_in; ++c)
            {
                int idx = w_offset + c;
                grad_c5[c] += delta_f6 * f6_weight[idx];
                f6_weight[idx] -= n * delta_f6 * C5[c];
            }

            f6_bias[j] -= n * delta_f6;
        }

        return grad_c5;
    }

    std::vector<float> backpropC5(const std::vector<float> &S4, const std::vector<float> &C5, const std::vector<float> &grad_c5)
    {
        constexpr int in_channels = 16;
        constexpr int W = 4;
        constexpr int kernel_elems = W * W;
        constexpr int map_cells = W * W;
        constexpr int num_filters = 120;

        std::vector<float> grad_s4(in_channels * map_cells, 0.0f);

        for (int f = 0; f < num_filters; ++f)
        {
            float delta_c5 = grad_c5[f] * deActivateFromY(C5[f]);
            int w_filter_offset = f * in_channels * kernel_elems;

            for (int ch = 0; ch < in_channels; ++ch)
            {
                int in_ch_offset = ch * map_cells;
                int w_ch_offset = w_filter_offset + ch * kernel_elems;

                for (int i = 0; i < kernel_elems; ++i)
                {
                    int w_idx = w_ch_offset + i;
                    int in_idx = in_ch_offset + i;

                    grad_s4[in_idx] += delta_c5 * c_filters[w_idx];
                    c_filters[w_idx] -= n * delta_c5 * S4[in_idx];
                }
            }

            c_biases[f] -= n * delta_c5;
        }

        return grad_s4;
    }

    std::vector<float> backpropS(const std::vector<float> &S_out, const std::vector<float> &grad_s_out, int layer, int channels, int W, int size)
    {
        int out_W = W / size;
        int in_map_cells = W * W;
        int out_map_cells = out_W * out_W;
        int pool_area = size * size;

        std::vector<float> grad_in(channels * in_map_cells, 0.0f);

        float grad_bias = 0.0f;
        for (int ch = 0; ch < channels; ++ch)
        {
            int in_ch_offset = ch * in_map_cells;
            int out_ch_offset = ch * out_map_cells;

            for (int cell = 0; cell < out_map_cells; ++cell)
            {
                int out_idx = out_ch_offset + cell;
                float delta_s = grad_s_out[out_idx] * deActivateFromY(S_out[out_idx]);

                grad_bias += delta_s;

                int base_r = (cell / out_W) * size;
                int base_c = (cell % out_W) * size;
                float shared_grad = delta_s / static_cast<float>(pool_area);

                for (int i = 0; i < pool_area; ++i)
                {
                    int r = base_r + (i / size);
                    int c = base_c + (i % size);
                    grad_in[in_ch_offset + r * W + c] += shared_grad;
                }
            }
        }

        s_bias[layer] -= n * grad_bias;
        return grad_in;
    }

    std::vector<float> backpropC(const std::vector<float> &In, const std::vector<float> &C_out, const std::vector<float> &grad_c_out, int in_channels, int W, int size, int num_filters, int w_offset, int b_offset)
    {
        int kernel_elems = size * size;
        int out_W = W - size + 1;
        int in_map_cells = W * W;
        int out_map_cells = out_W * out_W;

        std::vector<float> grad_in(in_channels * in_map_cells, 0.0f);
        for (int f = 0; f < num_filters; ++f)
        {
            int out_offset = f * out_map_cells;
            float grad_bias = 0.0f;

            for (int cell = 0; cell < out_map_cells; ++cell)
            {
                int out_idx = out_offset + cell;
                float delta = grad_c_out[out_idx] * deActivateFromY(C_out[out_idx]);
                grad_bias += delta;

                int base_r = cell / out_W;
                int base_c = cell % out_W;

                for (int ch = 0; ch < in_channels; ++ch)
                {
                    int in_ch_offset = ch * in_map_cells;
                    int filter_ch_offset = w_offset + ((f * in_channels) + ch) * kernel_elems;

                    for (int i = 0; i < kernel_elems; ++i)
                    {
                        int r = base_r + (i / size);
                        int c = base_c + (i % size);
                        int in_idx = in_ch_offset + r * W + c;
                        int w_idx = filter_ch_offset + i;

                        grad_in[in_idx] += delta * c_filters[w_idx];
                        c_filters[w_idx] -= n * delta * In[in_idx];
                    }
                }
            }

            c_biases[b_offset + f] -= n * grad_bias;
        }

        return grad_in;
    }

public:
    std::vector<float> predictWithProbs(const std::vector<float> &input)
    {
        std::vector<float> probs(out_size, 0.0f);
        if (input.size() != 784)
        {
            return probs;
        }

        std::vector<float> C1 = C(input, 1, 28, 5, 6, c1_w_offset, c1_b_offset);
        std::vector<float> S2 = S(C1, 0, 6, 24, 2);
        std::vector<float> C3 = C(S2, 6, 12, 5, 16, c3_w_offset, c3_b_offset);
        std::vector<float> S4 = S(C3, 1, 16, 8, 2);
        std::vector<float> C5 = C(S4, 16, 4, 4, 120, c5_w_offset, c5_b_offset);
        std::vector<float> F6 = F6_(C5);
        std::vector<float> Out = Output(F6);

        if (Out.size() != out_size)
        {
            return probs;
        }

        float maxValue = *std::max_element(Out.begin(), Out.end());
        float sum = 0.0f;
        for (int i = 0; i < out_size; ++i)
        {
            probs[i] = std::exp(Out[i] - maxValue);
            sum += probs[i];
        }

        if (sum > 0.0f)
        {
            for (float &prob : probs)
            {
                prob /= sum;
            }
        }

        return probs;
    }

    int predict(const std::vector<float> &input)
    {
        std::vector<float> probs = predictWithProbs(input);
        if (probs.size() != out_size)
        {
            return -1;
        }

        return argmax(probs);
    }

    bool elabImg(std::vector<float> Input, int ans)
    {
        /*
        execution:
        C1 -> S2 -> C3 -> S4 -> C5 -> F6 -> out

        backpropagation:
        C1 <- S2 <- C3 <- S4 <- C5 <- F6 <- out

        */
        std::vector<float> C1 = C(Input, 1, 28, 5, 6, c1_w_offset, c1_b_offset);
        std::vector<float> S2 = S(C1, 0, 6, 24, 2);
        std::vector<float> C3 = C(S2, 6, 12, 5, 16, c3_w_offset, c3_b_offset);
        std::vector<float> S4 = S(C3, 1, 16, 8, 2);
        std::vector<float> C5 = C(S4, 16, 4, 4, 120, c5_w_offset, c5_b_offset);
        std::vector<float> F6 = F6_(C5);
        std::vector<float> Out = Output(F6);

        int pred = argmax(Out);
        std::vector<float> target = getAnswerVector(ans);

        if (!train)
        {
            return (pred == ans);
        }

        float mse = 0.0f;
        for (int i = 0; i < out_size; ++i)
        {
            float diff = Out[i] - target[i];
            mse += 0.5f * diff * diff;
        }
        // std::cout << "Prevista: " << pred << " | Effettiva: " << ans << " | MSE: " << mse << std::endl;

        std::vector<float> grad_F6 = backpropOutput(F6, Out, target);
        std::vector<float> grad_C5 = backpropF6(C5, F6, grad_F6);
        std::vector<float> grad_S4 = backpropC(S4, C5, grad_C5, 16, 4, 4, 120, c5_w_offset, c5_b_offset);
        std::vector<float> grad_C3 = backpropS(S4, grad_S4, 1, 16, 8, 2);
        std::vector<float> grad_S2 = backpropC(S2, C3, grad_C3, 6, 12, 5, 16, c3_w_offset, c3_b_offset);
        std::vector<float> grad_C1 = backpropS(C1, grad_S2, 0, 6, 24, 2);
        backpropC(Input, C1, grad_C1, 1, 28, 5, 6, c1_w_offset, c1_b_offset);

        return (pred == ans);
    }

    bool saveWeights(const std::string &filePath) const
    {
        if (!train)
            return true;
        std::ofstream out(filePath, std::ios::binary);
        if (!out.is_open())
        {
            return false;
        }

        const uint32_t magic = 0x4C4E4535; // "LNE5"
        out.write(reinterpret_cast<const char *>(&magic), sizeof(magic));

        auto writeVec = [&out](const std::vector<float> &vec)
        {
            out.write(reinterpret_cast<const char *>(vec.data()), static_cast<std::streamsize>(vec.size() * sizeof(float)));
        };

        // salvo i vettori sul bin
        writeVec(c_filters);
        writeVec(c_biases);
        writeVec(s_bias);
        writeVec(f6_weight);
        writeVec(f6_bias);
        writeVec(out_weight);
        writeVec(out_bias);

        return out.good();
    }

    bool loadWeights(const std::string filePath)
    {
        std::ifstream in(filePath, std::ios::binary);
        if (!in.is_open())
        {
            return false;
        }

        uint32_t magic = 0;
        in.read(reinterpret_cast<char *>(&magic), sizeof(magic));
        if (!in.good() || magic != 0x4C4E4535)
        {
            return false;
        }

        auto readVec = [&in](std::vector<float> &vec) -> bool
        {
            in.read(reinterpret_cast<char *>(vec.data()),
                    static_cast<std::streamsize>(vec.size() * sizeof(float)));
            return in.good();
        };

        return readVec(c_filters) && readVec(c_biases) && readVec(s_bias) && readVec(f6_weight) && readVec(f6_bias) && readVec(out_weight) && readVec(out_bias);
    }
};
