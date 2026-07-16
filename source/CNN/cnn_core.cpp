#include "cnn_core.hpp"

#include <cmath>
#include <random>
#include <chrono>
#include <algorithm>

void normalize_input (
    const Array<float, 4> &input,
          Array<float, 4> &output
)
{
    if (!(input.extents() == output.extents()))
    {
        throw_with_context<std::domain_error>("Domain error. Check source location.");
    }

    float n = input.extents(1) * input.extents(2) * input.extents(3);

    for(int64_t i = 0; i < input.extents(0); i++)
    {
        float m = 0.0;
        float v = 0.0;

        for(int64_t j = 0; j < input.extents(1); j++)
        for(int64_t k = 0; k < input.extents(2); k++)
        for(int64_t l = 0; l < input.extents(3); l++)
        {
            m += input[i, j, k, l];
        }

        m /= n;

        for(int64_t j = 0; j < input.extents(1); j++)
        for(int64_t k = 0; k < input.extents(2); k++)
        for(int64_t l = 0; l < input.extents(3); l++)
        {
            output[i, j, k, l] = input[i, j, k, l] - m;
            v += (input[i, j, k, l] - m) *  (input[i, j, k, l] - m);
        }

        v /= n;

        for(int64_t j = 0; j < input.extents(1); j++)
        for(int64_t k = 0; k < input.extents(2); k++)
        for(int64_t l = 0; l < input.extents(3); l++)
        {
            output[i, j, k, l] *= 1.0 / std::sqrt(v);
        }
    }
}

void softmax (
    const Array<float, 4> &input,
          Array<float, 4> &output
)
{
    if (!(input.extents() == output.extents()))
    {
        throw_with_context<std::domain_error>("Domain error. Check source location.");
    }

    for(int64_t i = 0; i < input.extents(0); i++)
    {
        float sum = 0.0;
        float max = input[i, 0, 0, 0];

        for(int64_t j = 0; j < input.extents(1); j++)
        for(int64_t k = 0; k < input.extents(2); k++)
        for(int64_t l = 0; l < input.extents(3); l++)
        {
            if(input[i, j, k, l] > max)
            {
                max = input[i, j, k, l];
            }
        }

        for(int64_t j = 0; j < input.extents(1); j++)
        for(int64_t k = 0; k < input.extents(2); k++)
        for(int64_t l = 0; l < input.extents(3); l++)
        {
            sum += std::exp(input[i, j, k, l] - max);
        }

        for(int64_t j = 0; j < input.extents(1); j++)
        for(int64_t k = 0; k < input.extents(2); k++)
        for(int64_t l = 0; l < input.extents(3); l++)
        {
            output[i, j, k, l] = std::exp(input[i, j, k, l] - max) / sum;
        }
    }
}

void softmax_gradient (
    const Array<float, 4> &input,
    const Array<float, 4> &output,
    const Array<float, 4> &output_gradient,
          Array<float, 4> &input_gradient
)
{
    if (
        !(input.extents() == output.extents())           ||
        !(output.extents() == output_gradient.extents()) ||
        !(input.extents() == input_gradient.extents())
    )
    {
        throw_with_context<std::domain_error>("Domain error. Check source location.");
    }

    for(int64_t i = 0; i < input.extents(0); i++)
    {
        float sum = 0.0;

        for(int64_t j = 0; j < input.extents(1); j++)
        for(int64_t k = 0; k < input.extents(2); k++)
        for(int64_t l = 0; l < input.extents(3); l++)
        {
            sum += output_gradient[i, j, k, l] * output[i, j, k, l];
        }

        for(int64_t j = 0; j < input.extents(1); j++)
        for(int64_t k = 0; k < input.extents(2); k++)
        for(int64_t l = 0; l < input.extents(3); l++)
        {
            input_gradient[i, j, k, l] = output[i, j, k, l] * (output_gradient[i, j, k, l] - sum);
        }
    }
}

void data_loss (
    const Array<float, 4> &input,
    const Array<int64_t, 1> &labels,
    float &output
)
{

    if (
        !(labels.extents(0) == input.extents(0)) ||
        !(input.extents(2) == 0)                 ||
        !(input.extents(3) == 0)
    )
    {
        throw_with_context<std::domain_error>("Domain error. Check source location.");
    }

    output = 0.0;

    for(int64_t i = 0; i < input.extents(0); i++)
    {
        output -= std::log(input[i, labels[i], 0, 0]);
    }

    output /= input.extents(0);
}

void data_loss_gradient (
    const Array<float, 4> &input,
    const Array<int64_t, 1> &labels,
    Array<float, 4> &input_gradient
)
{
    if (
        !(labels.extents(0) == input.extents(0)) ||
        !(input_gradient.extents() == input.extents())
    )
    {
        throw_with_context<std::domain_error>("Domain error. Check source location.");
    }

    for(int64_t i = 0; i < input.extents(0); i++)
    {
        input_gradient[i, labels[i], 0, 0] = -1.0 / (input.extents(0) * input[i, labels[i], 0, 0]);
    }
}

auto cross_correlation_output_extents (
    const Array<int64_t, 1, {4}> &input_extents,
    const Array<int64_t, 1, {2}> &f,
    const Array<int64_t, 1, {2}> &d,
    const Array<int64_t, 1, {2}> &s,
    const int64_t &n
) -> Array<int64_t, 1, {4}>
{
    if (
        !(std::all_of(input_extents.p_elements(), input_extents.p_elements() + 4, [] (const int64_t &x) { return x > 0; })) ||
        !(std::all_of(f.p_elements(), f.p_elements() + 2, [] (const int64_t &x) { return x >  0; }))                        ||
        !(std::all_of(d.p_elements(), d.p_elements() + 2, [] (const int64_t &x) { return x >  0; }))                        ||
        !(std::all_of(s.p_elements(), s.p_elements() + 2, [] (const int64_t &x) { return x >  0; }))                        ||
        !(n > 0)
    )
    {
        throw_with_context<std::domain_error>("Domain error. Check source location.");
    }

    Array<int64_t, 1, {4}> output_extents;

    output_extents[0] = input_extents[0];
    output_extents[1] = n;

    output_extents[2] = input_extents[2] - (f[0] - 1) * d[0];
    output_extents[3] = input_extents[3] - (f[1] - 1) * d[1];

    if (
        !((output_extents[2] > 0) && ((output_extents[2] - 1) % s[0] == 0)) ||
        !((output_extents[3] > 0) && ((output_extents[3] - 1) % s[1] == 0))
    )
    {
        throw_with_context<std::domain_error>("Domain error. Check source location.");
    }

    output_extents[2] = (output_extents[2] - 1) / s[0] + 1;
    output_extents[3] = (output_extents[3] - 1) / s[1] + 1;

    return output_extents;
}

void initialize_w(Array<float, 4> &w)
{
    float m = 0.0;
    float v = 2.0 / (static_cast<float>(w.size()) / static_cast<float>(w.extents(0)));

    std::mt19937_64 prng(std::chrono::system_clock::now().time_since_epoch().count());
    std::normal_distribution<float> nd(m, std::sqrt(v));

    for(int64_t i = 0; i < w.size(); i++)
        w.p_elements()[i] = nd(prng);
}

void cross_correlation (
    const Array<float, 4> &input,
    const Array<float, 4> &w,
    const Array<float, 1> &b,
    const Array<int64_t, 1, {4}> &input_extents,
    const Array<int64_t, 1, {2}> &f,
    const Array<int64_t, 1, {2}> &d,
    const Array<int64_t, 1, {2}> &s,
    int64_t n,
    float (*const fa)(float),
    Array<float, 4> &output
)
{
    if (
        !(
            input.extents(0) == input_extents[0] &&
            input.extents(1) == input_extents[1] &&
            input.extents(2) == input_extents[2] &&
            input.extents(3) == input_extents[3]
        )
    )
    {
        throw_with_context<std::domain_error>("Domain error. Check source location.");
    }

    auto output_extents_computed = cross_correlation_output_extents(input_extents, f, d, s, n);

    if (
        !(
            output.extents(0) == output_extents_computed[0] &&
            output.extents(1) == output_extents_computed[1] &&
            output.extents(2) == output_extents_computed[2] &&
            output.extents(3) == output_extents_computed[3]
        )
    )
    {
        throw_with_context<std::domain_error>("Domain error. Check source location.");
    }

    Extents<4> i_i = {0, 0, 0, 0};
    Extents<4> i_o = {0, 0, 0, 0};
    Extents<4> i_w = {0, 0, 0, 0};

    for(i_o[0] = 0; i_o[0] < output.extents(0); i_o[0]++)
    for(i_o[1] = 0; i_o[1] < output.extents(1); i_o[1]++)
    {
        i_w[0] = i_o[1];
        for(i_o[2] = 0; i_o[2] < output.extents(2); i_o[2]++)
        for(i_o[3] = 0; i_o[3] < output.extents(3); i_o[3]++)
        {
            output[i_o] = b[i_o[1]];
            for(i_w[1] = 0; i_w[1] < w.extents(1); i_w[1]++)
            for(i_w[2] = 0; i_w[2] < w.extents(2); i_w[2]++)
            for(i_w[3] = 0; i_w[3] < w.extents(3); i_w[3]++)
            {
                i_i[0] = i_o[0];
                i_i[1] = i_w[1];
                i_i[2] = i_o[2] * s[0] + i_w[2] * d[0];
                i_i[3] = i_o[3] * s[1] + i_w[3] * d[1];

                output[i_o] += w[i_w] * input[i_i];
            }
            output[i_o] = fa(output[i_o]);
        }
    }
}

void cross_correlation_gradient (
    const Array<float, 4> &input,
    const Array<float, 4> &w,
    const Array<float, 1> &b,
    const Array<int64_t, 1, {4}> &input_extents,
    const Array<int64_t, 1, {2}> &f,
    const Array<int64_t, 1, {2}> &d,
    const Array<int64_t, 1, {2}> &s,
    int64_t n,
    float (*const fa_gradient)(float),
    float (*const fa_inverse)(float),
    const Array<float, 4> &output,
    const Array<float, 4> &output_gradient,
    Array<float, 4> &input_gradient,
    Array<float, 4> &w_gradient,
    Array<float, 1> &b_gradient
)
{
    if (
        !(
            input.extents(0) == input_extents[0] &&
            input.extents(1) == input_extents[1] &&
            input.extents(2) == input_extents[2] &&
            input.extents(3) == input_extents[3]
        )
    )
    {
        throw_with_context<std::domain_error>("Domain error. Check source location.");
    }

    auto output_extents_computed = cross_correlation_output_extents(input_extents, f, d, s, n);

    if (
        !(
            output.extents(0) == output_extents_computed[0] &&
            output.extents(1) == output_extents_computed[1] &&
            output.extents(2) == output_extents_computed[2] &&
            output.extents(3) == output_extents_computed[3]
        )
    )
    {
        throw_with_context<std::domain_error>("Domain error. Check source location.");
    }

    if (!(output_gradient.extents() == output.extents()))
    {
        throw_with_context<std::domain_error>("Domain error. Check source location.");
    }

    if (
        !(
            input_gradient.extents(0) == input_extents[0] &&
            input_gradient.extents(1) == input_extents[1] &&
            input_gradient.extents(2) == input_extents[2] &&
            input_gradient.extents(3) == input_extents[3]
        )
    )
    {
        throw_with_context<std::domain_error>("Domain error. Check source location.");
    }

    if (!(w_gradient.extents() == w.extents()))
    {
        throw_with_context<std::domain_error>("Domain error. Check source location.");
    }

    if (!(b_gradient.extents() == b.extents()))
    {
        throw_with_context<std::domain_error>("Domain error. Check source location.");
    }

    Extents<4> i_i = {0, 0, 0, 0};
    Extents<4> i_o = {0, 0, 0, 0};
    Extents<4> i_w = {0, 0, 0, 0};

    std::fill_n(input_gradient.p_elements(), input_gradient.size(), 0.0f);
    std::fill_n(w_gradient.p_elements(), w_gradient.size(), 0.0f);
    std::fill_n(b_gradient.p_elements(), b_gradient.size(), 0.0f);

    for(i_o[0] = 0; i_o[0] < output.extents(0); i_o[0]++)
    for(i_o[1] = 0; i_o[1] < output.extents(1); i_o[1]++)
    {
        i_w[0] = i_o[1];
        for(i_o[2] = 0; i_o[2] < output.extents(2); i_o[2]++)
        for(i_o[3] = 0; i_o[3] < output.extents(3); i_o[3]++)
        {
            float c = fa_gradient(fa_inverse(output[i_o])) * output_gradient[i_o];
            b_gradient[i_o[1]] += c;
            for(i_w[1] = 0; i_w[1] < w.extents(1); i_w[1]++)
            for(i_w[2] = 0; i_w[2] < w.extents(2); i_w[2]++)
            for(i_w[3] = 0; i_w[3] < w.extents(3); i_w[3]++)
            {
                i_i[0] = i_o[0];
                i_i[1] = i_w[1];
                i_i[2] = i_o[2] * s[0] + i_w[2] * d[0];
                i_i[3] = i_o[3] * s[1] + i_w[3] * d[1];

                w_gradient[i_w] += input[i_i] * c;
                input_gradient[i_i] += w[i_w] * c;
            }
        }
    }
}

void batch_normalization_batch_statistics (
    const Array<float, 4> &input,
    Array<float, 1> &m,
    Array<float, 1> &v
)
{
    if (!(m.extents(0) == input.extents(1)))
    {
        throw_with_context<std::domain_error>("Domain error. Check source location.");
    }

    if (!(v.extents(0) == input.extents(1)))
    {
        throw_with_context<std::domain_error>("Domain error. Check source location.");
    }

    std::fill_n(m.p_elements(), m.size(), 0.0f);
    std::fill_n(v.p_elements(), v.size(), 0.0f);

    for(int64_t j = 0; j < input.extents(1); j++)
    {
        for(int64_t i = 0; i < input.extents(0); i++)
        for(int64_t k = 0; k < input.extents(2); k++)
        for(int64_t l = 0; l < input.extents(3); l++)
            m[j] += input[i, j, k, l];

        m[j] /= input.extents(0) * input.extents(2) * input.extents(3);

        for(int64_t i = 0; i < input.extents(0); i++)
        for(int64_t k = 0; k < input.extents(2); k++)
        for(int64_t l = 0; l < input.extents(3); l++)
            v[j] += (input[i, j, k, l] - m[j]) * (input[i, j, k, l] - m[j]);

        v[j] /= input.extents(0) * input.extents(2) * input.extents(3);
    }
}

void batch_normalization (
    const Array<float, 4> &input,
    const Array<float, 1> &m,
    const Array<float, 1> &v,
    const Array<float, 1> &a,
    const Array<float, 1> &b,
    float epsilon,
    float (*const f)(float),
    Array<float, 4> &output
)
{
    if (
        !(input.extents() == output.extents()) ||
        !(m.extents(0) == input.extents(1)) ||
        !(m.extents(0) == v.extents(0)) ||
        !(m.extents(0) == a.extents(0)) ||
        !(m.extents(0) == b.extents(0))
    )
    {
        throw_with_context<std::domain_error>("Domain error. Check source location.");
    }

    for(int64_t j = 0; j < input.extents(1); j++)
    {
        float s = std::sqrt(v[j] + epsilon);
        for(int64_t i = 0; i < input.extents(0); i++)
        for(int64_t k = 0; k < input.extents(2); k++)
        for(int64_t l = 0; l < input.extents(3); l++)
            output[i, j, k, l] = f(a[j] * (input[i, j, k, l] - m[j]) / s + b[j]);
    }
}

void batch_normalization_gradient (
    const Array<float, 4> &input,
    const Array<float, 1> &m,
    const Array<float, 1> &v,
    const Array<float, 1> &a,
    const Array<float, 1> &b,
    float epsilon,
    float (*const f_gradient)(float),
    float (*const f_inverse)(float),
    const Array<float, 4> &output,
    const Array<float, 4> &output_gradient,
    Array<float, 4> &input_gradient,
    Array<float, 1> &a_gradient,
    Array<float, 1> &b_gradient
)
{
    if (
        !(input.extents() == output.extents()) ||
        !(m.extents(0) == input.extents(1)) ||
        !(m.extents(0) == v.extents(0)) ||
        !(m.extents(0) == a.extents(0)) ||
        !(m.extents(0) == b.extents(0)) ||
        !(output_gradient.extents() == output.extents()) ||
        !(input_gradient.extents() == input.extents()) ||
        !(a_gradient.extents() == a.extents()) ||
        !(b_gradient.extents() == b.extents())
    )
    {
        throw_with_context<std::domain_error>("Domain error. Check source location.");
    }

    std::fill_n(input_gradient.p_elements(), input_gradient.size(), 0.0f);
    std::fill_n(a_gradient.p_elements(), a_gradient.size(), 0.0f);
    std::fill_n(b_gradient.p_elements(), b_gradient.size(), 0.0f);

    float n = static_cast<float>(input.size()) / static_cast<float>(input.extents(1));

    for(int64_t j = 0; j < input.extents(1); j++)
    {
        float s = std::sqrt(v[j] + epsilon);
        float sum1 = 0.0;
        float sum2 = 0.0;
        for(int64_t i = 0; i < input.extents(0); i++)
        for(int64_t k = 0; k < input.extents(2); k++)
        for(int64_t l = 0; l < input.extents(3); l++)
        {
            float c = f_gradient(f_inverse(output[i, j, k, l])) * output_gradient[i, j, k, l];
            sum1 -= c;
            sum2 -= c * (input[i, j, k, l] - m[j]);
        }

        sum1 *= a[j] / (s * n);
        sum2 *= a[j] / (2.0 * s * s * s);

        for(int64_t i = 0; i < input.extents(0); i++)
        for(int64_t k = 0; k < input.extents(2); k++)
        for(int64_t l = 0; l < input.extents(3); l++)
        {
            float c = f_gradient(f_inverse(output[i, j, k, l])) * output_gradient[i, j, k, l];
            a_gradient[j] += c * (input[i, j, k, l] - m[j]) / s;
            b_gradient[j] += c;
            input_gradient[i, j, k, l] = sum1 + c * a[j] / s + sum2 * 2.0 * (input[i, j, k, l] - m[j]) / n;
        }
    }
}

void dropout (
    const Array<float, 4> &input,
    float p,
    Array<float, 4> &output,
    Array<bool, 4> &mask
)
{

    if (
        !(input.extents() == output.extents()) ||
        !(input.extents() == mask.extents())
    )
    {
        throw_with_context<std::domain_error>("Domain error. Check source location.");
    }

    std::mt19937_64 prng(std::chrono::system_clock::now().time_since_epoch().count());
    std::uniform_real_distribution<float> ud(0.0, 1.0);

    for(int64_t i = 0; i < input.size(); i++)
    {
        float r = ud(prng);
        if(r >= p)
        {
            mask.p_elements()[i] = false;
            output.p_elements()[i] = 0.0;
        }
        else
        {
            mask.p_elements()[i] = true;
            output.p_elements()[i] = input.p_elements()[i] / p;
        }
    }
}

void dropout_gradient (
    float p,
    const Array<float, 4> &output_gradient,
    const Array<bool, 4> &mask,
    Array<float, 4> &input_gradient
)
{
    if (
        !(output_gradient.extents() == mask.extents()) ||
        !(output_gradient.extents() == input_gradient.extents())
    )
    {
        throw_with_context<std::domain_error>("Domain error. Check source location.");
    }

    for(int64_t i = 0; i < output_gradient.size(); i++)
        input_gradient.p_elements()[i] = mask.p_elements()[i] ? output_gradient.p_elements()[i] / p : 0.0;
}

float classification_accuracy (
    const Array<float, 2> &predictions,
    const Array<int64_t, 1> &labels
)
{
    if (!(predictions.extents(0) == labels.extents(0)))
    {
        throw_with_context<std::domain_error>("Domain error. Check source location.");
    }

    float accuracy = 0.0;

    for(int64_t i = 0; i < predictions.extents(0); i++)
    {
        int64_t j_max = 0;

        for(int64_t j = 0; j < predictions.extents(1); j++)
        {
            if(predictions[i, j] > predictions[i, j_max])
            {
                j_max = j;
            }
        }

        if(j_max == labels[i])
            accuracy += 1.0;
    }

    accuracy /= labels.extents(0);

    return accuracy;
}

template<int64_t D>
void gradient_descent_step_momentum (
    const Array<float, D> &x,
    const Array<float, D> &v,
    const Array<float, D> &x_gradient,
    float a, float m,
    Array<float, D> &x_updated,
    Array<float, D> &v_updated
)
{
    if (
        !(x.extents() == v.extents()) ||
        !(x.extents() == x_gradient.extents()) ||
        !(x.extents() == x_updated.extents()) ||
        !(x.extents() == v_updated.extents())
    )
    {
        throw_with_context<std::domain_error>("Domain error. Check source location.");
    }

    for(int64_t i = 0; i < x.size(); i++)
    {
        v_updated.p_elements()[i] = m * v.p_elements()[i] - a * x_gradient.p_elements()[i];
        x_updated.p_elements()[i] = x.p_elements()[i] + v_updated.p_elements()[i];
    }
}

template<int64_t D>
void gradient_descent_step_adam (
    const Array<float, D> &x,
    const Array<float, D> &v,
    const Array<float, D> &m,
    const Array<float, D> &x_gradient,
    float a, float b1, float b2, float epsilon, int64_t t,
    Array<float, D> &x_updated,
    Array<float, D> &v_updated,
    Array<float, D> &m_updated
)
{
    if (
        !(x.extents() == v.extents())          ||
        !(x.extents() == m.extents())          ||
        !(x.extents() == x_gradient.extents()) ||
        !(x.extents() == x_updated.extents())  ||
        !(x.extents() == v_updated.extents())  ||
        !(x.extents() == m_updated.extents())
    )
    {
        throw_with_context<std::domain_error>("Domain error. Check source location.");
    }

    for(int64_t i = 0; i < x.size(); i++)
    {
        m_updated.p_elements()[i] = b1 * m.p_elements()[i] + (1.0 - b1) * x_gradient.p_elements()[i];
        v_updated.p_elements()[i] = b2 * v.p_elements()[i] + (1.0 - b2) * x_gradient.p_elements()[i] * x_gradient.p_elements()[i];
        float mt = m_updated.p_elements()[i] / (1.0 - std::pow(b1, t));
        float vt = v_updated.p_elements()[i] / (1.0 - std::pow(b2, t));
        x_updated.p_elements()[i] = x.p_elements()[i] - a * mt / (std::sqrt(vt) + epsilon);
    }
}
