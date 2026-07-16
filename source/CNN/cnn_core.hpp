#ifndef CNN_CORE_HPP
#define CNN_CORE_HPP

#include <Array.hpp>

// Declarations

void normalize_input (
    const Array<float, 4> &input,
          Array<float, 4> &output
);

void softmax (
    const Array<float, 4> &input,
          Array<float, 4> &output
);

void softmax_gradient (
    const Array<float, 4> &input,
    const Array<float, 4> &output,
    const Array<float, 4> &output_gradient,
          Array<float, 4> &input_gradient
);

void data_loss (
    const Array<float, 4> &input,
    const Array<int64_t, 1> &labels,
    float &output
);

void data_loss_gradient (
    const Array<float, 4> &input,
    const Array<int64_t, 1> &labels,
    Array<float, 4> &input_gradient
);

auto cross_correlation_output_extents (
    const Array<int64_t, 1, {4}> &input_extents,
    const Array<int64_t, 1, {2}> &f,
    const Array<int64_t, 1, {2}> &d,
    const Array<int64_t, 1, {2}> &s,
    const int64_t &n
) -> Array<int64_t, 1, {4}>;

void initialize_w(Array<float, 4> &w);

void cross_correlation (
    const Array<float, 4> &input,
    const Array<float, 4> &w,
    const Array<float, 1> &b,
    const Array<int64_t, 1, {4}> &input_shape,
    const Array<int64_t, 1, {2}> &f,
    const Array<int64_t, 1, {2}> &d,
    const Array<int64_t, 1, {2}> &s,
    int64_t n,
    float (*const fa)(float),
    Array<float, 4> &output
);

void cross_correlation_gradient (
    const Array<float, 4> &input,
    const Array<float, 4> &w,
    const Array<float, 1> &b,
    const Array<int64_t, 1, {4}> &input_shape,
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
);

void batch_normalization_batch_statistics (
    const Array<float, 4> &input,
    Array<float, 1> &m,
    Array<float, 1> &v
);

void batch_normalization (
    const Array<float, 4> &input,
    const Array<float, 1> &m,
    const Array<float, 1> &v,
    const Array<float, 1> &a,
    const Array<float, 1> &b,
    float epsilon,
    float (*const f)(float),
    Array<float, 4> &output
);

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
);

void dropout (
    const Array<float, 4> &input,
    float p,
    Array<float, 4> &output,
    Array<bool, 4> &mask
);

void dropout_gradient (
    float p,
    const Array<float, 4> &output_gradient,
    const Array<bool, 4> &mask,
    Array<float, 4> &input_gradient
);

float classification_accuracy (
    const Array<float, 4> &predictions,
    const Array<int64_t, 1> &labels
);

template<int64_t D>
void gradient_descent_step_momentum (
    const Array<float, D> &x,
    const Array<float, D> &v,
    const Array<float, D> &x_gradient,
    float a, float m,
    Array<float, D> &x_updated,
    Array<float, D> &v_updated
);

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
);

#endif // CNN_CORE_HPP
