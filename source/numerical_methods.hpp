#ifndef NUMERICAL_METHODS_HPP
#define NUMERICAL_METHODS_HPP

#include "Array.hpp"

//******************************************************************************
// Newton-Cotes numerical integration
//******************************************************************************

// Declarations

template<typename A>
requires (( ArrayType<A> || ViewType<A> ) && A::dimension() == 1 )
auto integrate_nc_c_2p(const A &samples, double h, int64_t stride = 1) -> typename A::Element;

template<typename A>
requires (( ArrayType<A> || ViewType<A> ) && A::dimension() == 1 )
auto integrate_nc_c_2p_romberg(const A &samples, double h) -> typename A::Element;

template<typename A>
requires (( ArrayType<A> || ViewType<A> ) && A::dimension() == 2 )
auto integrate_nc_c_2p_romberg_2d(const A &samples, double hx, double hy) -> typename A::Element;

// Definitions

template<typename A>
requires (( ArrayType<A> || ViewType<A> ) && A::dimension() == 1 )
auto integrate_nc_c_2p(const A &samples, double h, int64_t stride) -> typename A::Element
{
    auto n = samples.extents(0);

    if ( n < 2                 ) throw_with_context<std::domain_error>("Domain error. Check source location.");
    if ( stride < 1            ) throw_with_context<std::domain_error>("Domain error. Check source location.");
    if ( stride >= n           ) throw_with_context<std::domain_error>("Domain error. Check source location.");
    if ( (n - 1) % stride != 0 ) throw_with_context<std::domain_error>("Domain error. Check source location.");

    h *= stride;

    typename A::Element sum = 0.0;

    for (int64_t i = stride; i < n - stride; i += stride)
    {
        sum += samples[i];
    }

    sum *= 2.0;
    sum += samples[0] + samples[(n - 1)];
    sum *= h / 2.0;

    return sum;
}

template<typename A>
requires (( ArrayType<A> || ViewType<A> ) && A::dimension() == 1 )
auto integrate_nc_c_2p_romberg(const A &samples, double h) -> typename A::Element
{
    typename A::Element I[64] = {0};

    int64_t n = samples.extents(0);
    int64_t stride = 1;
    int64_t n_panels_log2 = std::countr_zero(uint64_t(n - 1));

    for(int64_t i = 0; i <= n_panels_log2; i++)
    {
        I[i] = integrate_nc_c_2p(samples, h, stride);
        stride *= 2;
    }

    for(int64_t i = n_panels_log2; i > 0; i--)
        for(int64_t k = 0; k < i; k++)
            I[k] = (std::pow(4.0, n_panels_log2 - i + 1) * I[k] - I[k + 1])
                 / (std::pow(4.0, n_panels_log2 - i + 1) - 1.0);

    return I[0];
}

template<typename A>
requires (( ArrayType<A> || ViewType<A> ) && A::dimension() == 2 )
auto integrate_nc_c_2p_romberg_2d(const A &samples, double hx, double hy) -> typename A::Element
{
    Array<typename A::Element, 1> integrals_x(samples.extents(0));

    for(int64_t i = 0; i < integrals_x.extents(0); i++)
    {
        auto view = make_read_only_slice_view<1, {1}>(samples, {i, 0});
        integrals_x[i] = integrate_nc_c_2p_romberg(view, hx);
    }

    auto sum = integrate_nc_c_2p_romberg(integrals_x, hy);

    return sum;
}


//******************************************************************************
// Reverse the bits of an integer
//******************************************************************************

// Declarations

uint8_t  reverse_bit_order( uint8_t  x );
uint16_t reverse_bit_order( uint16_t x );
uint32_t reverse_bit_order( uint32_t x );
uint64_t reverse_bit_order( uint64_t x );

int8_t  reverse_bit_order ( int8_t  x );
int16_t reverse_bit_order ( int16_t x );
int32_t reverse_bit_order ( int32_t x );
int64_t reverse_bit_order ( int64_t x );

// Definitions

uint8_t reverse_bit_order_lut[256] =
{
    0x00, 0x80, 0x40, 0xC0, 0x20, 0xA0, 0x60, 0xE0, 0x10, 0x90, 0x50, 0xD0, 0x30, 0xB0, 0x70, 0xF0,
    0x08, 0x88, 0x48, 0xC8, 0x28, 0xA8, 0x68, 0xE8, 0x18, 0x98, 0x58, 0xD8, 0x38, 0xB8, 0x78, 0xF8,
    0x04, 0x84, 0x44, 0xC4, 0x24, 0xA4, 0x64, 0xE4, 0x14, 0x94, 0x54, 0xD4, 0x34, 0xB4, 0x74, 0xF4,
    0x0C, 0x8C, 0x4C, 0xCC, 0x2C, 0xAC, 0x6C, 0xEC, 0x1C, 0x9C, 0x5C, 0xDC, 0x3C, 0xBC, 0x7C, 0xFC,
    0x02, 0x82, 0x42, 0xC2, 0x22, 0xA2, 0x62, 0xE2, 0x12, 0x92, 0x52, 0xD2, 0x32, 0xB2, 0x72, 0xF2,
    0x0A, 0x8A, 0x4A, 0xCA, 0x2A, 0xAA, 0x6A, 0xEA, 0x1A, 0x9A, 0x5A, 0xDA, 0x3A, 0xBA, 0x7A, 0xFA,
    0x06, 0x86, 0x46, 0xC6, 0x26, 0xA6, 0x66, 0xE6, 0x16, 0x96, 0x56, 0xD6, 0x36, 0xB6, 0x76, 0xF6,
    0x0E, 0x8E, 0x4E, 0xCE, 0x2E, 0xAE, 0x6E, 0xEE, 0x1E, 0x9E, 0x5E, 0xDE, 0x3E, 0xBE, 0x7E, 0xFE,
    0x01, 0x81, 0x41, 0xC1, 0x21, 0xA1, 0x61, 0xE1, 0x11, 0x91, 0x51, 0xD1, 0x31, 0xB1, 0x71, 0xF1,
    0x09, 0x89, 0x49, 0xC9, 0x29, 0xA9, 0x69, 0xE9, 0x19, 0x99, 0x59, 0xD9, 0x39, 0xB9, 0x79, 0xF9,
    0x05, 0x85, 0x45, 0xC5, 0x25, 0xA5, 0x65, 0xE5, 0x15, 0x95, 0x55, 0xD5, 0x35, 0xB5, 0x75, 0xF5,
    0x0D, 0x8D, 0x4D, 0xCD, 0x2D, 0xAD, 0x6D, 0xED, 0x1D, 0x9D, 0x5D, 0xDD, 0x3D, 0xBD, 0x7D, 0xFD,
    0x03, 0x83, 0x43, 0xC3, 0x23, 0xA3, 0x63, 0xE3, 0x13, 0x93, 0x53, 0xD3, 0x33, 0xB3, 0x73, 0xF3,
    0x0B, 0x8B, 0x4B, 0xCB, 0x2B, 0xAB, 0x6B, 0xEB, 0x1B, 0x9B, 0x5B, 0xDB, 0x3B, 0xBB, 0x7B, 0xFB,
    0x07, 0x87, 0x47, 0xC7, 0x27, 0xA7, 0x67, 0xE7, 0x17, 0x97, 0x57, 0xD7, 0x37, 0xB7, 0x77, 0xF7,
    0x0F, 0x8F, 0x4F, 0xCF, 0x2F, 0xAF, 0x6F, 0xEF, 0x1F, 0x9F, 0x5F, 0xDF, 0x3F, 0xBF, 0x7F, 0xFF
};

uint8_t reverse_bit_order(uint8_t x)
{
    return reverse_bit_order_lut[x];
}

uint16_t reverse_bit_order(uint16_t x)
{
    x = ( uint16_t ( reverse_bit_order_lut [  x         & 0xFF ] )  <<  8 ) |
        ( uint16_t ( reverse_bit_order_lut [ (x >>  8 ) & 0xFF ] ) );

    return x;
}

uint32_t reverse_bit_order(uint32_t x)
{
    x = ( uint32_t ( reverse_bit_order_lut [   x         & 0xFF ] ) << 24 ) |
        ( uint32_t ( reverse_bit_order_lut [ ( x >>  8 ) & 0xFF ] ) << 16 ) |
        ( uint32_t ( reverse_bit_order_lut [ ( x >> 16 ) & 0xFF ] ) <<  8 ) |
        ( uint32_t ( reverse_bit_order_lut [ ( x >> 24 ) & 0xFF ] ) <<  0 );

    return x;
}

uint64_t reverse_bit_order(uint64_t x)
{
    x = ( uint64_t ( reverse_bit_order_lut [   x         & 0xFF ] ) << 56 ) |
        ( uint64_t ( reverse_bit_order_lut [ ( x >>  8 ) & 0xFF ] ) << 48 ) |
        ( uint64_t ( reverse_bit_order_lut [ ( x >> 16 ) & 0xFF ] ) << 40 ) |
        ( uint64_t ( reverse_bit_order_lut [ ( x >> 24 ) & 0xFF ] ) << 32 ) |
        ( uint64_t ( reverse_bit_order_lut [ ( x >> 32 ) & 0xFF ] ) << 24 ) |
        ( uint64_t ( reverse_bit_order_lut [ ( x >> 40 ) & 0xFF ] ) << 16 ) |
        ( uint64_t ( reverse_bit_order_lut [ ( x >> 48 ) & 0xFF ] ) <<  8 ) |
        ( uint64_t ( reverse_bit_order_lut [ ( x >> 56 ) & 0xFF ] ) <<  0 );

    return x;
}

int8_t  reverse_bit_order ( int8_t  x ) { return reverse_bit_order ( uint8_t  (x) ); }
int16_t reverse_bit_order ( int16_t x ) { return reverse_bit_order ( uint16_t (x) ); }
int32_t reverse_bit_order ( int32_t x ) { return reverse_bit_order ( uint32_t (x) ); }
int64_t reverse_bit_order ( int64_t x ) { return reverse_bit_order ( uint64_t (x) ); }

//******************************************************************************
// FFT
//******************************************************************************

constexpr double pi = std::numbers::pi;

const std::complex<double> w(uint32_t k, uint32_t n)
{
    return {
        std::cos(-2.0 * pi * k / n),
        std::sin(-2.0 * pi * k / n)
    };
}

const std::complex<double> w_lut[32] = {
    w(1, 1 <<  0), w(1, 1 <<  1), w(1, 1 <<  2), w(1, 1 <<  3),
    w(1, 1 <<  4), w(1, 1 <<  5), w(1, 1 <<  6), w(1, 1 <<  7),
    w(1, 1 <<  8), w(1, 1 <<  9), w(1, 1 << 10), w(1, 1 << 11),
    w(1, 1 << 12), w(1, 1 << 13), w(1, 1 << 14), w(1, 1 << 15),
    w(1, 1 << 16), w(1, 1 << 17), w(1, 1 << 18), w(1, 1 << 19),
    w(1, 1 << 20), w(1, 1 << 21), w(1, 1 << 22), w(1, 1 << 23),
    w(1, 1 << 24), w(1, 1 << 25), w(1, 1 << 26), w(1, 1 << 27),
    w(1, 1 << 28), w(1, 1 << 29), w(1, 1 << 30), w(1, 1 << 31)
};

template <typename A>
requires (
    ( ArrayType<A> || ViewType<A> ) &&
    ( A::dimension() == 1 ) &&
    ( ScalarComplexFloatingPointNumType<typename A::Element> )
)
void butterfly_radix_2 (
    A &data,
    int64_t index,
    int64_t stride,
    const std::complex<double> &w
)
{
    auto temp = data[index] + w * data[index + stride];
    data[index + stride] = data[index] - w * data[index + stride];
    data[index] = temp;
}

template <typename A>
requires (
    ( ArrayType<A> || ViewType<A> ) &&
    ( A::dimension() == 1 ) &&
    ( ScalarComplexFloatingPointNumType<typename A::Element> )
)
void butterfly_radix_4 (
    A &data,
    int64_t index,
    int64_t stride,
    const std::complex<double> &w1,
    const std::complex<double> &w2,
    const std::complex<double> &w3
)
{
    int64_t offset1 = index + (stride >> 2);
    int64_t offset2 = index + (stride >> 1);
    int64_t offset3 = index + offset1 + offset2;

    auto temp0 = data[index] + data[offset2] * w2;
    auto temp1 = data[index] - data[offset2] * w2;
    auto temp2 = data[offset1] * w1 + data[offset3] * w3;
    auto temp3 = data[offset1] * w1 - data[offset3] * w3;

    std::complex<double> temp4 = {-temp3.imag(), temp3.real()};

    data[0] = temp0 + temp2;
    data[offset1] = temp1 - temp4;
    data[offset2] = temp0 - temp2;
    data[offset3] = temp1 + temp4;
}

template <typename A>
requires (
    ( ArrayType<A> || ViewType<A> ) &&
    ( A::dimension() == 1 ) &&
    ( ScalarComplexFloatingPointNumType<typename A::Element> )
)
void butterflies(A &data, int64_t index, int64_t log2_n, int64_t li, int64_t lf)
{
    int64_t n = 1 << log2_n;

    for(int64_t i = li; i <= lf; i++)
    {
        int64_t stride = 1 << (i - 1);
        std::complex<double> w = {1.0, 0.0};
        for(int64_t k = 0; k < stride; k++)
        {
            for(int64_t j = 0; j < n; j += 2 * stride)
                butterfly_radix_2(data, index + j + k, stride, w);
            if(k % 16 == 0)
                w = exp(std::complex<double>{0.0, -2.0 * pi * (k + 1) / (1 << i)});
            else
                w *= w_lut[i];
        }
    }
}

void sno(uint32_t &o, uint32_t &l, bool &c)
{
    if(c)
    {
        o &= ~(1 << (l - 1));
        if((o & (1 << l)) == 0) c = false;
        l += 1;
    }
    else
    {
        o |= (1 << (l - 1));
        if(o & 1) c = true;
        l = 1;
    }
}

template <typename A>
requires (
    ( ArrayType<A> || ViewType<A> ) &&
    ( A::dimension() == 1 ) &&
    ( ScalarComplexFloatingPointNumType<typename A::Element> )
)
void dft(A &data, uint32_t log2_n, bool inverse)
{
    int64_t n = 1 << log2_n;

    if(inverse)
    {
        for(int64_t i = 0; i < n; i++)
        {
            data[i].real(data[i].real() *  1.0 / double(n));
            data[i].imag(data[i].imag() * -1.0 / double(n));
        }
    }

    int32_t brs = 128;
    for(int32_t k = 0; k < brs; k++)
    for(int32_t i = k; i < n; i += brs)
    {
        int32_t j = reverse_bit_order(i << (32 - log2_n));
        if(i < j)
        {
            std::swap(data[i], data[j]);
        }
    }

    bool c = 0;
    uint32_t l = 1;
    uint32_t o = 0;

    uint32_t li = 10;

    uint32_t osfb = 2;
    uint32_t osfli = 1 << (li - 1);
    uint32_t osf = osfb * osfli;

    uint32_t imax = n / (1 << (li - 1)) - 1;

    if(log2_n < li)
        butterflies(data, 0, log2_n, 1, log2_n);
    else
        for(uint32_t i = 0; i < imax; i++)
        {
            if(l == 1)
            {
                butterflies(data, osf * o, li, 1, li);
            }
            else
            {
                uint32_t bblog2_n = l + li - 1;
                uint32_t bbli = l + li - 1;
                uint32_t bblf = l + li - 1;
                butterflies(data, osf * o, bblog2_n, bbli, bblf);
            }
            sno(o, l, c);
        }

    if(inverse)
        for(int64_t i = 0; i < n; i++)
            data[i].imag(-data[i].imag());
}

template <typename T>
void wait_all(std::vector<std::future<T>> &futures)
{
    for (auto &future : futures)
    {
        future.get();
    }
}

void dft_2d(Array<std::complex<double>, 2> &data, bool inverse)
{

    if (
        data.size() == 0 ||
        !std::has_single_bit(uint64_t(data.extents(0))) ||
        !std::has_single_bit(uint64_t(data.extents(1)))
    )
    {
        throw_with_context<std::domain_error>("Domain error. Check source location.");
    }

    int64_t log2_n_0 = std::countr_zero(uint64_t(data.extents(0)));
    int64_t log2_n_1 = std::countr_zero(uint64_t(data.extents(1)));
    int64_t n_0 = data.extents(0);
    int64_t n_1 = data.extents(1);

    SingleProducerAsyncTaskQueuePool thread_pool(8);

    std::vector<std::future<void>> futures;
    futures.reserve(std::max(n_0, n_1));

    for(int64_t i = 0; i < n_0; i++)
    {
        futures.push_back (
            thread_pool.enqueue_task (
                [&, i]() {
                    auto row_view = make_slice_view<1, {1}>(data, {i, 0});
                    dft(row_view, log2_n_1, inverse);
                }
            )
        );
    }

    wait_all(futures);
    futures.clear();

    data = transposed(data);

    for(int64_t i = 0; i < n_1; i++)
    {
        futures.push_back (
            thread_pool.enqueue_task (
                [&, i]() {
                    auto row_view = make_slice_view<1, {1}>(data, {i, 0});
                    dft(row_view, log2_n_0, inverse);
                }
            )
        );
    }

    wait_all(futures);

    data = transposed(data);
}

//******************************************************************************
// RK4
//******************************************************************************

template<typename S, typename A, typename X>
struct ode_ivp_RK4_state
{
    ode_ivp_RK4_state (
        const S &t, const A &y, const X &auxiliary_data
    ) : t(t), auxiliary_data(auxiliary_data), y(y)
    {
        k1 = A(y.extents());
        k2 = A(y.extents());
        k3 = A(y.extents());
        k4 = A(y.extents());
    }

    S t;
    A y;
    A k1, k2, k3, k4;
    X auxiliary_data;
};

template<typename S, typename A, typename X>
void ode_ivp_RK4 (
    auto (*f)(const S &, const A &, const X &) -> A,
    ode_ivp_RK4_state<S, A, X> &state,
    const S &tf,
    const int64_t &n_steps
)
{
    if (n_steps <= 0)
    {
        throw_with_context<std::domain_error>("Domain error. Check source location.");
    }

    S h = (tf - state.t) / n_steps;

    for(int64_t i = 1; i <= n_steps; i++)
    {
        state.k1 = f(state.t + S(0.0) * h,                     state.y, state.auxiliary_data) * h;
        state.k2 = f(state.t + S(0.5) * h, S(0.5) * state.k1 + state.y, state.auxiliary_data) * h;
        state.k3 = f(state.t + S(0.5) * h, S(0.5) * state.k2 + state.y, state.auxiliary_data) * h;
        state.k4 = f(state.t + S(1.0) * h,          state.k3 + state.y, state.auxiliary_data) * h;

        state.y = state.y + 1.0 / 6.0 * (2.0 * (state.k2 + state.k3) + state.k1 + state.k4);

        state.t = tf - h * (n_steps - i);
    }
}

//******************************************************************************
// Solve a banded linear system
//******************************************************************************

template<typename T>
void solve_linear_system_banded (
    Array<T, 2> &A, Array<T, 1> &b, T *x, int64_t l, int64_t u
)
{
    if (A.extents(0) != b.extents(0))
    {
        throw_with_context<std::domain_error>("Domain error. Check source location.");
    }

    if (A.extents(1) != l + u + 1)
    {
        throw_with_context<std::domain_error>("Domain error. Check source location.");
    }

    int64_t n = A.extents(0);
    int64_t w = A.extents(1);

    for(int64_t j = 0; j < l; j++)
    {
        for(int64_t i = l - j; i < n; i++)
        {
            T c = A[i, j] / A[i - 1, j + 1];
            b[i] -= b[i - 1] * c;
            for(int64_t k = j; k < w - 1; k++)
            {
                A[i, k] -= A[i - 1, k + 1] * c;
            }
        }
    }

    int64_t i_init = n - 1 - u;

    for(int64_t j = w - 1; j > l; j--)
    {
        for(int64_t i = i_init; i >= 0; i--)
        {
            T c = A[i, j] / A[i + 1, j - 1];
            b[i] -= b[i + 1] * c;
            for(int64_t k = j; k >= l; k--)
            {
                A[i, k] -= A[i + 1, k - 1] * c;
            }
        }
        i_init++;
    }

    for(int64_t i = 0; i < n; i++)
    {
        x[i] = b[i] / A[i, l];
    }
}

#endif // NUMERICAL_METHODS_HPP