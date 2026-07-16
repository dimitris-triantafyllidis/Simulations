#include "cnn_utilities.hpp"

#include <chrono>
#include <fstream>

Dataset::Dataset()
{
    m_prng = std::mt19937(std::chrono::system_clock::now().time_since_epoch().count());

    m_training_cursor = 0;
    m_validation_cursor = 0;
    m_test_cursor = 0;
}

void Dataset::load_MNIST(const std::string &path)
{
    m_training_images   = Array<char, 4>(50000, 1, 28, 28);
    m_validation_images = Array<char, 4>(10000, 1, 28, 28);
    m_test_images       = Array<char, 4>(10000, 1, 28, 28);

    m_training_labels   = Array<char, 1>(50000);
    m_validation_labels = Array<char, 1>(10000);
    m_test_labels       = Array<char, 1>(10000);

    std::ifstream file;

    file.open(path + "train-images.idx3-ubyte", std::ios::binary);
    if (!(file.is_open()))
    {
        throw_with_context<std::runtime_error>("Runtime error. Check source location.");
    }
    file.seekg(16);
    file.read(m_training_images.p_elements(), 50000 * 28 * 28);
    file.read(m_validation_images.p_elements(), 10000 * 28 * 28);
    file.close();

    file.open(path + "train-labels.idx1-ubyte", std::ios::binary);
    if (!(file.is_open()))
    {
        throw_with_context<std::runtime_error>("Runtime error. Check source location.");
    }
    file.seekg(8);
    file.read(m_training_labels.p_elements(), 50000);
    file.read(m_validation_labels.p_elements(), 10000);
    file.close();

    file.open(path + "t10k-images.idx3-ubyte", std::ios::binary);
    if (!(file.is_open()))
    {
        throw_with_context<std::runtime_error>("Runtime error. Check source location.");
    }
    file.seekg(16);
    file.read(m_test_images.p_elements(), 10000 * 28 * 28);
    file.close();

    file.open(path + "t10k-labels.idx1-ubyte", std::ios::binary);
    if (!(file.is_open()))
    {
        throw_with_context<std::runtime_error>("Runtime error. Check source location.");
    }
    file.seekg(8);
    file.read(m_test_labels.p_elements(), 10000);
    file.close();

    m_training_permutation   = Array<int64_t, 1>(50000);
    m_validation_permutation = Array<int64_t, 1>(10000);
    m_test_permutation       = Array<int64_t, 1>(10000);

    std::iota(
        m_training_permutation.p_elements(),
        m_training_permutation.p_elements() + m_training_permutation.size(), 0
    );

    std::iota(
        m_validation_permutation.p_elements(),
        m_validation_permutation.p_elements() + m_validation_permutation.size(), 0
    );

    std::iota(
        m_test_permutation.p_elements(),
        m_test_permutation.p_elements() + m_test_permutation.size(), 0
    );

    std::shuffle(
        m_training_permutation.p_elements(),
        m_training_permutation.p_elements() + m_training_permutation.size(), m_prng
    );

    std::shuffle(
        m_validation_permutation.p_elements(),
        m_validation_permutation.p_elements() + m_validation_permutation.size(), m_prng
    );

    std::shuffle(
        m_test_permutation.p_elements(),
        m_test_permutation.p_elements() + m_test_permutation.size(), m_prng
    );

    m_training_cursor   = 0;
    m_validation_cursor = 0;
    m_test_cursor       = 0;
}

void Dataset::load_CIFAR10(const std::string &path)
{
    Array<char, 1> everything(60000 * 3073);

    std::ifstream file;

    file.open(path + "data_batch_1.bin", std::ios::binary);
    if (!(file.is_open()))
    {
        throw_with_context<std::runtime_error>("Runtime error. Check source location.");
    }
    file.seekg(0);
    file.read(everything.p_elements() + 0 * 10000 * 3073 , 10000 * 3073);
    file.close();

    file.open(path + "data_batch_2.bin", std::ios::binary);
    if (!(file.is_open()))
    {
        throw_with_context<std::runtime_error>("Runtime error. Check source location.");
    }
    file.seekg(0);
    file.read(everything.p_elements() + 1 * 10000 * 3073 , 10000 * 3073);
    file.close();

    file.open(path + "data_batch_3.bin", std::ios::binary);
    if (!(file.is_open()))
    {
        throw_with_context<std::runtime_error>("Runtime error. Check source location.");
    }
    file.seekg(0);
    file.read(everything.p_elements() + 2 * 10000 * 3073 , 10000 * 3073);
    file.close();

    file.open(path + "data_batch_4.bin", std::ios::binary);
    if (!(file.is_open()))
    {
        throw_with_context<std::runtime_error>("Runtime error. Check source location.");
    }
    file.seekg(0);
    file.read(everything.p_elements() + 3 * 10000 * 3073 , 10000 * 3073);
    file.close();

    file.open(path + "data_batch_5.bin", std::ios::binary);
    if (!(file.is_open()))
    {
        throw_with_context<std::runtime_error>("Runtime error. Check source location.");
    }
    file.seekg(0);
    file.read(everything.p_elements() + 4 * 10000 * 3073 , 10000 * 3073);
    file.close();

    file.open(path + "test_batch.bin", std::ios::binary);
    if (!(file.is_open()))
    {
        throw_with_context<std::runtime_error>("Runtime error. Check source location.");
    }
    file.seekg(0);
    file.read(everything.p_elements() + 5 * 10000 * 3073, 10000 * 3073);
    file.close();

    m_training_images   = Array<char, 4>(40000, 3, 32, 32);
    m_validation_images = Array<char, 4>(10000, 3, 32, 32);
    m_test_images       = Array<char, 4>(10000, 3, 32, 32);

    m_training_labels   = Array<char, 1>(40000);
    m_validation_labels = Array<char, 1>(10000);
    m_test_labels       = Array<char, 1>(10000);

    for(int64_t i = 0; i < 40000; i++)
    {
        std::copy_n(everything.p_elements() + 0 * 3073 + 3073 * i + 1, 3072, m_training_images.p_elements() + 3072 * i);
        m_training_labels[i] = everything[0 * 3073 + 3073 * i];
    }

    for(int64_t i = 0; i < 10000; i++)
    {
        std::copy_n(everything.p_elements() + 40000 * 3073 + 3073 * i + 1, 3072, m_validation_images.p_elements() + 3072 * i);
        m_validation_labels[i] = everything[40000 * 3073 + 3073 * i];

        std::copy_n(everything.p_elements() + 50000 * 3073 + 3073 * i + 1, 3072, m_test_images.p_elements() + 3072 * i);
        m_test_labels[i] = everything[50000 * 3073 + 3073 * i];
    }

    m_training_permutation   = Array<int64_t, 1>(40000);
    m_validation_permutation = Array<int64_t, 1>(10000);
    m_test_permutation       = Array<int64_t, 1>(10000);

    std::iota (
        m_training_permutation.p_elements(),
        m_training_permutation.p_elements() + m_training_permutation.size(), 0
    );

    std::iota (
        m_validation_permutation.p_elements(),
        m_validation_permutation.p_elements() + m_validation_permutation.size(), 0
    );

    std::iota (
        m_test_permutation.p_elements(),
        m_test_permutation.p_elements() + m_test_permutation.size(), 0
    );

    std::shuffle (
        m_training_permutation.p_elements(),
        m_training_permutation.p_elements() + m_training_permutation.size(), m_prng
    );

    std::shuffle (
        m_validation_permutation.p_elements(),
        m_validation_permutation.p_elements() + m_validation_permutation.size(), m_prng
    );

    std::shuffle (
        m_test_permutation.p_elements(),
        m_test_permutation.p_elements() + m_test_permutation.size(), m_prng
    );

    m_training_cursor   = 0;
    m_validation_cursor = 0;
    m_test_cursor       = 0;
}

void Dataset::get_training_batch (
    int64_t batch_size,
    Array<float, 4> &batch_images,
    Array<int64_t, 1> &batch_labels
)
{
    get_batch (
        batch_size,
        batch_images,
        batch_labels,
        m_training_images,
        m_training_labels,
        m_training_cursor,
        m_training_permutation
    );
}

void Dataset::get_validation_batch (
    int64_t batch_size,
    Array<float, 4> &batch_images,
    Array<int64_t, 1> &batch_labels
)
{
    get_batch (
        batch_size,
        batch_images,
        batch_labels,
        m_validation_images,
        m_validation_labels,
        m_validation_cursor,
        m_validation_permutation
    );
}

void Dataset::get_test_batch (
    int64_t batch_size,
    Array<float, 4> &batch_images,
    Array<int64_t, 1> &batch_labels
)
{
    get_batch (
        batch_size,
        batch_images,
        batch_labels,
        m_test_images,
        m_test_labels,
        m_test_cursor,
        m_test_permutation
    );
}

void Dataset::get_batch (
    int64_t batch_size,
    Array<float, 4> &batch_images,
    Array<int64_t, 1> &batch_labels,
    const Array<char, 4> &dataset_images,
    const Array<char, 1> &dataset_labels,
    int64_t &dataset_cursor,
    Array<int64_t, 1> &dataset_permutation
)
{
    if (
        !(batch_size > 0) ||
        !(batch_size + dataset_cursor <= dataset_images.extents(0)) ||
        !(batch_images.extents(0) == batch_size) ||
        !(batch_images.extents(1) == dataset_images.extents(1)) ||
        !(batch_images.extents(2) == dataset_images.extents(2)) ||
        !(batch_images.extents(3) == dataset_images.extents(3)) ||
        !(batch_labels.extents(0) == batch_size)
    )
    {
        throw_with_context<std::domain_error>("Domain error. Check source location.");
    }

    for(int64_t i = 0; i < batch_size; i++)
    {
        batch_labels[i] = dataset_labels[dataset_permutation[dataset_cursor + i]];
        for(int64_t j = 0; j < batch_images.extents(1); j++)
        for(int64_t k = 0; k < batch_images.extents(2); k++)
        for(int64_t l = 0; l < batch_images.extents(3); l++)
            batch_images[i, j, k, l] = dataset_images[dataset_permutation[dataset_cursor + i], j, k, l];
    }

    dataset_cursor += batch_size;

    if(dataset_cursor == dataset_images.extents(0))
    {
        dataset_cursor = 0;

        std::shuffle (
            dataset_permutation.p_elements(),
            dataset_permutation.p_elements() + dataset_permutation.size(), m_prng
        );
    }
}
