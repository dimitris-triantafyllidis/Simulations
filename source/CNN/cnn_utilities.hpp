#ifndef CNN_UTILITIES_HPP
#define CNN_UTILITIES_HPP

#include <random>

#include <Array.hpp>

class Dataset
{

public:

    Dataset();

    void load_MNIST   (const std::string &path);
    void load_CIFAR10 (const std::string &path);

    void get_training_batch (
        int64_t            batch_size,
        Array<float, 4>   &batch_images,
        Array<int64_t, 1> &batch_labels
    );

    void get_validation_batch (
        int64_t            batch_size,
        Array<float, 4>   &batch_images,
        Array<int64_t, 1> &batch_labels
    );

    void get_test_batch (
        int64_t            batch_size,
        Array<float, 4>   &batch_images,
        Array<int64_t, 1> &batch_labels
    );

    void get_batch (
        int64_t               batch_size,
        Array<float, 4>      &batch_images,
        Array<int64_t, 1>    &batch_labels,
        const Array<char, 4> &dataset_images,
        const Array<char, 1> &dataset_labels,
        int64_t              &dataset_cursor,
        Array<int64_t, 1>    &dataset_permutation
    );

private:

    Array<char, 4> m_training_images;
    Array<char, 4> m_validation_images;
    Array<char, 4> m_test_images;

    Array<char, 1> m_training_labels;
    Array<char, 1> m_validation_labels;
    Array<char, 1> m_test_labels;

    Array<int64_t, 1> m_training_permutation;
    Array<int64_t, 1> m_validation_permutation;
    Array<int64_t, 1> m_test_permutation;

    int64_t m_training_cursor;
    int64_t m_validation_cursor;
    int64_t m_test_cursor;

    std::mt19937 m_prng;

};

#endif // CNN_UTILITIES_HPP
