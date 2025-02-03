#include <stdio.h>

#include "ndarray.h"

void print_ndarray(ndarray_t* arr)
{
    for (uint32_t i = 0; i < arr->dimensions[0]; i++) {
        printf("[\n    ");
        for (uint32_t j = 0; j < arr->dimensions[1]; j++) {
            uint64_t idx[] = {i, j};
            printf("%.3f ", *(double*)ndarray_get_point(arr, idx));
        }
        printf("]\n");
    }
    printf("]\n");
}

int main()
{
    // Test 2D array creation
    uint64_t dims[] = {2, 3};
    ndarray_t* arr  = ndarray_create(dims, 2, 'd');
    printf("Created array with shape: [");
    for (uint32_t i = 0; i < arr->nd; i++) {
        printf("%llu%s", arr->dimensions[i], i < arr->nd - 1 ? ", " : "");
    }
    printf("]\n");

    // Test random array generation
    ndarray_t* rand_arr = ndarray_random_normal(3, 4, 2, 1.0, 'd');
    printf("Random array size: %llu\n", ndarray_size(rand_arr));
    printf("[\n  ");
    uint64_t idx[] = {0, 0};
    printf("====> bf: %.3f \n", *(double*)ndarray_get_point(rand_arr, idx));
    // ndarray_set_point_d(rand_arr, idx, -88888.0);
    double point = 88888.12345;
    ndarray_set_point(rand_arr, idx, &point);
    printf("====> af: %.3f \n", *(double*)ndarray_get_point(rand_arr, idx));
    print_ndarray(rand_arr);

    // Test transpose
    printf("test transpose:\n");
    ndarray_t* transposed = ndarray_transpose(rand_arr);
    printf("Transposed shape: [%llu, %llu]\n",
           transposed->dimensions[0],
           transposed->dimensions[1]);

    print_ndarray(transposed);

    printf("test subsample:\n");
    ndarray_t* subsample = ndarray_subsample(transposed, 2);
    print_ndarray(subsample);

    // Cleanup
    ndarray_free(arr);
    ndarray_free(rand_arr);
    ndarray_free(transposed);

    return 0;
}