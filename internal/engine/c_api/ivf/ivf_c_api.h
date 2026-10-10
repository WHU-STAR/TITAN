#ifndef IVF_C_API_H
#define IVF_C_API_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stddef.h>
#include <stdlib.h>

// Opaque type for IVF index
typedef struct IVFIndex IVFIndex;

// Create a new IVF index
IVFIndex *ivf_new(size_t dim, int *bucket_id_list, size_t bucket_num);

// Free the IVF index
void ivf_free(IVFIndex *index);

// Add a point to the index
int ivf_add_point(IVFIndex *index, int bucket_id, const float *point,
                  size_t point_id);

// Search for nearest neighbors
void ivf_search_knn(IVFIndex *index, const float *query, const int *bucket_ids,
                    size_t k, size_t *labels, float *distances);

// Get the bucket size
void ivf_get_bucket_size(IVFIndex *index, size_t *bucket_size);

// Check if the index contains the bucket
int ivf_contains_bucket(IVFIndex *index, int bucket_id);

// TODO: Save index to file
int ivf_save_index(IVFIndex *index, const char *path);

// TODO: Load index from file
IVFIndex *ivf_load_index(const char *path, size_t dim);

#ifdef __cplusplus
}
#endif

#endif // IVF_C_API_H
