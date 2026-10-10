#include "ivf_c_api.h"
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <ctime>
#include <cstring>
#include <limits>
#include <mutex>
#include <omp.h>
#include <queue>
#include <unordered_map>
#include <vector>
#ifdef __ARM_NEON
#include <arm_neon.h>
#endif
#ifdef __AVX2__
#include <immintrin.h>
#elif defined(__SSE2__)
#include <emmintrin.h>
#include <xmmintrin.h>
#endif

struct IVFIndex {
  size_t dim;
  size_t bucket_num;
  // bucket id list
  std::vector<int> bucket_id_list;
  // bucket id to point id list
  std::unordered_map<int, std::vector<size_t>> bucket_id_to_point_id;
  // point id to data
  std::unordered_map<size_t, std::vector<float>> point_id_to_data;
  // HAKES场景标志：如果所有bucket都只有1个向量，则为true（在加载数据时检测）
  bool is_hakes_scenario;
};

IVFIndex *ivf_new(size_t dim, int *bucket_id_list, size_t bucket_num) {
  auto index = new IVFIndex();
  index->dim = dim;
  index->bucket_num = bucket_num;
  index->bucket_id_list =
      std::vector<int>(bucket_id_list, bucket_id_list + bucket_num);
  index->bucket_id_to_point_id.clear();
  for (auto bucket_id : index->bucket_id_list) {
    index->bucket_id_to_point_id[bucket_id] = std::vector<size_t>();
  }
  // 初始化为true，如果后续发现任何bucket有超过1个向量，则设为false
  index->is_hakes_scenario = true;
  return index;
}

void ivf_free(IVFIndex *index) {
  if (index) {
    index->bucket_id_to_point_id.clear();
    index->point_id_to_data.clear();
    delete index;
  }
}

int ivf_add_point(IVFIndex *index, int bucket_id, const float *point,
                  size_t point_id) {
  if (!index || !point) {
    return -1;
  }

  try {
    auto bucket_it = index->bucket_id_to_point_id.find(bucket_id);
    if (bucket_it == index->bucket_id_to_point_id.end()) {
      // 如果桶不存在，创建新桶
      index->bucket_id_to_point_id[bucket_id] = std::vector<size_t>();
    }

    std::vector<float> point_data(point, point + index->dim);

    index->bucket_id_to_point_id[bucket_id].push_back(point_id);
    index->point_id_to_data[point_id] = std::move(point_data);
    
    // 检测HAKES场景：如果当前bucket已经有超过1个向量，则不是HAKES场景
    if (index->bucket_id_to_point_id[bucket_id].size() > 1) {
      index->is_hakes_scenario = false;
    }
    
    return 0;
  } catch (const std::exception &) {
    return -1;
  }
}

inline float compute_l2_distance(const float *a, const float *b, size_t dim) {
  float distance = 0.0f;
  for (size_t i = 0; i < dim; i++) {
    float diff = a[i] - b[i];
    distance += diff * diff;
  }
  return distance;
}

// Enhanced SIMD distance calculation with multi-platform support
inline float compute_l2_distance_simd(const float *a, const float *b, size_t dim) {
#ifdef __AVX2__
  // AVX2 implementation - process 8 floats at once
  __m256 sum = _mm256_setzero_ps();
  size_t d = 0;
  
  for (; d + 8 <= dim; d += 8) {
    __m256 va = _mm256_loadu_ps(a + d);
    __m256 vb = _mm256_loadu_ps(b + d);
    __m256 diff = _mm256_sub_ps(va, vb);
    sum = _mm256_fmadd_ps(diff, diff, sum);
  }
  
  // Horizontal sum
  __m128 sum128 = _mm_add_ps(_mm256_castps256_ps128(sum), _mm256_extractf128_ps(sum, 1));
  sum128 = _mm_hadd_ps(sum128, sum128);
  sum128 = _mm_hadd_ps(sum128, sum128);
  float distance = _mm_cvtss_f32(sum128);
  
  // Process remaining elements
  for (; d < dim; d++) {
    float diff = a[d] - b[d];
    distance += diff * diff;
  }
  
  return distance;

#elif defined(__SSE2__)
  // SSE2 implementation - process 4 floats at once  
  __m128 sum = _mm_setzero_ps();
  size_t d = 0;
  
  for (; d + 4 <= dim; d += 4) {
    __m128 va = _mm_loadu_ps(a + d);
    __m128 vb = _mm_loadu_ps(b + d);
    __m128 diff = _mm_sub_ps(va, vb);
    sum = _mm_add_ps(sum, _mm_mul_ps(diff, diff));
  }
  
  // Horizontal sum
  sum = _mm_hadd_ps(sum, sum);
  sum = _mm_hadd_ps(sum, sum);
  float distance = _mm_cvtss_f32(sum);
  
  // Process remaining elements
  for (; d < dim; d++) {
    float diff = a[d] - b[d];
    distance += diff * diff;
  }
  
  return distance;

#elif defined(__ARM_NEON)
  // ARM NEON implementation (existing code)
  float32x4_t sum = vdupq_n_f32(0);
  size_t d = 0;

  for (; d + 4 <= dim; d += 4) {
    float32x4_t va = vld1q_f32(a + d);
    float32x4_t vb = vld1q_f32(b + d);
    float32x4_t diff = vsubq_f32(va, vb);
    sum = vmlaq_f32(sum, diff, diff);
  }

  float32x2_t sum2 = vadd_f32(vget_low_f32(sum), vget_high_f32(sum));
  sum2 = vpadd_f32(sum2, sum2);
  float distance = vget_lane_f32(sum2, 0);

  for (; d < dim; d++) {
    float diff = a[d] - b[d];
    distance += diff * diff;
  }

  return distance;
#else
  // Fallback to standard implementation
  float distance = 0.0f;
  for (size_t d = 0; d < dim; d++) {
    float diff = a[d] - b[d];
    distance += diff * diff;
  }
  return distance;
#endif
}

// Batch distance calculation for better cache efficiency and SIMD utilization
inline void compute_batch_l2_distances_simd(const float *query, 
                                           const std::vector<const float*>& points,
                                           const std::vector<size_t>& point_ids,
                                           size_t dim,
                                           std::vector<std::pair<float, size_t>>& results) {
  results.reserve(results.size() + points.size());
  
  for (size_t i = 0; i < points.size(); ++i) {
    float dist = compute_l2_distance_simd(query, points[i], dim);
    results.emplace_back(dist, point_ids[i]);
  }
}

// Memory-aligned vector storage for better SIMD performance
struct AlignedVector {
  std::vector<float> data;
  
  AlignedVector(const float* src, size_t size) {
    // Ensure 32-byte alignment for AVX operations
    data.resize(size);
    std::memcpy(data.data(), src, size * sizeof(float));
  }
  
  const float* get() const { return data.data(); }
};

void ivf_search_knn(IVFIndex *index, const float *query, const int *bucket_ids,
                    size_t k, size_t *labels, float *distances) {
  try {
    if (!index || !query || !bucket_ids || k <= 0 || !labels || !distances) {
      printf("Invalid input parameters\n");
      return;
    }

    // Count number of buckets
    int num_buckets = 0;
    for (; bucket_ids[num_buckets] != -1; num_buckets++)
      ;

    // 使用最大堆（priority_queue）只保留top-k结果，避免存储所有候选点导致内存飙升
    // 堆中存储 (distance, point_id)，使用greater比较器使其成为最大堆（距离大的在顶部）
    // 当堆大小超过k时，弹出最大距离的元素，始终保持最小的k个距离
    std::priority_queue<std::pair<float, size_t>, 
                       std::vector<std::pair<float, size_t>>,
                       std::less<std::pair<float, size_t>>> topk_heap;
    std::mutex heap_mutex; // Mutex for thread-safe heap access (仅在多线程时使用)

    auto start = std::chrono::high_resolution_clock::now();

    // 辅助函数：添加结果到top-k堆
    auto add_to_heap = [&](float dist, size_t point_id) {
      if (topk_heap.size() < k) {
        topk_heap.emplace(dist, point_id);
      } else if (dist < topk_heap.top().first) {
        // 如果新距离更小，替换堆顶的最大距离
        topk_heap.pop();
        topk_heap.emplace(dist, point_id);
      }
    };

    // HAKES场景优化：单线程处理，减少内存占用和锁开销
    // is_hakes_scenario在加载数据时已检测并存储，无需每次查询都检测
    if (index->is_hakes_scenario) {
      // 单线程处理：直接添加到topk_heap，无需local_results和mutex
      for (int i = 0; i < num_buckets; i++) {
        try {
          int bucket_id = bucket_ids[i];
          auto bucket_it = index->bucket_id_to_point_id.find(bucket_id);
          if (bucket_it == index->bucket_id_to_point_id.end()) {
            printf("Bucket %d not found\n", bucket_id);
            continue;
          }

          // HAKES场景：bucket只有1个向量，直接用bucket_id作为point_id
          size_t point_id = static_cast<size_t>(bucket_id);
          auto it = index->point_id_to_data.find(point_id);
          if (it != index->point_id_to_data.end() && 
              it->second.size() == index->dim) {
            float dist = compute_l2_distance_simd(
                query, it->second.data(), index->dim);
            add_to_heap(dist, point_id);
          } else {
            // Fallback: 使用bucket中唯一的point_id
            if (!bucket_it->second.empty()) {
              size_t actual_point_id = bucket_it->second[0];
              auto fallback_it = index->point_id_to_data.find(actual_point_id);
              if (fallback_it != index->point_id_to_data.end() &&
                  fallback_it->second.size() == index->dim) {
                float dist = compute_l2_distance_simd(
                    query, fallback_it->second.data(), index->dim);
                add_to_heap(dist, actual_point_id);
              }
            }
          }
        } catch (const std::exception &e) {
          printf("Error processing bucket %d: %s\n", bucket_ids[i], e.what());
          continue;
        }
      }
    } else {
      // 多线程处理：标准场景（bucket中有多个向量）
#pragma omp parallel
      {
        // Thread-local storage for results (使用本地堆减少锁竞争)
        std::priority_queue<std::pair<float, size_t>, 
                           std::vector<std::pair<float, size_t>>,
                           std::less<std::pair<float, size_t>>> local_heap;

#pragma omp for schedule(dynamic)
        for (int i = 0; i < num_buckets; i++) {
          try {
            int bucket_id = bucket_ids[i];
            auto bucket_it = index->bucket_id_to_point_id.find(bucket_id);
            if (bucket_it == index->bucket_id_to_point_id.end()) {
#pragma omp critical
              {
                printf("Bucket %d not found\n", bucket_id);
              }
              continue;
            }

            // 优化：如果bucket只有1个向量，直接用bucket_id作为point_id，减少map查找和循环开销
            if (bucket_it->second.size() == 1) {
              // HAKES场景优化：bucket_id可以直接作为point_id
              size_t point_id = static_cast<size_t>(bucket_id);
              try {
                auto it = index->point_id_to_data.find(point_id);
                if (it != index->point_id_to_data.end() && 
                    it->second.size() == index->dim) {
                  float dist = compute_l2_distance_simd(
                      query, it->second.data(), index->dim);
                  if (local_heap.size() < k) {
                    local_heap.emplace(dist, point_id);
                  } else if (dist < local_heap.top().first) {
                    local_heap.pop();
                    local_heap.emplace(dist, point_id);
                  }
                } else {
                  // Fallback: 如果bucket_id != point_id，使用bucket中唯一的point_id
                  size_t actual_point_id = bucket_it->second[0];
                  auto fallback_it = index->point_id_to_data.find(actual_point_id);
                  if (fallback_it != index->point_id_to_data.end() &&
                      fallback_it->second.size() == index->dim) {
                    float dist = compute_l2_distance_simd(
                        query, fallback_it->second.data(), index->dim);
                    if (local_heap.size() < k) {
                      local_heap.emplace(dist, actual_point_id);
                    } else if (dist < local_heap.top().first) {
                      local_heap.pop();
                      local_heap.emplace(dist, actual_point_id);
                    }
                  }
                }
              } catch (const std::exception &e) {
                // 如果异常，fallback到标准流程
                size_t actual_point_id = bucket_it->second[0];
                auto fallback_it = index->point_id_to_data.find(actual_point_id);
                if (fallback_it != index->point_id_to_data.end() &&
                    fallback_it->second.size() == index->dim) {
                  float dist = compute_l2_distance_simd(
                      query, fallback_it->second.data(), index->dim);
                  if (local_heap.size() < k) {
                    local_heap.emplace(dist, actual_point_id);
                  } else if (dist < local_heap.top().first) {
                    local_heap.pop();
                    local_heap.emplace(dist, actual_point_id);
                  }
                }
              }
            } else {
              // 标准流程：遍历bucket中的所有点
              for (size_t point_id : bucket_it->second) {
                try {
                  auto it = index->point_id_to_data.find(point_id);
                  if (it == index->point_id_to_data.end()) {
#pragma omp critical
                    {
                      printf("Point %zu not found\n", point_id);
                    }
                    continue;
                  }

                  // Verify vector size
                  if (it->second.size() != index->dim) {
#pragma omp critical
                    {
                      printf("Vector dimension mismatch: expected %zu, got %zu\n",
                             index->dim, it->second.size());
                    }
                    continue;
                  }

                  // 使用优化的 L2 距离计算，直接使用查询向量指针
                  float dist = compute_l2_distance_simd(
                      query, it->second.data(), index->dim);
                  if (local_heap.size() < k) {
                    local_heap.emplace(dist, point_id);
                  } else if (dist < local_heap.top().first) {
                    local_heap.pop();
                    local_heap.emplace(dist, point_id);
                  }
                } catch (const std::exception &e) {
#pragma omp critical
                  {
                    printf("Error processing point %zu: %s\n", point_id, e.what());
                  }
                  continue;
                }
              }
            }
          } catch (const std::exception &e) {
#pragma omp critical
            {
              printf("Error processing bucket %d: %s\n", bucket_ids[i], e.what());
            }
            continue;
          }
        }

        // Merge thread-local heap into global heap
        if (!local_heap.empty()) {
          std::lock_guard<std::mutex> lock(heap_mutex);
          while (!local_heap.empty()) {
            auto top = local_heap.top();
            local_heap.pop();
            add_to_heap(top.first, top.second);
          }
          // local_heap已经清空，手动释放底层容器内存
          std::priority_queue<std::pair<float, size_t>, 
                             std::vector<std::pair<float, size_t>>,
                             std::less<std::pair<float, size_t>>>().swap(local_heap);
        }
      }
    }

    // 从堆中提取top-k结果（按距离从小到大排序）
    size_t num_results = std::min(k, topk_heap.size());
    {
      // 使用作用域限制sorted_results的生命周期，及时释放内存
      std::vector<std::pair<float, size_t>> sorted_results;
      sorted_results.reserve(num_results);
      while (!topk_heap.empty()) {
        sorted_results.push_back(topk_heap.top());
        topk_heap.pop();
      }
      // 堆是最大堆，需要反转顺序使其按距离从小到大
      std::reverse(sorted_results.begin(), sorted_results.end());

      // Fill in the top k results
      for (size_t i = 0; i < num_results; i++) {
        labels[i] = sorted_results[i].second;
        distances[i] = sorted_results[i].first;
      }
      // sorted_results在此作用域结束时自动释放内存
    }

    // Fill remaining slots with -1 and max float if k > number of results
    for (size_t i = num_results; i < k; i++) {
      labels[i] = -1;
      distances[i] = std::numeric_limits<float>::max();
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration =
        std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    // printf("Search time: %4f seconds\n", duration.count() / 1000.0);

  } catch (const std::exception &e) {
    printf("Fatal error in ivf_search_knn: %s\n", e.what());
  }
}

void ivf_get_bucket_size(IVFIndex *index, size_t *bucket_size) {
  if (!index) {
    return;
  }
  // 返回bucket数目和每一个bucket的size
  bucket_size[0] = index->bucket_num;
  for (size_t i = 0; i < index->bucket_num; i++) {
    bucket_size[i + 1] = index->bucket_id_to_point_id[i].size();
  }
}

int ivf_contains_bucket(IVFIndex *index, int bucket_id) {
  return index->bucket_id_to_point_id.find(bucket_id) != index->bucket_id_to_point_id.end();
}
