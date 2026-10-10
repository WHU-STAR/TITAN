#include "ivf_c_api.h"
#include <chrono>
#include <iostream>
#include <random>
#include <vector>

// 生成随机向量数据
std::vector<float> generate_random_vector(size_t dim, std::mt19937& gen) {
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<float> vec(dim);
    for (size_t i = 0; i < dim; ++i) {
        vec[i] = dist(gen);
    }
    return vec;
}

int main() {
    const size_t dim = 128;           // 向量维度
    const size_t num_points = 10000;  // 索引中的点数
    const size_t num_queries = 100;   // 查询数量
    const size_t k = 10;              // 返回最近邻数量
    
    std::random_device rd;
    std::mt19937 gen(rd());
    
    std::cout << "=== IVF SIMD 性能测试 ===" << std::endl;
    std::cout << "向量维度: " << dim << std::endl;
    std::cout << "索引点数: " << num_points << std::endl;
    std::cout << "查询数量: " << num_queries << std::endl;
    std::cout << "K值: " << k << std::endl;
    
    // 创建桶ID列表
    std::vector<int> bucket_ids = {0, 1, 2, 3, 4};
    
    // 创建IVF索引
    IVFIndex* index = ivf_new(dim, bucket_ids.data(), bucket_ids.size());
    
    // 添加随机点到索引
    std::cout << "\n正在生成并添加数据点..." << std::endl;
    auto start_build = std::chrono::high_resolution_clock::now();
    
    for (size_t i = 0; i < num_points; ++i) {
        auto point = generate_random_vector(dim, gen);
        int bucket_id = bucket_ids[i % bucket_ids.size()];
        ivf_add_point(index, bucket_id, point.data(), i);
    }
    
    auto end_build = std::chrono::high_resolution_clock::now();
    auto build_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_build - start_build);
    std::cout << "索引构建时间: " << build_time.count() << " 毫秒" << std::endl;
    
    // 生成查询向量
    std::vector<std::vector<float>> queries;
    for (size_t i = 0; i < num_queries; ++i) {
        queries.push_back(generate_random_vector(dim, gen));
    }
    
    // 性能测试
    std::cout << "\n开始查询性能测试..." << std::endl;
    
    std::vector<size_t> labels(k);
    std::vector<float> distances(k);
    std::vector<int> search_buckets = {0, 1, 2, -1}; // -1表示结束
    
    auto start_search = std::chrono::high_resolution_clock::now();
    
    for (size_t i = 0; i < num_queries; ++i) {
        ivf_search_knn(index, queries[i].data(), search_buckets.data(), 
                       k, labels.data(), distances.data());
    }
    
    auto end_search = std::chrono::high_resolution_clock::now();
    auto search_time = std::chrono::duration_cast<std::chrono::microseconds>(end_search - start_search);
    
    // 输出性能结果
    std::cout << "\n=== 性能结果 ===" << std::endl;
    std::cout << "总查询时间: " << search_time.count() << " 微秒" << std::endl;
    std::cout << "平均单次查询时间: " << search_time.count() / (double)num_queries << " 微秒" << std::endl;
    std::cout << "QPS (每秒查询数): " << (num_queries * 1000000.0) / search_time.count() << std::endl;
    
    // 输出编译标志信息
    std::cout << "\n=== SIMD 支持信息 ===" << std::endl;
#ifdef __AVX2__
    std::cout << "✓ AVX2 支持已启用" << std::endl;
#elif defined(__SSE2__)
    std::cout << "✓ SSE2 支持已启用" << std::endl;
#elif defined(__ARM_NEON)
    std::cout << "✓ ARM NEON 支持已启用" << std::endl;
#else
    std::cout << "⚠ 使用标准实现 (无SIMD优化)" << std::endl;
#endif

#ifdef _OPENMP
    std::cout << "✓ OpenMP 并行化已启用" << std::endl;
#else
    std::cout << "⚠ OpenMP 未启用" << std::endl;
#endif
    
    // 清理资源
    ivf_free(index);
    
    std::cout << "\n测试完成!" << std::endl;
    return 0;
} 