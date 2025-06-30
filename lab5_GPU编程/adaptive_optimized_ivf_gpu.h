#pragma once

#include <vector>
#include <queue>
#include <cstdint>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <cfloat>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cublas_v2.h>
#include <thrust/scan.h>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>

// 优化的错误检查宏
#ifndef CUDA_CHECK
#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        printf("CUDA error at %s:%d - %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(1); \
    } \
} while(0)
#endif

#ifndef CUBLAS_CHECK
#define CUBLAS_CHECK(call) do { \
    cublasStatus_t stat = call; \
    if (stat != CUBLAS_STATUS_SUCCESS) { \
        printf("cuBLAS error at %s:%d\n", __FILE__, __LINE__); \
        exit(1); \
    } \
} while(0)
#endif

// 内存自适应的GPU IVF实现
class AdaptiveOptimizedIVFGPU {
private:
    // GPU内存指针
    float* d_base_data;
    float* d_centroids;
    float* d_queries;
    float* d_centroid_distances;
    float* d_point_distances;
    uint32_t* d_selected_points;
    uint32_t* d_results;
    float* d_result_distances;
    int* d_top_clusters_indices;
    uint32_t* d_inverted_lists_data;
    int* d_inverted_lists_offsets;
    int* d_query_point_counts;
    int* d_query_point_offsets;
    
    // 多个内存池用于不同大小的请求
    struct MemoryPool {
        uint32_t* d_points;
        float* d_distances;
        size_t capacity;
        bool in_use;
    };
    std::vector<MemoryPool> memory_pools;
    
    // cuBLAS句柄
    cublasHandle_t cublas_handle;
    
    // 参数
    int num_base_vectors;
    int vector_dim;
    int num_clusters;
    int max_batch_size;
    size_t base_memory_pool_size;
    size_t available_gpu_memory;
    
    // CPU端数据
    std::vector<float> h_centroids;
    std::vector<std::vector<uint32_t>> h_inverted_lists;

public:
    AdaptiveOptimizedIVFGPU(float* base_data, int n_base, int dim, int n_clusters, int batch_size = 128)
        : num_base_vectors(n_base), vector_dim(dim), num_clusters(n_clusters), 
          max_batch_size(batch_size) {
        
        // 获取GPU内存信息并动态调整
        size_t free_mem, total_mem;
        CUDA_CHECK(cudaMemGetInfo(&free_mem, &total_mem));
        available_gpu_memory = free_mem * 0.8; // 保留20%作为安全边界
        
        printf("Available GPU memory: %zu MB\n", available_gpu_memory / (1024*1024));
        
        // 基础内存占用估算
        size_t basic_memory = n_base * dim * sizeof(float) + // base data
                             n_clusters * dim * sizeof(float) + // centroids  
                             batch_size * dim * sizeof(float) + // queries
                             batch_size * n_clusters * sizeof(float) + // centroid distances
                             batch_size * 100 * (sizeof(uint32_t) + sizeof(float)) + // results
                             batch_size * 64 * sizeof(int); // top clusters
        
        size_t remaining_memory = available_gpu_memory - basic_memory;
        base_memory_pool_size = remaining_memory * 0.9 / (sizeof(uint32_t) + sizeof(float));
        
        printf("Base memory pool size: %zu points\n", base_memory_pool_size);
        
        // 初始化cuBLAS
        CUBLAS_CHECK(cublasCreate(&cublas_handle));
        CUBLAS_CHECK(cublasSetMathMode(cublas_handle, CUBLAS_TENSOR_OP_MATH));
        
        // 预分配所有GPU内存
        allocate_gpu_memory();
        
        // 复制基准数据到GPU
        CUDA_CHECK(cudaMemcpy(d_base_data, base_data, 
                             n_base * dim * sizeof(float), cudaMemcpyHostToDevice));
        
        // 在CPU上构建索引
        build_index_cpu(base_data);
        
        // 复制质心和倒排列表到GPU
        CUDA_CHECK(cudaMemcpy(d_centroids, h_centroids.data(), 
                             n_clusters * dim * sizeof(float), cudaMemcpyHostToDevice));
        upload_inverted_lists_to_gpu();
        
        printf("AdaptiveOptimizedIVFGPU initialized successfully\n");
    }
    
    ~AdaptiveOptimizedIVFGPU() {
        // 释放GPU内存
        if (d_base_data) cudaFree(d_base_data);
        if (d_centroids) cudaFree(d_centroids);
        if (d_queries) cudaFree(d_queries);
        if (d_centroid_distances) cudaFree(d_centroid_distances);
        if (d_point_distances) cudaFree(d_point_distances);
        if (d_selected_points) cudaFree(d_selected_points);
        if (d_results) cudaFree(d_results);
        if (d_result_distances) cudaFree(d_result_distances);
        if (d_top_clusters_indices) cudaFree(d_top_clusters_indices);
        if (d_inverted_lists_data) cudaFree(d_inverted_lists_data);
        if (d_inverted_lists_offsets) cudaFree(d_inverted_lists_offsets);
        if (d_query_point_counts) cudaFree(d_query_point_counts);
        if (d_query_point_offsets) cudaFree(d_query_point_offsets);
        
        // 释放内存池
        for (auto& pool : memory_pools) {
            if (pool.d_points) cudaFree(pool.d_points);
            if (pool.d_distances) cudaFree(pool.d_distances);
        }
        
        if (cublas_handle) cublasDestroy(cublas_handle);
    }

private:
    void allocate_gpu_memory() {
        // 基本数据
        CUDA_CHECK(cudaMalloc(&d_base_data, num_base_vectors * vector_dim * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_centroids, num_clusters * vector_dim * sizeof(float)));
        
        // 批处理内存
        CUDA_CHECK(cudaMalloc(&d_queries, max_batch_size * vector_dim * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_centroid_distances, max_batch_size * num_clusters * sizeof(float)));
        
        // 结果内存
        CUDA_CHECK(cudaMalloc(&d_results, max_batch_size * 100 * sizeof(uint32_t)));
        CUDA_CHECK(cudaMalloc(&d_result_distances, max_batch_size * 100 * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_top_clusters_indices, max_batch_size * 64 * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_query_point_counts, max_batch_size * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_query_point_offsets, (max_batch_size + 1) * sizeof(int)));
        
        // 创建多个不同大小的内存池
        std::vector<size_t> pool_sizes = {
            base_memory_pool_size / 4,     // 小池
            base_memory_pool_size / 2,     // 中池  
            base_memory_pool_size,         // 大池
            base_memory_pool_size * 2      // 超大池
        };
        
        memory_pools.resize(pool_sizes.size());
        for (size_t i = 0; i < pool_sizes.size(); ++i) {
            try {
                CUDA_CHECK(cudaMalloc(&memory_pools[i].d_points, pool_sizes[i] * sizeof(uint32_t)));
                CUDA_CHECK(cudaMalloc(&memory_pools[i].d_distances, pool_sizes[i] * sizeof(float)));
                memory_pools[i].capacity = pool_sizes[i];
                memory_pools[i].in_use = false;
                printf("Allocated memory pool %zu with capacity: %zu points\n", i, pool_sizes[i]);
            } catch (...) {
                printf("Failed to allocate memory pool %zu, skipping\n", i);
                memory_pools[i].d_points = nullptr;
                memory_pools[i].d_distances = nullptr;
                memory_pools[i].capacity = 0;
                memory_pools[i].in_use = false;
            }
        }
    }
    
    // 获取合适的内存池
    MemoryPool* get_memory_pool(size_t required_size) {
        // 找到第一个足够大且未使用的池
        for (auto& pool : memory_pools) {
            if (!pool.in_use && pool.capacity >= required_size && pool.d_points != nullptr) {
                pool.in_use = true;
                return &pool;
            }
        }
        return nullptr;
    }
    
    // 释放内存池
    void release_memory_pool(MemoryPool* pool) {
        if (pool) {
            pool->in_use = false;
        }
    }

    void upload_inverted_lists_to_gpu() {
        std::vector<uint32_t> all_lists_data;
        std::vector<int> offsets(num_clusters + 1, 0);
        
        for (int i = 0; i < num_clusters; ++i) {
            offsets[i] = all_lists_data.size();
            all_lists_data.insert(all_lists_data.end(), h_inverted_lists[i].begin(), h_inverted_lists[i].end());
        }
        offsets[num_clusters] = all_lists_data.size();

        if (all_lists_data.size() > 0) {
            CUDA_CHECK(cudaMalloc(&d_inverted_lists_data, all_lists_data.size() * sizeof(uint32_t)));
            CUDA_CHECK(cudaMemcpy(d_inverted_lists_data, all_lists_data.data(), 
                                 all_lists_data.size() * sizeof(uint32_t), cudaMemcpyHostToDevice));
        }
        
        CUDA_CHECK(cudaMalloc(&d_inverted_lists_offsets, (num_clusters + 1) * sizeof(int)));
        CUDA_CHECK(cudaMemcpy(d_inverted_lists_offsets, offsets.data(), 
                             (num_clusters + 1) * sizeof(int), cudaMemcpyHostToDevice));
    }
    
    void build_index_cpu(float* base_data) {
        // 使用更好的k-means++算法（与之前相同）
        h_centroids.resize(num_clusters * vector_dim);
        h_inverted_lists.resize(num_clusters);
        
        // k-means++初始化
        std::vector<bool> selected(num_base_vectors, false);
        srand(42);
        int first_centroid = rand() % num_base_vectors;
        selected[first_centroid] = true;
        std::copy(base_data + first_centroid * vector_dim,
                 base_data + (first_centroid + 1) * vector_dim,
                 h_centroids.begin());
        
        // k-means++选择剩余质心
        for (int c = 1; c < num_clusters; ++c) {
            std::vector<float> distances(num_base_vectors);
            float total_distance = 0.0f;
            
            #pragma omp parallel for reduction(+:total_distance)
            for (int i = 0; i < num_base_vectors; ++i) {
                if (selected[i]) {
                    distances[i] = 0.0f;
                    continue;
                }
                
                float min_dist = FLT_MAX;
                for (int prev_c = 0; prev_c < c; ++prev_c) {
                    float dist = 0.0f;
                    for (int d = 0; d < vector_dim; ++d) {
                        float diff = base_data[i * vector_dim + d] - h_centroids[prev_c * vector_dim + d];
                        dist += diff * diff;
                    }
                    min_dist = std::min(min_dist, dist);
                }
                distances[i] = min_dist;
                total_distance += min_dist;
            }
            
            // 按概率选择下一个质心
            float rand_val = ((float)rand() / RAND_MAX) * total_distance;
            float cumulative = 0.0f;
            int next_centroid = 0;
            
            for (int i = 0; i < num_base_vectors; ++i) {
                cumulative += distances[i];
                if (cumulative >= rand_val) {
                    next_centroid = i;
                    break;
                }
            }
            
            selected[next_centroid] = true;
            std::copy(base_data + next_centroid * vector_dim,
                     base_data + (next_centroid + 1) * vector_dim,
                     h_centroids.begin() + c * vector_dim);
        }
        
        // 迭代优化质心（并行化）
        for (int iter = 0; iter < 10; ++iter) {
            std::vector<int> assignments(num_base_vectors);
            
            // 并行分配点到最近质心
            #pragma omp parallel for
            for (int i = 0; i < num_base_vectors; ++i) {
                float min_dist = FLT_MAX;
                int best_cluster = 0;
                
                for (int c = 0; c < num_clusters; ++c) {
                    float dist = 0.0f;
                    for (int d = 0; d < vector_dim; ++d) {
                        float diff = base_data[i * vector_dim + d] - h_centroids[c * vector_dim + d];
                        dist += diff * diff;
                    }
                    
                    if (dist < min_dist) {
                        min_dist = dist;
                        best_cluster = c;
                    }
                }
                assignments[i] = best_cluster;
            }
            
            // 更新质心
            std::vector<std::vector<float>> new_centroids(num_clusters, std::vector<float>(vector_dim, 0.0f));
            std::vector<int> counts(num_clusters, 0);
            
            for (int i = 0; i < num_base_vectors; ++i) {
                int cluster = assignments[i];
                counts[cluster]++;
                for (int d = 0; d < vector_dim; ++d) {
                    new_centroids[cluster][d] += base_data[i * vector_dim + d];
                }
            }
            
            for (int c = 0; c < num_clusters; ++c) {
                if (counts[c] > 0) {
                    for (int d = 0; d < vector_dim; ++d) {
                        h_centroids[c * vector_dim + d] = new_centroids[c][d] / counts[c];
                    }
                }
            }
        }
        
        // 最终分配构建倒排列表
        for (int c = 0; c < num_clusters; ++c) {
            h_inverted_lists[c].clear();
        }
        
        for (int i = 0; i < num_base_vectors; ++i) {
            float min_dist = FLT_MAX;
            int best_cluster = 0;
            
            for (int c = 0; c < num_clusters; ++c) {
                float dist = 0.0f;
                for (int d = 0; d < vector_dim; ++d) {
                    float diff = base_data[i * vector_dim + d] - h_centroids[c * vector_dim + d];
                    dist += diff * diff;
                }
                
                if (dist < min_dist) {
                    min_dist = dist;
                    best_cluster = c;
                }
            }
            h_inverted_lists[best_cluster].push_back(i);
        }
    }

public:
    // 自适应批量搜索 - 根据内存需求动态调整
    std::vector<std::vector<std::pair<float, uint32_t>>> batch_search(
        const std::vector<float>& queries, int k, int nprobe) {
        
        int query_num = queries.size() / vector_dim;
        std::vector<std::vector<std::pair<float, uint32_t>>> results(query_num);
        
        // 估算内存需求
        size_t avg_points_per_cluster = num_base_vectors / num_clusters;
        size_t estimated_points_per_batch = max_batch_size * avg_points_per_cluster * nprobe;
        
        // 根据内存需求动态调整批处理大小
        int adaptive_batch_size = max_batch_size;
        MemoryPool* selected_pool = get_memory_pool(estimated_points_per_batch);
        
        if (!selected_pool) {
            // 如果没有足够的内存池，减小批处理大小
            adaptive_batch_size = std::max(1, max_batch_size / 4);
            estimated_points_per_batch = adaptive_batch_size * avg_points_per_cluster * nprobe;
            selected_pool = get_memory_pool(estimated_points_per_batch);
            
            if (!selected_pool) {
                printf("Warning: No suitable memory pool found, using fallback\n");
                // 使用最小的可用池
                for (auto& pool : memory_pools) {
                    if (!pool.in_use && pool.d_points != nullptr) {
                        selected_pool = &pool;
                        pool.in_use = true;
                        break;
                    }
                }
            }
        }
        
        printf("Using adaptive batch size: %d, estimated points: %zu\n", 
               adaptive_batch_size, estimated_points_per_batch);
        
        // 处理批量查询
        for (int batch_start = 0; batch_start < query_num; batch_start += adaptive_batch_size) {
            int batch_end = std::min(batch_start + adaptive_batch_size, query_num);
            int current_batch_size = batch_end - batch_start;
            
            // 复制查询数据到GPU
            CUDA_CHECK(cudaMemcpy(d_queries, queries.data() + batch_start * vector_dim,
                                  current_batch_size * vector_dim * sizeof(float), 
                                  cudaMemcpyHostToDevice));
            
            // 使用cuBLAS计算查询到质心的距离
            const float alpha = 1.0f, beta = 0.0f;
            CUBLAS_CHECK(cublasSgemm(cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N,
                                    num_clusters, current_batch_size, vector_dim,
                                    &alpha,
                                    d_centroids, vector_dim,
                                    d_queries, vector_dim,
                                    &beta,
                                    d_centroid_distances, num_clusters));
            
            // 转换内积为距离
            int total_dists = current_batch_size * num_clusters;
            int threads_per_block_dist = 256;
            int blocks_dist = (total_dists + threads_per_block_dist - 1) / threads_per_block_dist;
            
            convert_inner_product_to_distance<<<blocks_dist, threads_per_block_dist>>>(
                d_centroid_distances, total_dists);
            CUDA_CHECK(cudaGetLastError());

            // 使用选定的内存池处理批次
            if (selected_pool && selected_pool->d_points) {
                process_batch_gpu_adaptive(d_queries, current_batch_size, k, nprobe, selected_pool);
            } else {
                printf("Warning: No memory pool available for processing\n");
                continue;
            }
            
            // 复制最终结果回CPU
            std::vector<uint32_t> h_results(current_batch_size * k);
            std::vector<float> h_result_distances(current_batch_size * k);
            CUDA_CHECK(cudaMemcpy(h_results.data(), d_results, 
                                  current_batch_size * k * sizeof(uint32_t), 
                                  cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(h_result_distances.data(), d_result_distances, 
                                  current_batch_size * k * sizeof(float), 
                                  cudaMemcpyDeviceToHost));

            for (int i = 0; i < current_batch_size; ++i) {
                results[batch_start + i].reserve(k);
                for (int j = 0; j < k; ++j) {
                    if (h_result_distances[i * k + j] != FLT_MAX) {
                        results[batch_start + i].push_back({h_result_distances[i * k + j], h_results[i * k + j]});
                    }
                }
            }
        }
        
        // 释放内存池
        release_memory_pool(selected_pool);
        
        return results;
    }

private:
    void process_batch_gpu_adaptive(const float* d_queries, int batch_size, int k, int nprobe, MemoryPool* pool) {
        // 使用动态分配的内存池
        d_selected_points = pool->d_points;
        d_point_distances = pool->d_distances;
        size_t max_points = pool->capacity;
        
        // 1. 在GPU上找到每个查询的top nprobe个簇
        find_top_n_clusters_fixed<<<batch_size, 1>>>(
            d_centroid_distances, d_top_clusters_indices, 
            batch_size, num_clusters, nprobe);
        CUDA_CHECK(cudaGetLastError());

        // 2. 计算每个查询需要收集的点数
        calculate_gather_offsets_fixed<<<(batch_size + 255) / 256, 256>>>(
            d_top_clusters_indices, d_inverted_lists_offsets, 
            d_query_point_counts, batch_size, nprobe, num_clusters);
        CUDA_CHECK(cudaGetLastError());

        // 3. 使用Thrust执行前缀和来获取偏移量
        thrust::exclusive_scan(thrust::device, 
                               d_query_point_counts, 
                               d_query_point_counts + batch_size, 
                               d_query_point_offsets, 
                               0);

        // 4. 检查总选择点数
        int total_selected_points = 0;
        if (batch_size > 0) {
            int last_offset, last_count;
            CUDA_CHECK(cudaMemcpy(&last_offset, d_query_point_offsets + batch_size - 1, 
                                  sizeof(int), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(&last_count, d_query_point_counts + batch_size - 1, 
                                  sizeof(int), cudaMemcpyDeviceToHost));
            
            total_selected_points = last_offset + last_count;
            CUDA_CHECK(cudaMemcpy(d_query_point_offsets + batch_size, &total_selected_points, 
                                  sizeof(int), cudaMemcpyHostToDevice));
        }

        if (total_selected_points > max_points) {
            printf("Warning: Still truncating points. Required: %d, Allocated: %zu\n", 
                   total_selected_points, max_points);
            // 继续处理，但结果可能不完整
        }
        
        // 5. 在GPU上收集所有候选项
        gather_points_fixed<<<(batch_size + 255) / 256, 256>>>(
            d_top_clusters_indices, d_inverted_lists_offsets, d_inverted_lists_data, 
            d_query_point_offsets, d_selected_points, batch_size, nprobe, num_clusters);
        CUDA_CHECK(cudaGetLastError());

        // 6. 计算到这些点的距离
        size_t shared_mem_size = vector_dim * sizeof(float);
        optimized_point_distances<<<batch_size, 256, shared_mem_size>>>(
            d_base_data, d_queries, d_selected_points, d_query_point_offsets,
            d_point_distances, batch_size, vector_dim);
        CUDA_CHECK(cudaGetLastError());

        // 7. GPU Top-K 选择
        batch_gpu_topk_selection_fixed<<<batch_size, 256>>>(
            d_point_distances, d_selected_points, d_query_point_offsets,
            d_results, d_result_distances, batch_size, k);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
    }

public:
    // 单个查询搜索
    std::priority_queue<std::pair<float, uint32_t>> search(const float* query, int k, int nprobe) {
        std::vector<float> query_vec(query, query + vector_dim);
        auto results = batch_search(query_vec, k, nprobe);
        
        std::priority_queue<std::pair<float, uint32_t>> pq;
        if (!results.empty()) {
            for (const auto& pair : results[0]) {
                pq.push(pair);
            }
        }
        return pq;
    }
};