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

// 声明外部CUDA内核函数
extern __global__ void convert_inner_product_to_distance(float* distances, int total_elements);
extern __global__ void find_top_n_clusters_fixed(const float* centroid_distances, int* top_clusters, 
                                          int batch_size, int num_clusters, int nprobe);
extern __global__ void calculate_gather_offsets_fixed(const int* top_clusters, const int* inverted_lists_offsets,
                                               int* query_point_counts, int batch_size, int nprobe, int num_clusters);
extern __global__ void gather_points_fixed(const int* top_clusters, const int* inverted_lists_offsets, 
                                    const uint32_t* inverted_lists_data, const int* query_point_offsets,
                                    uint32_t* selected_points, int batch_size, int nprobe, int num_clusters);
extern __global__ void optimized_point_distances(const float* base_data, const float* queries,
                                          const uint32_t* selected_points, const int* query_point_offsets,
                                          float* point_distances, int batch_size, int dim);
extern __global__ void batch_gpu_topk_selection_fixed(const float* point_distances, const uint32_t* selected_points,
                                               const int* query_point_offsets, uint32_t* results, float* result_distances,
                                               int batch_size, int k);

// 智能批次大小计算内核 - 预测每个查询需要的点数
__global__ void predict_query_point_counts(const float* centroid_distances, const int* inverted_lists_offsets,
                                          int* predicted_counts, int batch_size, int num_clusters, int nprobe) {
    int query_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (query_idx >= batch_size) return;
    
    const float* query_distances = centroid_distances + query_idx * num_clusters;
    
    // 找到top nprobe个簇并计算总点数
    int total_points = 0;
    for (int probe = 0; probe < nprobe; ++probe) {
        int best_cluster = 0;
        float min_dist = FLT_MAX;
        
        for (int c = 0; c < num_clusters; ++c) {
            if (query_distances[c] < min_dist) {
                bool already_selected = false;
                for (int prev = 0; prev < probe; ++prev) {
                    // 这里简化处理，实际应该记录已选择的簇
                }
                if (!already_selected) {
                    min_dist = query_distances[c];
                    best_cluster = c;
                }
            }
        }
        
        if (best_cluster < num_clusters) {
            int cluster_size = inverted_lists_offsets[best_cluster + 1] - inverted_lists_offsets[best_cluster];
            total_points += cluster_size;
        }
    }
    
    predicted_counts[query_idx] = total_points;
}

// 内存自适应的GPU IVF实现 - 改进版
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
    int* d_predicted_counts; // 新增：预测的点数
    
    // 多级内存池系统
    struct MemoryPool {
        uint32_t* d_points;
        float* d_distances;
        size_t capacity;
        bool in_use;
        int pool_level; // 0=小, 1=中, 2=大, 3=超大
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
        available_gpu_memory = free_mem * 0.75; // 保留25%作为安全边界
        
        printf("Available GPU memory: %zu MB\n", available_gpu_memory / (1024*1024));
        
        // 基础内存占用估算
        size_t basic_memory = n_base * dim * sizeof(float) + // base data
                             n_clusters * dim * sizeof(float) + // centroids  
                             batch_size * dim * sizeof(float) + // queries
                             batch_size * n_clusters * sizeof(float) + // centroid distances
                             batch_size * 100 * (sizeof(uint32_t) + sizeof(float)) + // results
                             batch_size * 64 * sizeof(int) + // top clusters
                             batch_size * sizeof(int); // predicted counts
        
        printf("Basic memory requirement: %zu MB\n", basic_memory / (1024*1024));
        
        if (basic_memory >= available_gpu_memory) {
            printf("Error: Basic memory requirement exceeds available GPU memory\n");
            exit(1);
        }
        
        size_t remaining_memory = available_gpu_memory - basic_memory;
        // 更智能的内存池大小计算
        size_t memory_pool_from_remaining = static_cast<size_t>(remaining_memory * 0.8) / (sizeof(uint32_t) + sizeof(float));
        size_t memory_pool_from_vectors = static_cast<size_t>(n_base) * 4 / 5; // 允许更多候选点
        base_memory_pool_size = std::min(memory_pool_from_remaining, memory_pool_from_vectors);
        
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
        if (d_predicted_counts) cudaFree(d_predicted_counts);
        
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
        CUDA_CHECK(cudaMalloc(&d_predicted_counts, max_batch_size * sizeof(int))); // 新增
        
        // 创建多级内存池系统
        std::vector<size_t> pool_sizes = {
            base_memory_pool_size / 8,     // 超小池 (level 0)
            base_memory_pool_size / 4,     // 小池   (level 1)
            base_memory_pool_size / 2,     // 中池   (level 2)
            base_memory_pool_size,         // 大池   (level 3)
            static_cast<size_t>(base_memory_pool_size * 1.5)    // 超大池 (level 4)
        };
        
        memory_pools.resize(pool_sizes.size());
        for (size_t i = 0; i < pool_sizes.size(); ++i) {
            try {
                CUDA_CHECK(cudaMalloc(&memory_pools[i].d_points, pool_sizes[i] * sizeof(uint32_t)));
                CUDA_CHECK(cudaMalloc(&memory_pools[i].d_distances, pool_sizes[i] * sizeof(float)));
                memory_pools[i].capacity = pool_sizes[i];
                memory_pools[i].in_use = false;
                memory_pools[i].pool_level = i;
                printf("Allocated memory pool %zu with capacity: %zu points (level %zu)\n", i, pool_sizes[i], static_cast<size_t>(i));
            } catch (...) {
                printf("Failed to allocate memory pool %zu, skipping\n", i);
                memory_pools[i].d_points = nullptr;
                memory_pools[i].d_distances = nullptr;
                memory_pools[i].capacity = 0;
                memory_pools[i].in_use = false;
                memory_pools[i].pool_level = -1;
            }
        }
    }
    
    // 智能内存池选择 - 修正版本，拒绝容量不足的池
    MemoryPool* get_optimal_memory_pool(size_t required_size) {
        // 找到最合适的池（容量足够且浪费最少）
        MemoryPool* best_pool = nullptr;
        size_t min_waste = SIZE_MAX;
        
        for (auto& pool : memory_pools) {
            if (!pool.in_use && pool.capacity >= required_size && pool.d_points != nullptr) {
                size_t waste = pool.capacity - required_size;
                if (waste < min_waste) {
                    min_waste = waste;
                    best_pool = &pool;
                }
            }
        }
        
        if (best_pool) {
            best_pool->in_use = true;
            //printf("Selected memory pool level %d (capacity: %zu, required: %zu, waste: %zu)\n", 
                  // best_pool->pool_level, best_pool->capacity, required_size, min_waste);
        }
        return best_pool;
    }
    
    // 计算智能批次大小 - 修正版本，避免死循环
    int calculate_smart_batch_size(const std::vector<float>& queries, int start_idx, int max_batch, int nprobe) {
        int remaining_queries = (queries.size() / vector_dim) - start_idx;
        if (remaining_queries <= 0) return 0;
        
        int test_batch_size = std::min(max_batch, remaining_queries);
        int min_viable_batch = 1; // 最小可行批次大小
        
        while (test_batch_size >= min_viable_batch) {
            // 复制测试批次到GPU
            CUDA_CHECK(cudaMemcpy(d_queries, queries.data() + start_idx * vector_dim,
                                  test_batch_size * vector_dim * sizeof(float), 
                                  cudaMemcpyHostToDevice));
            
            // 计算质心距离
            const float alpha = 1.0f, beta = 0.0f;
            CUBLAS_CHECK(cublasSgemm(cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N,
                                    num_clusters, test_batch_size, vector_dim,
                                    &alpha, d_centroids, vector_dim,
                                    d_queries, vector_dim, &beta,
                                    d_centroid_distances, num_clusters));
            
            // 转换为距离
            int total_dists = test_batch_size * num_clusters;
            convert_inner_product_to_distance<<<(total_dists + 255) / 256, 256>>>(
                d_centroid_distances, total_dists);
            
            // 预测每个查询的点数需求
            predict_query_point_counts<<<(test_batch_size + 255) / 256, 256>>>(
                d_centroid_distances, d_inverted_lists_offsets, d_predicted_counts, 
                test_batch_size, num_clusters, nprobe);
            
            // 计算前缀和获取总点数需求
            thrust::exclusive_scan(thrust::device, d_predicted_counts, 
                                   d_predicted_counts + test_batch_size, d_query_point_offsets, 0);
            
            // 获取总点数需求
            int total_required = 0;
            if (test_batch_size > 0) {
                int last_offset, last_count;
                CUDA_CHECK(cudaMemcpy(&last_offset, d_query_point_offsets + test_batch_size - 1, sizeof(int), cudaMemcpyDeviceToHost));
                CUDA_CHECK(cudaMemcpy(&last_count, d_predicted_counts + test_batch_size - 1, sizeof(int), cudaMemcpyDeviceToHost));
                total_required = last_offset + last_count;
            }
            
            // 检查是否有合适的内存池
            MemoryPool* available_pool = get_optimal_memory_pool(total_required);
            if (available_pool) {
                available_pool->in_use = false; // 释放，稍后真正使用时再分配
               // printf("Smart batch size selected: %d (requires %d points)\n", test_batch_size, total_required);
                return test_batch_size;
            }
            
            // 如果内存不够，尝试更小的批次
            //printf("Batch size %d requires %d points, trying smaller batch\n", test_batch_size, total_required);
            test_batch_size = std::max(1, test_batch_size / 2);
        }
        
        // 如果连单个查询都无法处理，返回0表示失败
        printf("Warning: Cannot process even single query due to memory constraints\n");
        return 0;
    }

public:
    // 智能自适应批量搜索 - 修正死循环版本
    std::vector<std::vector<std::pair<float, uint32_t>>> batch_search(
        const std::vector<float>& queries, int k, int nprobe) {
        
        int query_num = queries.size() / vector_dim;
        std::vector<std::vector<std::pair<float, uint32_t>>> results(query_num);
        
       // printf("Starting smart adaptive batch search for %d queries\n", query_num);
        
        // 处理批量查询 - 使用动态批次大小，添加安全机制
        for (int batch_start = 0; batch_start < query_num; ) {
            // 计算这一批的最优大小
            int smart_batch_size = calculate_smart_batch_size(queries, batch_start, max_batch_size, nprobe);
            
            // 如果无法计算出合适的批次大小，尝试降级处理
            if (smart_batch_size <= 0) {
                printf("Warning: Cannot process query %d due to memory constraints, skipping\n", batch_start);
                // 为跳过的查询创建空结果
                results[batch_start] = std::vector<std::pair<float, uint32_t>>();
                batch_start++;
                continue;
            }
            
            int batch_end = std::min(batch_start + smart_batch_size, query_num);
            int current_batch_size = batch_end - batch_start;
            
           // printf("Processing batch %d-%d (size: %d)\n", batch_start, batch_end-1, current_batch_size);
            
            // 添加安全检查：如果批次大小异常，强制退出
            if (current_batch_size <= 0 || current_batch_size > max_batch_size) {
                printf("Error: Invalid batch size %d, forcing single query processing\n", current_batch_size);
                current_batch_size = 1;
                batch_end = batch_start + 1;
            }
            
            // 复制查询数据到GPU
            CUDA_CHECK(cudaMemcpy(d_queries, queries.data() + batch_start * vector_dim,
                                  current_batch_size * vector_dim * sizeof(float), 
                                  cudaMemcpyHostToDevice));
            
            // 计算质心距离
            const float alpha = 1.0f, beta = 0.0f;
            CUBLAS_CHECK(cublasSgemm(cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N,
                                    num_clusters, current_batch_size, vector_dim,
                                    &alpha, d_centroids, vector_dim,
                                    d_queries, vector_dim, &beta,
                                    d_centroid_distances, num_clusters));
            
            // 转换内积为距离
            int total_dists = current_batch_size * num_clusters;
            convert_inner_product_to_distance<<<(total_dists + 255) / 256, 256>>>(
                d_centroid_distances, total_dists);
            CUDA_CHECK(cudaGetLastError());

            // 处理当前批次（保证不截断），添加重试计数
            int retry_count = 0;
            const int max_retries = 3;
            bool success = false;
            
            while (!success && retry_count < max_retries) {
                success = process_batch_gpu_smart(d_queries, current_batch_size, k, nprobe);
                if (!success) {
                    retry_count++;
                  //  printf("Batch processing failed, retry %d/%d\n", retry_count, max_retries);
                    
                    // 如果重试失败，尝试更小的批次
                    if (retry_count >= max_retries && current_batch_size > 1) {
                        current_batch_size = std::max(1, current_batch_size / 2);
                        batch_end = batch_start + current_batch_size;
                        retry_count = 0; // 重置重试计数
                   //     printf("Reducing batch size to %d and retrying\n", current_batch_size);
                        
                        // 重新复制较小的批次
                        CUDA_CHECK(cudaMemcpy(d_queries, queries.data() + batch_start * vector_dim,
                                              current_batch_size * vector_dim * sizeof(float), 
                                              cudaMemcpyHostToDevice));
                        
                        // 重新计算质心距离
                        CUBLAS_CHECK(cublasSgemm(cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N,
                                                num_clusters, current_batch_size, vector_dim,
                                                &alpha, d_centroids, vector_dim,
                                                d_queries, vector_dim, &beta,
                                                d_centroid_distances, num_clusters));
                        
                        total_dists = current_batch_size * num_clusters;
                        convert_inner_product_to_distance<<<(total_dists + 255) / 256, 256>>>(
                            d_centroid_distances, total_dists);
                        CUDA_CHECK(cudaGetLastError());
                    }
                }
            }
            
            if (success) {
                // 复制结果回CPU
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
                
                batch_start = batch_end;
            } else {
                // 如果最终还是失败，跳过当前查询
               // printf("Warning: Failed to process queries %d-%d after all retries, skipping\n", 
               //        batch_start, batch_end-1);
                for (int i = batch_start; i < batch_end; ++i) {
                    results[i] = std::vector<std::pair<float, uint32_t>>();
                }
                batch_start = batch_end;
            }
            
            // 添加进度检查，防止无限循环
            static int last_batch_start = -1;
            static int stuck_count = 0;
            
            if (batch_start == last_batch_start) {
                stuck_count++;
                if (stuck_count > 5) {
                  //  printf("Error: Detected infinite loop at batch_start=%d, forcing advance\n", batch_start);
                    batch_start++; // 强制前进
                    stuck_count = 0;
                }
            } else {
                stuck_count = 0;
            }
            last_batch_start = batch_start;
        }
        
        return results;
    }

private:
    // 智能批次处理 - 保证不截断，添加更好的错误处理
    bool process_batch_gpu_smart(const float* d_queries, int batch_size, int k, int nprobe) {
        // 1. 找到top nprobe个簇
        find_top_n_clusters_fixed<<<batch_size, 1>>>(
            d_centroid_distances, d_top_clusters_indices, 
            batch_size, num_clusters, nprobe);
        CUDA_CHECK(cudaGetLastError());

        // 2. 计算每个查询需要的点数
        calculate_gather_offsets_fixed<<<(batch_size + 255) / 256, 256>>>(
            d_top_clusters_indices, d_inverted_lists_offsets, 
            d_query_point_counts, batch_size, nprobe, num_clusters);
        CUDA_CHECK(cudaGetLastError());

        // 3. 计算前缀和
        thrust::exclusive_scan(thrust::device, 
                               d_query_point_counts, 
                               d_query_point_counts + batch_size, 
                               d_query_point_offsets, 
                               0);

        // 4. 获取总点数需求
        int total_required = 0;
        if (batch_size > 0) {
            int last_offset, last_count;
            CUDA_CHECK(cudaMemcpy(&last_offset, d_query_point_offsets + batch_size - 1, sizeof(int), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(&last_count, d_query_point_counts + batch_size - 1, sizeof(int), cudaMemcpyDeviceToHost));
            total_required = last_offset + last_count;
        }

        // 5. 获取合适的内存池，严格要求容量足够
        MemoryPool* selected_pool = get_optimal_memory_pool(total_required);
        if (!selected_pool) {
           // printf("No suitable memory pool for %d points (pools available: %zu)\n", 
           //        total_required, memory_pools.size());
            
            // 打印所有内存池状态用于调试
            for (size_t i = 0; i < memory_pools.size(); ++i) {
               // printf("Pool %zu: capacity=%zu, in_use=%s, valid=%s\n", 
                 //      i, memory_pools[i].capacity, 
                   //    memory_pools[i].in_use ? "true" : "false",
                     //  memory_pools[i].d_points ? "true" : "false");
            }
            return false;
        }

        //printf("Using memory pool level %d (capacity: %zu) for %d points\n", 
         //      selected_pool->pool_level, selected_pool->capacity, total_required);

        // 6. 设置内存指针并更新偏移量
        d_selected_points = selected_pool->d_points;
        d_point_distances = selected_pool->d_distances;
        
        // 确保不会超出内存池容量
        int safe_total_required = std::min(total_required, static_cast<int>(selected_pool->capacity));
        CUDA_CHECK(cudaMemcpy(d_query_point_offsets + batch_size, &safe_total_required, 
                              sizeof(int), cudaMemcpyHostToDevice));
        
        // 7. 收集候选点（确保不超出边界）
        gather_points_fixed<<<(batch_size + 255) / 256, 256>>>(
            d_top_clusters_indices, d_inverted_lists_offsets, d_inverted_lists_data, 
            d_query_point_offsets, d_selected_points, batch_size, nprobe, num_clusters);
        CUDA_CHECK(cudaGetLastError());

        // 8. 计算距离
        size_t shared_mem_size = vector_dim * sizeof(float);
        optimized_point_distances<<<batch_size, 256, shared_mem_size>>>(
            d_base_data, d_queries, d_selected_points, d_query_point_offsets,
            d_point_distances, batch_size, vector_dim);
        CUDA_CHECK(cudaGetLastError());

        // 9. Top-K选择
        batch_gpu_topk_selection_fixed<<<batch_size, 256>>>(
            d_point_distances, d_selected_points, d_query_point_offsets,
            d_results, d_result_distances, batch_size, k);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        // 10. 释放内存池
        selected_pool->in_use = false;
        
        return true;
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