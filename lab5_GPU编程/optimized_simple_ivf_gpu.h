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
#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        printf("CUDA error at %s:%d - %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(1); \
    } \
} while(0)

#define CUBLAS_CHECK(call) do { \
    cublasStatus_t stat = call; \
    if (stat != CUBLAS_STATUS_SUCCESS) { \
        printf("cuBLAS error at %s:%d\n", __FILE__, __LINE__); \
        exit(1); \
    } \
} while(0)

// 转换内积为距离的核函数
__global__ void convert_inner_product_to_distance(float* distances, int total_elements) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total_elements) {
        distances[idx] = 1.0f - distances[idx];
    }
}

// 修复后的GPU Top-K选择核函数 - 使用单线程避免竞争条件
__global__ void batch_gpu_topk_selection_fixed(const float* __restrict__ distances,
                                               const uint32_t* __restrict__ point_indices,
                                               const int* __restrict__ query_offsets,
                                               uint32_t* __restrict__ results,
                                               float* __restrict__ result_distances,
                                               int batch_size, int k) {
    int query_idx = blockIdx.x;
    if (query_idx >= batch_size) return;

    // 边界检查
    if (query_idx >= batch_size) return;
    
    int start_offset = query_offsets[query_idx];
    int end_offset = query_offsets[query_idx + 1];
    int num_points = end_offset - start_offset;

    const float* query_dists = distances + start_offset;
    const uint32_t* query_indices = point_indices + start_offset;
    
    uint32_t* query_results = results + query_idx * k;
    float* query_result_dists = result_distances + query_idx * k;

    // 只用第一个线程处理Top-K选择，避免竞争条件
    if (threadIdx.x == 0) {
        // 初始化结果为最大值
        for (int i = 0; i < k; ++i) {
            query_result_dists[i] = FLT_MAX;
            query_results[i] = 0;
        }
        
        // 边界检查
        if (num_points <= 0) return;
        
        // 对每个候选点，检查是否应该插入到Top-K中
        for (int i = 0; i < num_points; ++i) {
            float dist = query_dists[i];
            uint32_t p_idx = query_indices[i];
            
            // 检查是否应该插入到Top-K列表
            if (dist < query_result_dists[k - 1]) {
                // 找到插入位置
                int insert_pos = k - 1;
                while (insert_pos > 0 && dist < query_result_dists[insert_pos - 1]) {
                    insert_pos--;
                }
                
                // 向后移动元素为新元素腾出空间
                for (int j = k - 1; j > insert_pos; --j) {
                    query_result_dists[j] = query_result_dists[j - 1];
                    query_results[j] = query_results[j - 1];
                }
                
                // 插入新元素
                query_result_dists[insert_pos] = dist;
                query_results[insert_pos] = p_idx;
            }
        }
    }
}

// 优化的距离计算核函数：直接在GPU上计算到选定点的距离
__global__ void optimized_point_distances(const float* __restrict__ base_data,
                                         const float* __restrict__ queries,
                                         const uint32_t* __restrict__ selected_points,
                                         const int* __restrict__ query_offsets,
                                         float* __restrict__ distances,
                                         int batch_size, int dim) {
    extern __shared__ float shared_mem[];
    float* shared_query = shared_mem;

    int query_idx = blockIdx.x;
    if (query_idx >= batch_size) return;

    // 协作加载查询向量到共享内存
    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        if (i < dim) {
            shared_query[i] = queries[query_idx * dim + i];
        }
    }
    __syncthreads();

    int start_offset = query_offsets[query_idx];
    int end_offset = query_offsets[query_idx + 1];
    int num_points_for_query = end_offset - start_offset;

    for (int i = threadIdx.x; i < num_points_for_query; i += blockDim.x) {
        if (start_offset + i < end_offset) {
            uint32_t point_id = selected_points[start_offset + i];
            
            float dot_product = 0.0f;
            #pragma unroll 4
            for (int d = 0; d < dim; ++d) {
                dot_product += base_data[point_id * dim + d] * shared_query[d];
            }
            distances[start_offset + i] = 1.0f - dot_product;
        }
    }
}

// 在GPU上为每个查询找到Top-N个簇 - 修复版本
__global__ void find_top_n_clusters_fixed(const float* __restrict__ centroid_distances,
                                          int* __restrict__ top_clusters_indices,
                                          int query_num, int centroid_num, int nprobe) {
    int query_idx = blockIdx.x;
    if (query_idx >= query_num) return;

    const float* query_dists = centroid_distances + query_idx * centroid_num;
    int* query_top_clusters = top_clusters_indices + query_idx * nprobe;

    // 初始化为-1
    for (int i = 0; i < nprobe; ++i) {
        query_top_clusters[i] = -1;
    }

    // 使用简单的选择排序来找到Top-Nprobe
    for (int i = 0; i < nprobe && i < centroid_num; ++i) {
        float min_dist = FLT_MAX;
        int min_idx = -1;
        
        for (int j = 0; j < centroid_num; ++j) {
            // 检查是否已经被选择
            bool already_selected = false;
            for (int prev = 0; prev < i; ++prev) {
                if (query_top_clusters[prev] == j) {
                    already_selected = true;
                    break;
                }
            }
            
            if (!already_selected && query_dists[j] < min_dist) {
                min_dist = query_dists[j];
                min_idx = j;
            }
        }
        
        if (min_idx >= 0) {
            query_top_clusters[i] = min_idx;
        }
    }
}

// Kernel to calculate offsets for gathered points - 修复版本
__global__ void calculate_gather_offsets_fixed(const int* __restrict__ top_clusters_indices,
                                               const int* __restrict__ inv_lists_offsets,
                                               int* __restrict__ query_point_counts,
                                               int batch_size, int nprobe, int num_clusters) {
    int query_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (query_idx >= batch_size) return;

    int count = 0;
    for (int i = 0; i < nprobe; ++i) {
        int cluster_idx = top_clusters_indices[query_idx * nprobe + i];
        if (cluster_idx >= 0 && cluster_idx < num_clusters) {
            int list_size = inv_lists_offsets[cluster_idx + 1] - inv_lists_offsets[cluster_idx];
            count += list_size;
        }
    }
    query_point_counts[query_idx] = count;
}

// Kernel to gather points from inverted lists - 修复版本
__global__ void gather_points_fixed(const int* __restrict__ top_clusters_indices,
                                    const int* __restrict__ inv_lists_offsets,
                                    const uint32_t* __restrict__ inv_lists_data,
                                    const int* __restrict__ query_gather_offsets,
                                    uint32_t* __restrict__ gathered_points,
                                    int batch_size, int nprobe, int num_clusters) {
    int query_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (query_idx >= batch_size) return;

    int write_offset = query_gather_offsets[query_idx];
    
    for (int i = 0; i < nprobe; ++i) {
        int cluster_idx = top_clusters_indices[query_idx * nprobe + i];
        if (cluster_idx >= 0 && cluster_idx < num_clusters) {
            int list_start = inv_lists_offsets[cluster_idx];
            int list_end = inv_lists_offsets[cluster_idx + 1];
            
            for (int j = list_start; j < list_end; ++j) {
                gathered_points[write_offset++] = inv_lists_data[j];
            }
        }
    }
}

// 优化的GPU IVF实现
class OptimizedSimpleIVFGPU {
private:
    // 预分配的GPU内存
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
    
    // cuBLAS句柄
    cublasHandle_t cublas_handle;
    
    // 参数
    int num_base_vectors;
    int vector_dim;
    int num_clusters;
    int max_batch_size;
    int max_total_selected_points_in_batch;
    
    // CPU端数据
    std::vector<float> h_centroids;
    std::vector<std::vector<uint32_t>> h_inverted_lists;

public:
    OptimizedSimpleIVFGPU(float* base_data, int n_base, int dim, int n_clusters, int batch_size = 64)
        : num_base_vectors(n_base), vector_dim(dim), num_clusters(n_clusters), 
          max_batch_size(batch_size) {
        
        // 更保守的内存分配策略
        max_total_selected_points_in_batch = max_batch_size * std::max(1, n_base / n_clusters) * 4;
        if (max_total_selected_points_in_batch == 0) max_total_selected_points_in_batch = max_batch_size * 100;

        // 初始化cuBLAS
        CUBLAS_CHECK(cublasCreate(&cublas_handle));
        
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
        
        printf("OptimizedSimpleIVFGPU initialized: %d vectors, %d dim, %d clusters\n",
               n_base, dim, n_clusters);
    }
    
    ~OptimizedSimpleIVFGPU() {
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
        
        // Pre-allocate a large buffer for selected points and their distances
        CUDA_CHECK(cudaMalloc(&d_selected_points, max_total_selected_points_in_batch * sizeof(uint32_t)));
        CUDA_CHECK(cudaMalloc(&d_point_distances, max_total_selected_points_in_batch * sizeof(float)));

        CUDA_CHECK(cudaMalloc(&d_results, max_batch_size * 100 * sizeof(uint32_t))); // 假设k<=100
        CUDA_CHECK(cudaMalloc(&d_result_distances, max_batch_size * 100 * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_top_clusters_indices, max_batch_size * 32 * sizeof(int))); // 假设nprobe<=32
        CUDA_CHECK(cudaMalloc(&d_query_point_counts, max_batch_size * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_query_point_offsets, (max_batch_size + 1) * sizeof(int)));
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
        // 使用更简单但更稳定的k-means算法
        h_centroids.resize(num_clusters * vector_dim);
        h_inverted_lists.resize(num_clusters);
        
        // 随机初始化质心
        srand(42);
        for (int c = 0; c < num_clusters; ++c) {
            int random_idx = rand() % num_base_vectors;
            std::copy(base_data + random_idx * vector_dim,
                     base_data + (random_idx + 1) * vector_dim,
                     h_centroids.begin() + c * vector_dim);
        }
        
        // 迭代优化质心
        for (int iter = 0; iter < 5; ++iter) {
            std::vector<int> assignments(num_base_vectors);
            
            // 分配点到最近质心
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
    // 高性能批量搜索
    std::vector<std::vector<std::pair<float, uint32_t>>> batch_search(
        const std::vector<float>& queries, int k, int nprobe) {
        
        int query_num = queries.size() / vector_dim;
        std::vector<std::vector<std::pair<float, uint32_t>>> results(query_num);
        
        // 处理批量查询
        for (int batch_start = 0; batch_start < query_num; batch_start += max_batch_size) {
            int batch_end = std::min(batch_start + max_batch_size, query_num);
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
            CUDA_CHECK(cudaDeviceSynchronize());

            // 在GPU上处理整个批次
            process_batch_gpu(d_queries, current_batch_size, k, nprobe);
            
            // 复制最终结果回CPU
            std::vector<uint32_t> h_results(current_batch_size * k);
            std::vector<float> h_result_distances(current_batch_size * k);
            CUDA_CHECK(cudaMemcpy(h_results.data(), d_results, 
                                 current_batch_size * k * sizeof(uint32_t), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(h_result_distances.data(), d_result_distances, 
                                 current_batch_size * k * sizeof(float), cudaMemcpyDeviceToHost));

            for (int i = 0; i < current_batch_size; ++i) {
                results[batch_start + i].reserve(k);
                for (int j = 0; j < k; ++j) {
                    if (h_result_distances[i * k + j] != FLT_MAX) {
                        results[batch_start + i].push_back({h_result_distances[i * k + j], h_results[i * k + j]});
                    }
                }
            }
        }
        
        return results;
    }

private:
    void process_batch_gpu(const float* d_queries, int batch_size, int k, int nprobe) {
        // 1. 在GPU上找到每个查询的top nprobe个簇
        find_top_n_clusters_fixed<<<batch_size, 1>>>(
            d_centroid_distances, d_top_clusters_indices, 
            batch_size, num_clusters, nprobe);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        // 2. 计算每个查询需要收集的点数
        calculate_gather_offsets_fixed<<<(batch_size + 255) / 256, 256>>>(
            d_top_clusters_indices, d_inverted_lists_offsets, 
            d_query_point_counts, batch_size, nprobe, num_clusters);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        // 3. 在GPU上使用Thrust执行前缀和来获取偏移量
        thrust::exclusive_scan(thrust::device, 
                               d_query_point_counts, 
                               d_query_point_counts + batch_size, 
                               d_query_point_offsets, 
                               0);

        // 4. 获取选择的总点数以进行健全性检查
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

        if (total_selected_points > max_total_selected_points_in_batch) {
            printf("Warning: Truncating points. Required: %d, Allocated: %d\n", 
                   total_selected_points, max_total_selected_points_in_batch);
            return;
        }
        
        // 5. 在GPU上收集所有候选项
        gather_points_fixed<<<(batch_size + 255) / 256, 256>>>(
            d_top_clusters_indices, d_inverted_lists_offsets, d_inverted_lists_data, 
            d_query_point_offsets, d_selected_points, batch_size, nprobe, num_clusters);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        // 6. 计算到这些点的距离
        size_t shared_mem_size = vector_dim * sizeof(float);
        optimized_point_distances<<<batch_size, 256, shared_mem_size>>>(
            d_base_data, d_queries, d_selected_points, d_query_point_offsets,
            d_point_distances, batch_size, vector_dim);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        // 7. GPU Top-K 选择 - 使用修复版本
        batch_gpu_topk_selection_fixed<<<batch_size, 256>>>(
            d_point_distances, d_selected_points, d_query_point_offsets,
            d_results, d_result_distances, batch_size, k);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
    }

public:
    // 单个查询搜索的简单实现
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