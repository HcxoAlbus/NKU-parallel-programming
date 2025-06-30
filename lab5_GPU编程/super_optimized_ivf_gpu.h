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
#include <cub/cub.cuh>

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

// 高效的Top-K选择核函数 - 使用Warp-level操作优化
__global__ void optimized_warp_topk_selection(const float* __restrict__ distances,
                                              const uint32_t* __restrict__ point_indices,
                                              const int* __restrict__ query_offsets,
                                              uint32_t* __restrict__ results,
                                              float* __restrict__ result_distances,
                                              int batch_size, int k) {
    int query_idx = blockIdx.x;
    if (query_idx >= batch_size) return;

    int start_offset = query_offsets[query_idx];
    int end_offset = query_offsets[query_idx + 1];
    int num_points = end_offset - start_offset;

    const float* query_dists = distances + start_offset;
    const uint32_t* query_indices = point_indices + start_offset;
    
    uint32_t* query_results = results + query_idx * k;
    float* query_result_dists = result_distances + query_idx * k;

    // 使用每个warp处理一个查询，提高并行度
    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;
    
    if (warp_id == 0) {  // 只用第一个warp处理这个查询
        // 使用寄存器数组存储top-k结果（适用于小k值）
        float topk_dists[10];  // 假设k <= 10
        uint32_t topk_indices[10];
        
        // 初始化
        for (int i = 0; i < k && i < 10; ++i) {
            topk_dists[i] = FLT_MAX;
            topk_indices[i] = 0;
        }
        
        // 并行处理点，每个线程处理一部分
        for (int i = lane_id; i < num_points; i += 32) {
            float dist = query_dists[i];
            uint32_t p_idx = query_indices[i];
            
            // 插入到top-k中
            if (dist < topk_dists[k-1]) {
                int insert_pos = k - 1;
                while (insert_pos > 0 && dist < topk_dists[insert_pos - 1]) {
                    insert_pos--;
                }
                
                // 移动元素
                for (int j = k - 1; j > insert_pos; --j) {
                    topk_dists[j] = topk_dists[j - 1];
                    topk_indices[j] = topk_indices[j - 1];
                }
                
                topk_dists[insert_pos] = dist;
                topk_indices[insert_pos] = p_idx;
            }
        }
        
        // 只有第一个线程写回结果
        if (lane_id == 0) {
            for (int i = 0; i < k && i < 10; ++i) {
                query_result_dists[i] = topk_dists[i];
                query_results[i] = topk_indices[i];
            }
        }
    }
}

// 内存融合的距离计算核函数
__global__ void fused_distance_and_topk(const float* __restrict__ base_data,
                                        const float* __restrict__ queries,
                                        const uint32_t* __restrict__ selected_points,
                                        const int* __restrict__ query_offsets,
                                        uint32_t* __restrict__ results,
                                        float* __restrict__ result_distances,
                                        int batch_size, int dim, int k) {
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

    // 在同一个核函数中进行距离计算和Top-K选择，减少内存访问
    if (threadIdx.x == 0 && num_points_for_query > 0) {
        const uint32_t* query_indices = selected_points + start_offset;
        uint32_t* query_results = results + query_idx * k;
        float* query_result_dists = result_distances + query_idx * k;
        
        // 使用堆来维护Top-K（更高效的方法）
        for (int i = 0; i < k && i < num_points_for_query; ++i) {
            uint32_t point_id = query_indices[i];
            
            float dot_product = 0.0f;
            #pragma unroll 4
            for (int d = 0; d < dim; ++d) {
                dot_product += base_data[point_id * dim + d] * shared_query[d];
            }
            float distance = 1.0f - dot_product;
            
            query_result_dists[i] = distance;
            query_results[i] = point_id;
        }
        
        // 对于剩余的点，维护最小堆
        for (int i = k; i < num_points_for_query; ++i) {
            uint32_t point_id = query_indices[i];
            
            float dot_product = 0.0f;
            #pragma unroll 4
            for (int d = 0; d < dim; ++d) {
                dot_product += base_data[point_id * dim + d] * shared_query[d];
            }
            float distance = 1.0f - dot_product;
            
            // 找到最大距离的位置
            int max_idx = 0;
            float max_dist = query_result_dists[0];
            for (int j = 1; j < k; ++j) {
                if (query_result_dists[j] > max_dist) {
                    max_dist = query_result_dists[j];
                    max_idx = j;
                }
            }
            
            // 如果当前距离更小，替换
            if (distance < max_dist) {
                query_result_dists[max_idx] = distance;
                query_results[max_idx] = point_id;
            }
        }
    }
}

// 超级优化的GPU IVF实现
class SuperOptimizedIVFGPU {
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
    
    // 多流处理
    cudaStream_t* streams;
    int num_streams;
    
    // cuBLAS句柄
    cublasHandle_t cublas_handle;
    
    // 临时缓冲区用于CUB操作
    void* d_temp_storage;
    size_t temp_storage_bytes;
    
    // 参数
    int num_base_vectors;
    int vector_dim;
    int num_clusters;
    int max_batch_size;
    size_t max_total_selected_points_in_batch;
    
    // CPU端数据
    std::vector<float> h_centroids;
    std::vector<std::vector<uint32_t>> h_inverted_lists;

public:
    SuperOptimizedIVFGPU(float* base_data, int n_base, int dim, int n_clusters, int batch_size = 128)
        : num_base_vectors(n_base), vector_dim(dim), num_clusters(n_clusters), 
          max_batch_size(batch_size), num_streams(4) {
        
        // 动态调整内存分配策略
        size_t avg_points_per_cluster = n_base / n_clusters;
        size_t max_nprobe = 32;  // 最大nprobe值
        max_total_selected_points_in_batch = max_batch_size * avg_points_per_cluster * max_nprobe * 2;
        
        // 获取GPU内存信息
        size_t free_mem, total_mem;
        CUDA_CHECK(cudaMemGetInfo(&free_mem, &total_mem));
        
        // 根据可用内存调整分配策略
        size_t required_mem = max_total_selected_points_in_batch * (sizeof(uint32_t) + sizeof(float));
        if (required_mem > free_mem * 0.8) {
            max_total_selected_points_in_batch = (free_mem * 0.6) / (sizeof(uint32_t) + sizeof(float));
            printf("Adjusted memory allocation to fit GPU memory: %zu points\n", max_total_selected_points_in_batch);
        }

        // 初始化cuBLAS
        CUBLAS_CHECK(cublasCreate(&cublas_handle));
        CUBLAS_CHECK(cublasSetMathMode(cublas_handle, CUBLAS_TENSOR_OP_MATH));
        
        // 创建多个CUDA流
        streams = new cudaStream_t[num_streams];
        for (int i = 0; i < num_streams; ++i) {
            CUDA_CHECK(cudaStreamCreate(&streams[i]));
        }
        
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
        
        printf("SuperOptimizedIVFGPU initialized: %d vectors, %d dim, %d clusters, %zu max_points\n",
               n_base, dim, n_clusters, max_total_selected_points_in_batch);
    }
    
    ~SuperOptimizedIVFGPU() {
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
        if (d_temp_storage) cudaFree(d_temp_storage);
        
        // 销毁流
        for (int i = 0; i < num_streams; ++i) {
            cudaStreamDestroy(streams[i]);
        }
        delete[] streams;
        
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
        
        // 动态分配的大缓冲区
        CUDA_CHECK(cudaMalloc(&d_selected_points, max_total_selected_points_in_batch * sizeof(uint32_t)));
        CUDA_CHECK(cudaMalloc(&d_point_distances, max_total_selected_points_in_batch * sizeof(float)));

        CUDA_CHECK(cudaMalloc(&d_results, max_batch_size * 100 * sizeof(uint32_t)));
        CUDA_CHECK(cudaMalloc(&d_result_distances, max_batch_size * 100 * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_top_clusters_indices, max_batch_size * 64 * sizeof(int))); // 支持更大的nprobe
        CUDA_CHECK(cudaMalloc(&d_query_point_counts, max_batch_size * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_query_point_offsets, (max_batch_size + 1) * sizeof(int)));
        
        // 为CUB操作分配临时存储
        temp_storage_bytes = 0;
        cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, 
                                      d_query_point_counts, d_query_point_offsets, max_batch_size);
        CUDA_CHECK(cudaMalloc(&d_temp_storage, temp_storage_bytes));
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
        // 使用更好的k-means++算法
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
    // 超高性能批量搜索 - 使用多流并行
    std::vector<std::vector<std::pair<float, uint32_t>>> batch_search(
        const std::vector<float>& queries, int k, int nprobe) {
        
        int query_num = queries.size() / vector_dim;
        std::vector<std::vector<std::pair<float, uint32_t>>> results(query_num);
        
        // 处理批量查询 - 使用多流并行
        for (int batch_start = 0; batch_start < query_num; batch_start += max_batch_size) {
            int batch_end = std::min(batch_start + max_batch_size, query_num);
            int current_batch_size = batch_end - batch_start;
            
            // 选择当前流
            int stream_idx = (batch_start / max_batch_size) % num_streams;
            cudaStream_t current_stream = streams[stream_idx];
            
            // 异步复制查询数据到GPU
            CUDA_CHECK(cudaMemcpyAsync(d_queries, queries.data() + batch_start * vector_dim,
                                      current_batch_size * vector_dim * sizeof(float), 
                                      cudaMemcpyHostToDevice, current_stream));
            
            // 使用cuBLAS计算查询到质心的距离
            const float alpha = 1.0f, beta = 0.0f;
            CUBLAS_CHECK(cublasSetStream(cublas_handle, current_stream));
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
            
            convert_inner_product_to_distance<<<blocks_dist, threads_per_block_dist, 0, current_stream>>>(
                d_centroid_distances, total_dists);
            CUDA_CHECK(cudaGetLastError());

            // 在GPU上处理整个批次
            process_batch_gpu_stream(d_queries, current_batch_size, k, nprobe, current_stream);
            
            // 异步复制最终结果回CPU
            std::vector<uint32_t> h_results(current_batch_size * k);
            std::vector<float> h_result_distances(current_batch_size * k);
            CUDA_CHECK(cudaMemcpyAsync(h_results.data(), d_results, 
                                      current_batch_size * k * sizeof(uint32_t), 
                                      cudaMemcpyDeviceToHost, current_stream));
            CUDA_CHECK(cudaMemcpyAsync(h_result_distances.data(), d_result_distances, 
                                      current_batch_size * k * sizeof(float), 
                                      cudaMemcpyDeviceToHost, current_stream));
            
            // 等待当前流完成
            CUDA_CHECK(cudaStreamSynchronize(current_stream));

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
    void process_batch_gpu_stream(const float* d_queries, int batch_size, int k, int nprobe, cudaStream_t stream) {
        // 1. 在GPU上找到每个查询的top nprobe个簇
        find_top_n_clusters_fixed<<<batch_size, 1, 0, stream>>>(
            d_centroid_distances, d_top_clusters_indices, 
            batch_size, num_clusters, nprobe);
        CUDA_CHECK(cudaGetLastError());

        // 2. 计算每个查询需要收集的点数
        calculate_gather_offsets_fixed<<<(batch_size + 255) / 256, 256, 0, stream>>>(
            d_top_clusters_indices, d_inverted_lists_offsets, 
            d_query_point_counts, batch_size, nprobe, num_clusters);
        CUDA_CHECK(cudaGetLastError());

        // 3. 使用CUB进行高效的前缀和计算
        cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, 
                                      d_query_point_counts, d_query_point_offsets, 
                                      batch_size, stream);

        // 4. 检查总选择点数
        int total_selected_points = 0;
        if (batch_size > 0) {
            int last_offset, last_count;
            CUDA_CHECK(cudaMemcpyAsync(&last_offset, d_query_point_offsets + batch_size - 1, 
                                      sizeof(int), cudaMemcpyDeviceToHost, stream));
            CUDA_CHECK(cudaMemcpyAsync(&last_count, d_query_point_counts + batch_size - 1, 
                                      sizeof(int), cudaMemcpyDeviceToHost, stream));
            CUDA_CHECK(cudaStreamSynchronize(stream));
            
            total_selected_points = last_offset + last_count;
            CUDA_CHECK(cudaMemcpyAsync(d_query_point_offsets + batch_size, &total_selected_points, 
                                      sizeof(int), cudaMemcpyHostToDevice, stream));
        }

        if (total_selected_points > max_total_selected_points_in_batch) {
            printf("Warning: Truncating points. Required: %d, Allocated: %zu\n", 
                   total_selected_points, max_total_selected_points_in_batch);
            // 仍然继续处理，但可能影响召回率
        }
        
        // 5. 在GPU上收集所有候选项
        gather_points_fixed<<<(batch_size + 255) / 256, 256, 0, stream>>>(
            d_top_clusters_indices, d_inverted_lists_offsets, d_inverted_lists_data, 
            d_query_point_offsets, d_selected_points, batch_size, nprobe, num_clusters);
        CUDA_CHECK(cudaGetLastError());

        // 6. 融合的距离计算和Top-K选择
        size_t shared_mem_size = vector_dim * sizeof(float);
        fused_distance_and_topk<<<batch_size, 256, shared_mem_size, stream>>>(
            d_base_data, d_queries, d_selected_points, d_query_point_offsets,
            d_results, d_result_distances,
            batch_size, vector_dim, k);
        CUDA_CHECK(cudaGetLastError());
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