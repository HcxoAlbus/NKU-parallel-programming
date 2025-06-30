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
#include <thrust/device_vector.h>
#include <thrust/sort.h>
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

// 优化的CUDA核函数：使用共享内存的距离计算
__global__ void optimized_distance_kernel(const float* __restrict__ base_data, 
                                         const float* __restrict__ queries,
                                         float* __restrict__ distances,
                                         const uint32_t* __restrict__ selected_points,
                                         int num_selected, int query_num, int dim) {
    __shared__ float shared_query[256]; // 共享内存存储查询向量
    
    int tid = threadIdx.x;
    int query_idx = blockIdx.y;
    int point_idx = blockIdx.x * blockDim.x + tid;
    
    if (query_idx >= query_num) return;
    
    // 协作加载查询向量到共享内存
    for (int i = tid; i < dim; i += blockDim.x) {
        if (i < dim) {
            shared_query[i] = queries[query_idx * dim + i];
        }
    }
    __syncthreads();
    
    if (point_idx < num_selected) {
        uint32_t actual_point_idx = selected_points[point_idx];
        
        // 计算内积
        float dot_product = 0.0f;
        #pragma unroll 4
        for (int d = 0; d < dim; ++d) {
            dot_product += base_data[actual_point_idx * dim + d] * shared_query[d];
        }
        
        distances[query_idx * num_selected + point_idx] = 1.0f - dot_product;
    }
}

// GPU端Top-K选择核函数（使用堆排序）
__global__ void gpu_topk_selection(const float* __restrict__ distances,
                                  const uint32_t* __restrict__ point_indices,
                                  uint32_t* __restrict__ results,
                                  float* __restrict__ result_distances,
                                  int num_points, int query_num, int k) {
    int query_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (query_idx >= query_num) return;
    
    // 每个线程处理一个查询的Top-K选择
    const float* query_dists = distances + query_idx * num_points;
    uint32_t* query_results = results + query_idx * k;
    float* query_result_dists = result_distances + query_idx * k;
    
    // 使用选择排序找Top-K（对于小K值很高效）
    for (int i = 0; i < k && i < num_points; ++i) {
        float min_dist = FLT_MAX;
        int min_idx = -1;
        
        for (int j = 0; j < num_points; ++j) {
            float dist = query_dists[j];
            
            // 检查是否已被选择
            bool already_selected = false;
            for (int prev = 0; prev < i; ++prev) {
                if (query_results[prev] == point_indices[j]) {
                    already_selected = true;
                    break;
                }
            }
            
            if (!already_selected && dist < min_dist) {
                min_dist = dist;
                min_idx = j;
            }
        }
        
        if (min_idx != -1) {
            query_results[i] = point_indices[min_idx];
            query_result_dists[i] = min_dist;
        }
    }
}

// 优化的GPU端Top-K选择（使用Thrust库）
__global__ void prepare_distance_index_pairs(const float* distances,
                                            const uint32_t* point_indices,
                                            float* out_distances,
                                            uint32_t* out_indices,
                                            int num_points, int query_idx) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid < num_points) {
        out_distances[tid] = distances[query_idx * num_points + tid];
        out_indices[tid] = point_indices[tid];
    }
}

// cuBLAS矩阵乘法的包装核函数
__global__ void convert_inner_product_to_distance(float* distances, int total_elements) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total_elements) {
        distances[idx] = 1.0f - distances[idx];
    }
}

// 高性能优化的GPU IVF实现
class OptimizedIVFGPU {
private:
    // GPU内存指针
    float* d_base_data;
    float* d_centroids;
    uint32_t* d_inverted_lists;     // 扁平化的倒排列表
    int* d_list_offsets;            // 每个簇的起始偏移
    int* d_list_sizes;              // 每个簇的大小
    
    // 临时GPU内存（预分配以避免频繁分配）
    float* d_queries;
    float* d_centroid_distances;
    float* d_point_distances;
    uint32_t* d_selected_points;
    uint32_t* d_results;
    float* d_result_distances;
    float* d_temp_distances;
    uint32_t* d_temp_indices;
    
    // cuBLAS句柄
    cublasHandle_t cublas_handle;
    
    // 参数
    int num_base_vectors;
    int vector_dim;
    int num_clusters;
    int max_batch_size;
    int max_points_per_query;
    
    // CPU端数据
    std::vector<float> h_centroids;
    std::vector<std::vector<uint32_t>> h_inverted_lists;
    std::vector<int> h_list_offsets;
    std::vector<int> h_list_sizes;

public:
    OptimizedIVFGPU(float* base_data, int n_base, int dim, int n_clusters, int batch_size = 256)
        : num_base_vectors(n_base), vector_dim(dim), num_clusters(n_clusters), 
          max_batch_size(batch_size), max_points_per_query(n_base / n_clusters * 8) {
        
        // 初始化cuBLAS
        CUBLAS_CHECK(cublasCreate(&cublas_handle));
        CUBLAS_CHECK(cublasSetMathMode(cublas_handle, CUBLAS_TENSOR_OP_MATH));
        
        // 构建IVF索引
        build_index_cpu(base_data);
        
        // 分配GPU内存
        allocate_gpu_memory();
        
        // 复制数据到GPU
        copy_data_to_gpu(base_data);
        
        printf("OptimizedIVFGPU initialized: %d vectors, %d dim, %d clusters, batch_size=%d\n",
               n_base, dim, n_clusters, batch_size);
    }
    
    ~OptimizedIVFGPU() {
        // 释放GPU内存
        cudaFree(d_base_data);
        cudaFree(d_centroids);
        cudaFree(d_inverted_lists);
        cudaFree(d_list_offsets);
        cudaFree(d_list_sizes);
        cudaFree(d_queries);
        cudaFree(d_centroid_distances);
        cudaFree(d_point_distances);
        cudaFree(d_selected_points);
        cudaFree(d_results);
        cudaFree(d_result_distances);
        cudaFree(d_temp_distances);
        cudaFree(d_temp_indices);
        
        cublasDestroy(cublas_handle);
    }

private:
    void allocate_gpu_memory() {
        // 基本数据
        CUDA_CHECK(cudaMalloc(&d_base_data, num_base_vectors * vector_dim * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_centroids, num_clusters * vector_dim * sizeof(float)));
        
        // 倒排列表
        CUDA_CHECK(cudaMalloc(&d_inverted_lists, num_base_vectors * sizeof(uint32_t)));
        CUDA_CHECK(cudaMalloc(&d_list_offsets, num_clusters * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_list_sizes, num_clusters * sizeof(int)));
        
        // 预分配批处理内存
        CUDA_CHECK(cudaMalloc(&d_queries, max_batch_size * vector_dim * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_centroid_distances, max_batch_size * num_clusters * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_point_distances, max_batch_size * max_points_per_query * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_selected_points, max_points_per_query * sizeof(uint32_t)));
        CUDA_CHECK(cudaMalloc(&d_results, max_batch_size * 100 * sizeof(uint32_t))); // 假设k<=100
        CUDA_CHECK(cudaMalloc(&d_result_distances, max_batch_size * 100 * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_temp_distances, max_points_per_query * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_temp_indices, max_points_per_query * sizeof(uint32_t)));
    }
    
    void build_index_cpu(float* base_data) {
        // 使用更好的k-means算法
        h_centroids.resize(num_clusters * vector_dim);
        h_inverted_lists.resize(num_clusters);
        
        // k-means++初始化
        std::vector<bool> selected(num_base_vectors, false);
        
        // 随机选择第一个质心
        int first_centroid = rand() % num_base_vectors;
        selected[first_centroid] = true;
        std::copy(base_data + first_centroid * vector_dim,
                 base_data + (first_centroid + 1) * vector_dim,
                 h_centroids.begin());
        
        // k-means++选择剩余质心
        for (int c = 1; c < num_clusters; ++c) {
            std::vector<float> distances(num_base_vectors);
            float total_distance = 0.0f;
            
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
        
        // 迭代优化质心
        for (int iter = 0; iter < 15; ++iter) {
            std::vector<int> assignments(num_base_vectors);
            
            // 分配点到最近质心
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
        std::vector<int> assignments(num_base_vectors);
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
        
        // 构建倒排列表
        for (int c = 0; c < num_clusters; ++c) {
            h_inverted_lists[c].clear();
        }
        
        for (int i = 0; i < num_base_vectors; ++i) {
            h_inverted_lists[assignments[i]].push_back(i);
        }
        
        // 准备GPU数据结构
        h_list_offsets.resize(num_clusters);
        h_list_sizes.resize(num_clusters);
        
        int offset = 0;
        for (int c = 0; c < num_clusters; ++c) {
            h_list_offsets[c] = offset;
            h_list_sizes[c] = h_inverted_lists[c].size();
            offset += h_list_sizes[c];
        }
    }
    
    void copy_data_to_gpu(float* base_data) {
        // 复制基准数据和质心
        CUDA_CHECK(cudaMemcpy(d_base_data, base_data, 
                             num_base_vectors * vector_dim * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_centroids, h_centroids.data(),
                             num_clusters * vector_dim * sizeof(float), cudaMemcpyHostToDevice));
        
        // 扁平化并复制倒排列表
        std::vector<uint32_t> flat_lists;
        for (int c = 0; c < num_clusters; ++c) {
            flat_lists.insert(flat_lists.end(), h_inverted_lists[c].begin(), h_inverted_lists[c].end());
        }
        
        CUDA_CHECK(cudaMemcpy(d_inverted_lists, flat_lists.data(),
                             flat_lists.size() * sizeof(uint32_t), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_list_offsets, h_list_offsets.data(),
                             num_clusters * sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_list_sizes, h_list_sizes.data(),
                             num_clusters * sizeof(int), cudaMemcpyHostToDevice));
    }

public:
    // 高性能批量搜索
    std::vector<std::vector<std::pair<float, uint32_t>>> batch_search(
        const std::vector<float>& queries, int k, int nprobe) {
        
        int query_num = queries.size() / vector_dim;
        std::vector<std::vector<std::pair<float, uint32_t>>> results(query_num);
        
        // 使用CUDA流进行异步处理
        cudaStream_t stream;
        CUDA_CHECK(cudaStreamCreate(&stream));
        
        // 处理批量查询
        for (int batch_start = 0; batch_start < query_num; batch_start += max_batch_size) {
            int batch_end = std::min(batch_start + max_batch_size, query_num);
            int current_batch_size = batch_end - batch_start;
            
            // 复制查询数据到GPU
            CUDA_CHECK(cudaMemcpyAsync(d_queries, 
                                      queries.data() + batch_start * vector_dim,
                                      current_batch_size * vector_dim * sizeof(float),
                                      cudaMemcpyHostToDevice, stream));
            
            // 使用cuBLAS计算查询到质心的距离
            const float alpha = 1.0f, beta = 0.0f;
            CUBLAS_CHECK(cublasSgemmStridedBatched(cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N,
                                                  num_clusters, current_batch_size, vector_dim,
                                                  &alpha,
                                                  d_centroids, vector_dim, 0,
                                                  d_queries, vector_dim, vector_dim,
                                                  &beta,
                                                  d_centroid_distances, num_clusters, num_clusters,
                                                  1));
            
            // 转换内积为距离
            int total_centroid_dists = current_batch_size * num_clusters;
            dim3 convert_block(256);
            dim3 convert_grid((total_centroid_dists + convert_block.x - 1) / convert_block.x);
            convert_inner_product_to_distance<<<convert_grid, convert_block, 0, stream>>>(
                d_centroid_distances, total_centroid_dists);
            
            // 处理每个查询的搜索
            auto batch_results = process_batch_gpu(current_batch_size, k, nprobe, stream);
            
            // 复制结果
            for (int i = 0; i < current_batch_size; ++i) {
                results[batch_start + i] = batch_results[i];
            }
        }
        
        CUDA_CHECK(cudaStreamDestroy(stream));
        return results;
    }

private:
    std::vector<std::vector<std::pair<float, uint32_t>>> process_batch_gpu(
        int batch_size, int k, int nprobe, cudaStream_t stream) {
        
        std::vector<std::vector<std::pair<float, uint32_t>>> results(batch_size);
        
        // 复制质心距离回CPU以选择最近的簇
        std::vector<float> h_centroid_distances(batch_size * num_clusters);
        CUDA_CHECK(cudaMemcpyAsync(h_centroid_distances.data(), d_centroid_distances,
                                  batch_size * num_clusters * sizeof(float),
                                  cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        
        // 为每个查询选择要搜索的簇
        for (int q = 0; q < batch_size; ++q) {
            std::vector<std::pair<float, int>> cluster_dists;
            for (int c = 0; c < num_clusters; ++c) {
                cluster_dists.push_back({h_centroid_distances[q * num_clusters + c], c});
            }
            std::sort(cluster_dists.begin(), cluster_dists.end());
            
            // 收集选定簇中的所有点
            std::vector<uint32_t> selected_points;
            for (int i = 0; i < std::min(nprobe, num_clusters); ++i) {
                int cluster_idx = cluster_dists[i].second;
                const auto& point_list = h_inverted_lists[cluster_idx];
                selected_points.insert(selected_points.end(), point_list.begin(), point_list.end());
            }
            
            if (selected_points.empty()) continue;
            
            // 复制选定的点索引到GPU
            CUDA_CHECK(cudaMemcpyAsync(d_selected_points, selected_points.data(),
                                      selected_points.size() * sizeof(uint32_t),
                                      cudaMemcpyHostToDevice, stream));
            
            // 在GPU上计算距离
            dim3 dist_block(256);
            dim3 dist_grid((selected_points.size() + dist_block.x - 1) / dist_block.x, 1);
            optimized_distance_kernel<<<dist_grid, dist_block, 0, stream>>>(
                d_base_data, d_queries + q * vector_dim, 
                d_point_distances + q * max_points_per_query,
                d_selected_points, selected_points.size(), 1, vector_dim);
            
            // 使用Thrust进行GPU端排序
            thrust::device_ptr<float> d_dist_ptr(d_point_distances + q * max_points_per_query);
            thrust::device_ptr<uint32_t> d_idx_ptr(d_selected_points);
            
            // 创建索引序列
            thrust::sequence(thrust::cuda::par.on(stream), d_idx_ptr, d_idx_ptr + selected_points.size());
            
            // 按距离排序
            thrust::sort_by_key(thrust::cuda::par.on(stream), 
                               d_dist_ptr, d_dist_ptr + selected_points.size(),
                               d_idx_ptr);
            
            // 复制Top-K结果回CPU
            int result_k = std::min(k, (int)selected_points.size());
            std::vector<float> top_distances(result_k);
            std::vector<uint32_t> top_indices(result_k);
            
            CUDA_CHECK(cudaMemcpyAsync(top_distances.data(), d_point_distances + q * max_points_per_query,
                                      result_k * sizeof(float), cudaMemcpyDeviceToHost, stream));
            CUDA_CHECK(cudaMemcpyAsync(top_indices.data(), d_selected_points,
                                      result_k * sizeof(uint32_t), cudaMemcpyDeviceToHost, stream));
            CUDA_CHECK(cudaStreamSynchronize(stream));
            
            // 构建结果
            results[q].reserve(result_k);
            for (int i = 0; i < result_k; ++i) {
                results[q].push_back({top_distances[i], selected_points[top_indices[i]]});
            }
        }
        
        return results;
    }

public:
    // 单个查询搜索
    std::priority_queue<std::pair<float, uint32_t>> search(const float* query, int k, int nprobe) {
        std::vector<float> queries(query, query + vector_dim);
        auto batch_results = batch_search(queries, k, nprobe);
        
        std::priority_queue<std::pair<float, uint32_t>> result;
        if (!batch_results.empty()) {
            for (const auto& pair : batch_results[0]) {
                result.push(pair);
            }
        }
        return result;
    }
};