#pragma once

#include <vector>
#include <queue>
#include <cstdint>
#include <algorithm>
#include <limits>
#include <cmath>
#include <numeric>
#include <random>
#include <chrono>
#include <iostream>
#include <cfloat>  // 添加这个头文件以使用 FLT_MAX

#ifdef __cplusplus
extern "C" {
#endif

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <device_launch_parameters.h>

#ifdef __cplusplus
}
#endif

// CUDA错误检查宏
#define CHECK_CUDA(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << " - " << cudaGetErrorString(err) << std::endl; \
        exit(1); \
    } \
} while(0)

#define CHECK_CUBLAS(call) do { \
    cublasStatus_t stat = call; \
    if (stat != CUBLAS_STATUS_SUCCESS) { \
        std::cerr << "cuBLAS error at " << __FILE__ << ":" << __LINE__ << std::endl; \
        exit(1); \
    } \
} while(0)

// CUDA核函数：计算内积距离
__global__ void compute_distances_kernel(const float* base_data, const float* query_data,
                                        float* distances, size_t base_num, size_t query_num, size_t dim) {
    int base_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int query_idx = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (base_idx < base_num && query_idx < query_num) {
        float dot_product = 0.0f;
        for (size_t d = 0; d < dim; ++d) {
            dot_product += base_data[base_idx * dim + d] * query_data[query_idx * dim + d];
        }
        distances[query_idx * base_num + base_idx] = 1.0f - dot_product;
    }
}

// CUDA核函数：找到每个查询的top-k结果
__global__ void find_topk_kernel(const float* distances, uint32_t* indices, float* values,
                                size_t base_num, size_t query_num, size_t k) {
    int query_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (query_idx < query_num) {
        const float* query_distances = distances + query_idx * base_num;
        uint32_t* query_indices = indices + query_idx * k;
        float* query_values = values + query_idx * k;
        
        // 简单的选择排序找到top-k
        for (size_t i = 0; i < k && i < base_num; ++i) {
            float min_dist = FLT_MAX;
            uint32_t min_idx = 0;
            
            for (size_t j = 0; j < base_num; ++j) {
                bool already_selected = false;
                for (size_t prev = 0; prev < i; ++prev) {
                    if (query_indices[prev] == j) {
                        already_selected = true;
                        break;
                    }
                }
                
                if (!already_selected && query_distances[j] < min_dist) {
                    min_dist = query_distances[j];
                    min_idx = j;
                }
            }
            
            query_indices[i] = min_idx;
            query_values[i] = min_dist;
        }
    }
}

// CUDA核函数：计算查询到质心的距离
__global__ void compute_centroid_distances_kernel(const float* queries, const float* centroids,
                                                 float* distances, size_t query_num, size_t centroid_num, size_t dim) {
    int query_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int centroid_idx = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (query_idx < query_num && centroid_idx < centroid_num) {
        float dot_product = 0.0f;
        for (size_t d = 0; d < dim; ++d) {
            dot_product += queries[query_idx * dim + d] * centroids[centroid_idx * dim + d];
        }
        distances[query_idx * centroid_num + centroid_idx] = 1.0f - dot_product;
    }
}

class IVFIndexGPU {
private:
    float* d_base_data;           // GPU上的基准数据
    float* d_centroids;           // GPU上的质心数据
    uint32_t* d_inverted_lists;   // GPU上的倒排列表
    size_t* d_list_offsets;       // 每个簇的起始偏移
    size_t* d_list_sizes;         // 每个簇的大小
    
    float* d_temp_distances;      // 临时距离数组
    uint32_t* d_temp_indices;     // 临时索引数组
    float* d_temp_values;         // 临时值数组
    
    cublasHandle_t cublas_handle;
    
    size_t num_base_vectors;
    size_t vector_dim;
    size_t num_clusters;
    size_t max_batch_size;
    
    std::vector<float> h_centroids;
    std::vector<std::vector<uint32_t>> h_inverted_lists;

public:
    IVFIndexGPU(float* base_data, size_t n_base, size_t dim, size_t n_clusters, size_t batch_size = 64)
        : num_base_vectors(n_base), vector_dim(dim), num_clusters(n_clusters), max_batch_size(batch_size) {
        
        // 初始化cuBLAS
        CHECK_CUBLAS(cublasCreate(&cublas_handle));
        
        // 分配GPU内存
        allocate_gpu_memory();
        
        // 复制基准数据到GPU
        CHECK_CUDA(cudaMemcpy(d_base_data, base_data, n_base * dim * sizeof(float), cudaMemcpyHostToDevice));
        
        // 在CPU上构建IVF索引
        build_index_cpu(base_data);
        
        // 将索引数据复制到GPU
        copy_index_to_gpu();
    }
    
    ~IVFIndexGPU() {
        // 释放GPU内存
        cudaFree(d_base_data);
        cudaFree(d_centroids);
        cudaFree(d_inverted_lists);
        cudaFree(d_list_offsets);
        cudaFree(d_list_sizes);
        cudaFree(d_temp_distances);
        cudaFree(d_temp_indices);
        cudaFree(d_temp_values);
        
        cublasDestroy(cublas_handle);
    }

private:
    void allocate_gpu_memory() {
        // 分配基准数据内存
        CHECK_CUDA(cudaMalloc(&d_base_data, num_base_vectors * vector_dim * sizeof(float)));
        
        // 分配质心内存
        CHECK_CUDA(cudaMalloc(&d_centroids, num_clusters * vector_dim * sizeof(float)));
        
        // 分配倒排列表内存（预估最大大小）
        CHECK_CUDA(cudaMalloc(&d_inverted_lists, num_base_vectors * sizeof(uint32_t)));
        CHECK_CUDA(cudaMalloc(&d_list_offsets, num_clusters * sizeof(size_t)));
        CHECK_CUDA(cudaMalloc(&d_list_sizes, num_clusters * sizeof(size_t)));
        
        // 分配临时内存
        CHECK_CUDA(cudaMalloc(&d_temp_distances, max_batch_size * num_base_vectors * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_temp_indices, max_batch_size * 100 * sizeof(uint32_t))); // 假设k<=100
        CHECK_CUDA(cudaMalloc(&d_temp_values, max_batch_size * 100 * sizeof(float)));
    }
    
    void build_index_cpu(float* base_data) {
        // 简化的k-means算法构建质心
        h_centroids.resize(num_clusters * vector_dim);
        h_inverted_lists.resize(num_clusters);
        
        // 随机初始化质心
        std::mt19937 rng(42);
        std::uniform_int_distribution<size_t> dist(0, num_base_vectors - 1);
        
        for (size_t c = 0; c < num_clusters; ++c) {
            size_t random_idx = dist(rng);
            std::copy(base_data + random_idx * vector_dim,
                     base_data + (random_idx + 1) * vector_dim,
                     h_centroids.begin() + c * vector_dim);
        }
        
        // 简化的k-means迭代
        std::vector<int> assignments(num_base_vectors);
        for (int iter = 0; iter < 10; ++iter) {
            // 分配点到最近质心
            for (size_t i = 0; i < num_base_vectors; ++i) {
                float min_dist = std::numeric_limits<float>::max();
                int best_cluster = 0;
                
                for (size_t c = 0; c < num_clusters; ++c) {
                    float dist = 0.0f;
                    for (size_t d = 0; d < vector_dim; ++d) {
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
            
            for (size_t i = 0; i < num_base_vectors; ++i) {
                int cluster = assignments[i];
                counts[cluster]++;
                for (size_t d = 0; d < vector_dim; ++d) {
                    new_centroids[cluster][d] += base_data[i * vector_dim + d];
                }
            }
            
            for (size_t c = 0; c < num_clusters; ++c) {
                if (counts[c] > 0) {
                    for (size_t d = 0; d < vector_dim; ++d) {
                        h_centroids[c * vector_dim + d] = new_centroids[c][d] / counts[c];
                    }
                }
            }
        }
        
        // 构建倒排列表
        for (size_t c = 0; c < num_clusters; ++c) {
            h_inverted_lists[c].clear();
        }
        
        for (size_t i = 0; i < num_base_vectors; ++i) {
            h_inverted_lists[assignments[i]].push_back(i);
        }
    }
    
    void copy_index_to_gpu() {
        // 复制质心到GPU
        CHECK_CUDA(cudaMemcpy(d_centroids, h_centroids.data(), 
                             num_clusters * vector_dim * sizeof(float), cudaMemcpyHostToDevice));
        
        // 准备倒排列表的扁平化数据
        std::vector<uint32_t> flat_lists;
        std::vector<size_t> offsets(num_clusters);
        std::vector<size_t> sizes(num_clusters);
        
        for (size_t c = 0; c < num_clusters; ++c) {
            offsets[c] = flat_lists.size();
            sizes[c] = h_inverted_lists[c].size();
            flat_lists.insert(flat_lists.end(), h_inverted_lists[c].begin(), h_inverted_lists[c].end());
        }
        
        // 复制到GPU
        CHECK_CUDA(cudaMemcpy(d_inverted_lists, flat_lists.data(), 
                             flat_lists.size() * sizeof(uint32_t), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_list_offsets, offsets.data(), 
                             num_clusters * sizeof(size_t), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_list_sizes, sizes.data(), 
                             num_clusters * sizeof(size_t), cudaMemcpyHostToDevice));
    }

public:
    // 批量搜索函数
    std::vector<std::vector<std::pair<float, uint32_t>>> batch_search(
        const std::vector<float>& queries, size_t k, size_t nprobe) {
        
        size_t query_num = queries.size() / vector_dim;
        std::vector<std::vector<std::pair<float, uint32_t>>> results(query_num);
        
        // 处理批量查询
        for (size_t batch_start = 0; batch_start < query_num; batch_start += max_batch_size) {
            size_t batch_end = std::min(batch_start + max_batch_size, query_num);
            size_t current_batch_size = batch_end - batch_start;
            
            // 复制当前批次的查询到GPU
            float* d_queries;
            CHECK_CUDA(cudaMalloc(&d_queries, current_batch_size * vector_dim * sizeof(float)));
            CHECK_CUDA(cudaMemcpy(d_queries, queries.data() + batch_start * vector_dim,
                                 current_batch_size * vector_dim * sizeof(float), cudaMemcpyHostToDevice));
            
            // 处理当前批次
            auto batch_results = process_batch(d_queries, current_batch_size, k, nprobe);
            
            // 复制结果
            for (size_t i = 0; i < current_batch_size; ++i) {
                results[batch_start + i] = batch_results[i];
            }
            
            cudaFree(d_queries);
        }
        
        return results;
    }

private:
    std::vector<std::vector<std::pair<float, uint32_t>>> process_batch(
        float* d_queries, size_t batch_size, size_t k, size_t nprobe) {
        
        // 1. 计算查询到所有质心的距离
        float* d_centroid_distances;
        CHECK_CUDA(cudaMalloc(&d_centroid_distances, batch_size * num_clusters * sizeof(float)));
        
        dim3 block_size(16, 16);
        dim3 grid_size((batch_size + block_size.x - 1) / block_size.x,
                      (num_clusters + block_size.y - 1) / block_size.y);
        
        compute_centroid_distances_kernel<<<grid_size, block_size>>>(
            d_queries, d_centroids, d_centroid_distances, batch_size, num_clusters, vector_dim);
        CHECK_CUDA(cudaDeviceSynchronize());
        
        // 2. 找到每个查询的nprobe个最近质心
        std::vector<float> h_centroid_distances(batch_size * num_clusters);
        CHECK_CUDA(cudaMemcpy(h_centroid_distances.data(), d_centroid_distances,
                             batch_size * num_clusters * sizeof(float), cudaMemcpyDeviceToHost));
        
        std::vector<std::vector<int>> selected_clusters(batch_size);
        for (size_t q = 0; q < batch_size; ++q) {
            std::vector<std::pair<float, int>> cluster_dists;
            for (size_t c = 0; c < num_clusters; ++c) {
                cluster_dists.push_back({h_centroid_distances[q * num_clusters + c], c});
            }
            std::sort(cluster_dists.begin(), cluster_dists.end());
            
            selected_clusters[q].clear();
            for (size_t i = 0; i < std::min(nprobe, num_clusters); ++i) {
                selected_clusters[q].push_back(cluster_dists[i].second);
            }
        }
        
        // 3. 在选定的簇中搜索
        std::vector<std::vector<std::pair<float, uint32_t>>> results(batch_size);
        
        // 为每个查询单独处理（简化实现）
        for (size_t q = 0; q < batch_size; ++q) {
            std::priority_queue<std::pair<float, uint32_t>> pq;
            
            for (int cluster_idx : selected_clusters[q]) {
                size_t list_size = h_inverted_lists[cluster_idx].size();
                if (list_size == 0) continue;
                
                // 计算到簇内所有点的距离
                for (uint32_t point_idx : h_inverted_lists[cluster_idx]) {
                    float dist = 0.0f;
                    
                    // 从GPU复制查询向量和基准向量到CPU进行距离计算（简化实现）
                    std::vector<float> query_vec(vector_dim);
                    std::vector<float> base_vec(vector_dim);
                    
                    CHECK_CUDA(cudaMemcpy(query_vec.data(), d_queries + q * vector_dim,
                                         vector_dim * sizeof(float), cudaMemcpyDeviceToHost));
                    CHECK_CUDA(cudaMemcpy(base_vec.data(), d_base_data + point_idx * vector_dim,
                                         vector_dim * sizeof(float), cudaMemcpyDeviceToHost));
                    
                    float dot_product = 0.0f;
                    for (size_t d = 0; d < vector_dim; ++d) {
                        dot_product += query_vec[d] * base_vec[d];
                    }
                    dist = 1.0f - dot_product;
                    
                    if (pq.size() < k) {
                        pq.push({dist, point_idx});
                    } else if (dist < pq.top().first) {
                        pq.pop();
                        pq.push({dist, point_idx});
                    }
                }
            }
            
            // 将结果从优先队列转换为向量
            std::vector<std::pair<float, uint32_t>> query_result;
            while (!pq.empty()) {
                query_result.push_back(pq.top());
                pq.pop();
            }
            std::reverse(query_result.begin(), query_result.end());
            results[q] = query_result;
        }
        
        cudaFree(d_centroid_distances);
        return results;

public:
    // 单个查询搜索（为了兼容现有接口）
    std::priority_queue<std::pair<float, uint32_t>> search(const float* query, size_t k, size_t nprobe) {
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