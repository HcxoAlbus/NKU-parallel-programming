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

// 简化的错误检查宏
#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        printf("CUDA error at %s:%d - %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(1); \
    } \
} while(0)

// CUDA核函数：计算查询到质心的距离
__global__ void compute_centroid_distances(const float* queries, const float* centroids,
                                          float* distances, int query_num, int centroid_num, int dim) {
    int query_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int centroid_idx = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (query_idx < query_num && centroid_idx < centroid_num) {
        float dot_product = 0.0f;
        for (int d = 0; d < dim; ++d) {
            dot_product += queries[query_idx * dim + d] * centroids[centroid_idx * dim + d];
        }
        distances[query_idx * centroid_num + centroid_idx] = 1.0f - dot_product;
    }
}

// CUDA核函数：计算距离
__global__ void compute_distances(const float* base_data, const float* query_data,
                                 float* distances, int base_num, int query_num, int dim) {
    int base_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int query_idx = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (base_idx < base_num && query_idx < query_num) {
        float dot_product = 0.0f;
        for (int d = 0; d < dim; ++d) {
            dot_product += base_data[base_idx * dim + d] * query_data[query_idx * dim + d];
        }
        distances[query_idx * base_num + base_idx] = 1.0f - dot_product;
    }
}

// 简化的GPU IVF实现
class SimpleIVFGPU {
private:
    float* d_base_data;
    float* d_centroids;
    float* d_temp_distances;
    
    int num_base_vectors;
    int vector_dim;
    int num_clusters;
    int max_batch_size;
    
    std::vector<float> h_centroids;
    std::vector<std::vector<uint32_t>> h_inverted_lists;
    std::vector<float> h_base_data; // 添加CPU端的base_data副本

public:
    SimpleIVFGPU(float* base_data, int n_base, int dim, int n_clusters, int batch_size = 32)
        : num_base_vectors(n_base), vector_dim(dim), num_clusters(n_clusters), max_batch_size(batch_size) {
        
        // 保存base_data的副本到CPU内存
        h_base_data.resize(n_base * dim);
        std::copy(base_data, base_data + n_base * dim, h_base_data.begin());
        
        // 分配GPU内存
        CUDA_CHECK(cudaMalloc(&d_base_data, n_base * dim * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_centroids, n_clusters * dim * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_temp_distances, batch_size * n_base * sizeof(float)));
        
        // 复制基准数据到GPU
        CUDA_CHECK(cudaMemcpy(d_base_data, base_data, n_base * dim * sizeof(float), cudaMemcpyHostToDevice));
        
        // 在CPU上构建索引
        build_index_cpu(base_data);
        
        // 复制质心到GPU
        CUDA_CHECK(cudaMemcpy(d_centroids, h_centroids.data(), 
                             n_clusters * dim * sizeof(float), cudaMemcpyHostToDevice));
    }
    
    ~SimpleIVFGPU() {
        cudaFree(d_base_data);
        cudaFree(d_centroids);
        cudaFree(d_temp_distances);
    }

private:
    void build_index_cpu(float* base_data) {
        // 简化的k-means算法
        h_centroids.resize(num_clusters * vector_dim);
        h_inverted_lists.resize(num_clusters);
        
        // 随机选择初始质心
        for (int c = 0; c < num_clusters; ++c) {
            int random_idx = (c * 17 + 42) % num_base_vectors; // 简单的伪随机
            for (int d = 0; d < vector_dim; ++d) {
                h_centroids[c * vector_dim + d] = base_data[random_idx * vector_dim + d];
            }
        }
        
        // 分配点到最近质心
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
    }

public:
    // 批量搜索
    std::vector<std::vector<std::pair<float, uint32_t>>> batch_search(
        const std::vector<float>& queries, int k, int nprobe) {
        
        int query_num = queries.size() / vector_dim;
        std::vector<std::vector<std::pair<float, uint32_t>>> results(query_num);
        
        // 处理批量查询
        for (int batch_start = 0; batch_start < query_num; batch_start += max_batch_size) {
            int batch_end = std::min(batch_start + max_batch_size, query_num);
            int current_batch_size = batch_end - batch_start;
            
            // 分配当前批次的GPU内存
            float* d_queries;
            float* d_centroid_distances;
            CUDA_CHECK(cudaMalloc(&d_queries, current_batch_size * vector_dim * sizeof(float)));
            CUDA_CHECK(cudaMalloc(&d_centroid_distances, current_batch_size * num_clusters * sizeof(float)));
            
            // 复制查询到GPU
            CUDA_CHECK(cudaMemcpy(d_queries, queries.data() + batch_start * vector_dim,
                                 current_batch_size * vector_dim * sizeof(float), cudaMemcpyHostToDevice));
            
            // 计算到质心的距离
            dim3 block_size(16, 16);
            dim3 grid_size((current_batch_size + block_size.x - 1) / block_size.x,
                          (num_clusters + block_size.y - 1) / block_size.y);
            
            compute_centroid_distances<<<grid_size, block_size>>>(
                d_queries, d_centroids, d_centroid_distances, current_batch_size, num_clusters, vector_dim);
            CUDA_CHECK(cudaDeviceSynchronize());
            
            // 复制结果回CPU
            std::vector<float> h_centroid_distances(current_batch_size * num_clusters);
            CUDA_CHECK(cudaMemcpy(h_centroid_distances.data(), d_centroid_distances,
                                 current_batch_size * num_clusters * sizeof(float), cudaMemcpyDeviceToHost));
            
            // 在CPU上处理其余逻辑
            auto batch_results = process_batch_cpu(queries, batch_start, current_batch_size, 
                                                  h_centroid_distances, k, nprobe);
            
            // 复制结果
            for (int i = 0; i < current_batch_size; ++i) {
                results[batch_start + i] = batch_results[i];
            }
            
            cudaFree(d_queries);
            cudaFree(d_centroid_distances);
        }
        
        return results;
    }

private:
    std::vector<std::vector<std::pair<float, uint32_t>>> process_batch_cpu(
        const std::vector<float>& queries, int batch_start, int batch_size,
        const std::vector<float>& centroid_distances, int k, int nprobe) {
        
        std::vector<std::vector<std::pair<float, uint32_t>>> results(batch_size);
        
        for (int q = 0; q < batch_size; ++q) {
            // 找到最近的nprobe个质心
            std::vector<std::pair<float, int>> cluster_dists;
            for (int c = 0; c < num_clusters; ++c) {
                cluster_dists.push_back({centroid_distances[q * num_clusters + c], c});
            }
            std::sort(cluster_dists.begin(), cluster_dists.end());
            
            // 在选定的簇中搜索
            std::priority_queue<std::pair<float, uint32_t>> pq;
            
            for (int i = 0; i < std::min(nprobe, num_clusters); ++i) {
                int cluster_idx = cluster_dists[i].second;
                const auto& point_list = h_inverted_lists[cluster_idx];
                
                for (uint32_t point_idx : point_list) {
                    // 边界检查
                    if (point_idx >= (uint32_t)num_base_vectors) {
                        continue; // 跳过无效索引
                    }
                    
                    // 计算距离 - 修复：使用h_base_data而不是queries
                    float dot_product = 0.0f;
                    int query_idx = batch_start + q;
                    for (int d = 0; d < vector_dim; ++d) {
                        dot_product += queries[query_idx * vector_dim + d] * 
                                      h_base_data[point_idx * vector_dim + d]; // 修复：使用base_data
                    }
                    float dist = 1.0f - dot_product;
                    
                    if ((int)pq.size() < k) {
                        pq.push({dist, point_idx});
                    } else if (dist < pq.top().first) {
                        pq.pop();
                        pq.push({dist, point_idx});
                    }
                }
            }
            
            // 转换结果
            std::vector<std::pair<float, uint32_t>> query_result;
            while (!pq.empty()) {
                query_result.push_back(pq.top());
                pq.pop();
            }
            std::reverse(query_result.begin(), query_result.end());
            results[q] = query_result;
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