#pragma once
#include <queue>
#include <vector>
#include <arm_neon.h>
#include <cstdint>
#include <algorithm>
#include <random>

class FastScanPQ {
private:
    const size_t vecdim;           // 原始向量维度
    const size_t nsub;             // 子空间数量
    const size_t dsub;             // 每个子空间的维度
    const size_t ksub;             // 每个子空间的聚类数量 (256)
    
    std::vector<std::vector<float>> centroids;  // [nsub][ksub * dsub]
    std::vector<std::vector<uint8_t>> codes;    // [nbase][nsub]
    
    // 训练代码省略，与PQ相同...
    void train(float* base, size_t nbase) {
        // 初始化中心点
        centroids.resize(nsub, std::vector<float>(ksub * dsub, 0));
        
        // 为每个子空间准备训练数据
        for (size_t sub = 0; sub < nsub; ++sub) {
            size_t d_start = sub * dsub;
            size_t d_end = std::min(d_start + dsub, vecdim);
            size_t d_sub_actual = d_end - d_start;
            
            std::vector<std::vector<float>> subspace_data(nbase, std::vector<float>(d_sub_actual));
            
            for (size_t i = 0; i < nbase; ++i) {
                for (size_t j = 0; j < d_sub_actual; ++j) {
                    subspace_data[i][j] = base[(i * vecdim) + d_start + j];
                }
            }
            
            // 对子空间进行k-means聚类
            kmeans(subspace_data, sub);
        }
    }
    
    // k-means聚类
    void kmeans(const std::vector<std::vector<float>>& traindata, size_t sub, 
                size_t n_iter = 25) {
        const size_t n = traindata.size();
        std::vector<size_t> assign(n);
        std::vector<size_t> nassign(ksub);
        std::vector<float> dist(n);
        
        // 随机初始化中心点
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<> dis(0, n - 1);
        
        // 随机选择ksub个点作为初始中心
        std::vector<size_t> init_ids(ksub);
        for (size_t i = 0; i < ksub; ++i) {
            init_ids[i] = dis(gen);
        }
        
        for (size_t i = 0; i < ksub; ++i) {
            for (size_t j = 0; j < dsub; ++j) {
                centroids[sub][i * dsub + j] = traindata[init_ids[i]][j];
            }
        }
        
        // k-means迭代
        for (size_t iter = 0; iter < n_iter; ++iter) {
            // 计算每个点到最近中心的距离
            for (size_t i = 0; i < n; ++i) {
                float min_dist = std::numeric_limits<float>::max();
                size_t min_idx = 0;
                
                for (size_t k = 0; k < ksub; ++k) {
                    float d = 0;
                    for (size_t j = 0; j < dsub; ++j) {
                        float diff = traindata[i][j] - centroids[sub][k * dsub + j];
                        d += diff * diff;
                    }
                    
                    if (d < min_dist) {
                        min_dist = d;
                        min_idx = k;
                    }
                }
                
                assign[i] = min_idx;
                dist[i] = min_dist;
            }
            
            // 更新中心点
            std::fill(nassign.begin(), nassign.end(), 0);
            std::vector<float> new_centroids(ksub * dsub, 0);
            
            for (size_t i = 0; i < n; ++i) {
                size_t cluster = assign[i];
                nassign[cluster]++;
                
                for (size_t j = 0; j < dsub; ++j) {
                    new_centroids[cluster * dsub + j] += traindata[i][j];
                }
            }
            
            for (size_t k = 0; k < ksub; ++k) {
                if (nassign[k] == 0) continue;
                
                for (size_t j = 0; j < dsub; ++j) {
                    centroids[sub][k * dsub + j] = new_centroids[k * dsub + j] / nassign[k];
                }
            }
        }
    }
    
    // 编码向量集
    void encode(float* base, size_t nbase) {
        codes.resize(nbase, std::vector<uint8_t>(nsub, 0));
        
        for (size_t i = 0; i < nbase; ++i) {
            for (size_t sub = 0; sub < nsub; ++sub) {
                size_t d_start = sub * dsub;
                size_t d_end = std::min(d_start + dsub, vecdim);
                size_t d_sub_actual = d_end - d_start;
                
                float min_dist = std::numeric_limits<float>::max();
                uint8_t min_idx = 0;
                
                for (size_t k = 0; k < ksub; ++k) {
                    float dist = 0;
                    for (size_t j = 0; j < d_sub_actual; ++j) {
                        float diff = base[i * vecdim + d_start + j] - centroids[sub][k * dsub + j];
                        dist += diff * diff;
                    }
                    
                    if (dist < min_dist) {
                        min_dist = dist;
                        min_idx = k;
                    }
                }
                
                codes[i][sub] = min_idx;
            }
        }
    }

public:
    FastScanPQ(float* base, size_t nbase, size_t vecdim, size_t nsub = 4)
        : vecdim(vecdim), nsub(nsub), dsub(vecdim / nsub), ksub(256) {
        // 训练码本
        train(base, nbase);
        // 编码基础向量集
        encode(base, nbase);
    }
    
    // 使用FastScan技术的PQ搜索
    std::priority_queue<std::pair<float, uint32_t>> search(
        float* query, size_t nbase, size_t k) {
        
        // 计算每个子空间的距离表
        std::vector<float32x4_t> distance_tables[nsub];
        for (size_t sub = 0; sub < nsub; ++sub) {
            distance_tables[sub].resize(ksub / 4);  // 每个SIMD寄存器存储4个float
            
            size_t d_start = sub * dsub;
            size_t d_end = std::min(d_start + dsub, vecdim);
            size_t d_sub_actual = d_end - d_start;
            
            std::vector<float> distances(ksub);
            
            // 计算查询向量与每个中心点的距离
            for (size_t centroid = 0; centroid < ksub; ++centroid) {
                float ip = 0;
                for (size_t j = 0; j < d_sub_actual; ++j) {
                    ip += query[d_start + j] * centroids[sub][centroid * dsub + j];
                }
                distances[centroid] = 1.0f - ip;  // 转换为距离
            }
            
            // 填充SIMD寄存器
            for (size_t i = 0; i < ksub; i += 4) {
                distance_tables[sub][i/4] = vld1q_f32(&distances[i]);
            }
        }
        
        // 使用SIMD计算距离并排序
        std::priority_queue<std::pair<float, uint32_t>> result;
        
        // 每次处理16个向量
        const size_t block_size = 16;
        
        for (size_t b = 0; b < nbase; b += block_size) {
            size_t current_block = std::min(block_size, nbase - b);
            
            std::vector<float32x4_t> block_distances(current_block);
            std::fill(block_distances.begin(), block_distances.end(), vdupq_n_f32(0));
            
            // 计算当前块中所有向量的距离
            for (size_t sub = 0; sub < nsub; ++sub) {
                for (size_t i = 0; i < current_block; ++i) {
                    size_t idx = b + i;
                    uint8_t code = codes[idx][sub];
                    
                    // 从寄存器中查找预计算的距离
                    float32x4_t distance = distance_tables[sub][code/4];
                    // 选择正确的距离值
                    float dist_val = vgetq_lane_f32(distance, code % 4);
                    
                    // 累加距离
                    float32x4_t current_dist = block_distances[i];
                    block_distances[i] = vsetq_lane_f32(
                        vgetq_lane_f32(current_dist, 0) + dist_val, 
                        current_dist, 0);
                }
            }
            
            // 收集结果并更新优先队列
            for (size_t i = 0; i < current_block; ++i) {
                float dist = vgetq_lane_f32(block_distances[i], 0);
                size_t idx = b + i;
                
                if (result.size() < k) {
                    result.push({dist, idx});
                } else if (dist < result.top().first) {
                    result.pop();
                    result.push({dist, idx});
                }
            }
        }
        
        return result;
    }
};

// FastScan PQ优化的ANNS搜索
std::priority_queue<std::pair<float, uint32_t>> fastscan_pq_search(
    float* base, float* query, size_t base_number, size_t vecdim, size_t k) {
    
    // 构建FastScan PQ量化器
    FastScanPQ fpq(base, base_number, vecdim);
    
    // 使用FastScan PQ进行搜索
    return fpq.search(query, base_number, k);
}