#pragma once
#include <queue>
#include <vector>
#include <algorithm>
#include <random>
#include <arm_neon.h>
#include <cstdint>
#include <cstring>

class ProductQuantizer {
private:
    const size_t vecdim;           // 原始向量维度
    const size_t nsub;             // 子空间数量
    const size_t dsub;             // 每个子空间的维度，即vecdim / nsub
    const size_t ksub;             // 每个子空间的聚类数量 (256)
    
    std::vector<std::vector<float>> centroids;  // 每个子空间的聚类中心点，维度为[nsub][ksub * dsub]
    std::vector<std::vector<uint8_t>> codes;    // 存储量化后的向量，[nbase][nsub]
    
    // k-means聚类
    /*
        * traindata: 训练数据集
        * sub: 当前子空间索引
        * n_iter: k-means算法的迭代次数
        * 
        * 使用k-means算法对每个子空间进行聚类，初始化中心点为随机选择的点
        * 然后迭代更新中心点，直到收敛或达到指定的迭代次数。
    */
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
    
    // 训练PQ码本
    /*
        * base: 原始数据集
        * nbase: 原始数据集的大小
        * 
        * 对每个子空间进行k-means聚类，得到每个子空间的中心点。
    */
    void train(float* base, size_t nbase) {
        // centroids用于存储每个子空间的码点
        // 每个子空间有 ksub 个聚类中心，每个中心点是一个 dsub 维的向量。
        centroids.resize(nsub, std::vector<float>(ksub * dsub, 0));
        
        // 为每个子空间准备训练数据
        for (size_t sub = 0; sub < nsub; ++sub) {
            // d_start 和 d_end 确定当前子空间的维度范围
            size_t d_start = sub * dsub;
            size_t d_end = std::min(d_start + dsub, vecdim);
            // 计算当前子空间的实际维度大小（最后一个子空间可能不满）
            size_t d_sub_actual = d_end - d_start;
            // 存储当前子空间的训练数据
            // 每个子空间的训练数据是 nbase 个向量，每个向量有 d_sub_actual 维
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
    
    // 编码向量集，PQ的核心算法流程
    void encode(float* base, size_t nbase) {
        // 存储编码后的结果
        // 每个子空间的编码结果是一个 uint8_t 值，表示该子空间中最接近的聚类中心点的索引
        codes.resize(nbase, std::vector<uint8_t>(nsub, 0));
        // 遍历每个向量
        for (size_t i = 0; i < nbase; ++i) {
            // 遍历每个子空间
            for (size_t sub = 0; sub < nsub; ++sub) {
                size_t d_start = sub * dsub;
                size_t d_end = std::min(d_start + dsub, vecdim);
                size_t d_sub_actual = d_end - d_start;
                // 当前子空间中的最小距离，初始设置为float的最大值
                float min_dist = std::numeric_limits<float>::max();
                uint8_t min_idx = 0;// 最接近的聚类中心点的索引
                // 遍历当前子空间的所有聚类中心（码本中的点），计算向量到每个聚类中心的欧几里得距离。
                for (size_t k = 0; k < ksub; ++k) {
                    float dist = 0;
                    for (size_t j = 0; j < d_sub_actual; ++j) {
                        float diff = base[i * vecdim + d_start + j] - centroids[sub][k * dsub + j];
                        dist += diff * diff;
                    }
                    // 更新最小距离和索引
                    if (dist < min_dist) {
                        min_dist = dist;
                        min_idx = k;
                    }
                }
                // 将当前子空间的编码结果（最接近的聚类中心的索引）保存到 codes 中
                codes[i][sub] = min_idx;
            }
        }
    }

public:
    ProductQuantizer(float* base, size_t nbase, size_t vecdim, size_t nsub = 4)
        : vecdim(vecdim), nsub(nsub), dsub(vecdim / nsub), ksub(256) {
        // 训练码本
        train(base, nbase);
        // 编码基础向量集
        encode(base, nbase);
    }
    
    // 使用PQ进行搜索
    std::priority_queue<std::pair<float, uint32_t>> search(
        float* query, size_t nbase, size_t k) {
        
        // 为查询计算距离表
        std::vector<std::vector<float>> distance_tables(nsub, std::vector<float>(ksub));
        
        for (size_t sub = 0; sub < nsub; ++sub) {
            size_t d_start = sub * dsub;
            size_t d_end = std::min(d_start + dsub, vecdim);
            size_t d_sub_actual = d_end - d_start;
            
            for (size_t centroid = 0; centroid < ksub; ++centroid) {
                float ip = 0;
                for (size_t j = 0; j < d_sub_actual; ++j) {
                    ip += query[d_start + j] * centroids[sub][centroid * dsub + j];
                }
                distance_tables[sub][centroid] = 1.0f - ip;  // 转换为距离
            }
        }
        
        // 近似距离计算和排序
        std::priority_queue<std::pair<float, uint32_t>> result;
        
        for (size_t i = 0; i < nbase; ++i) {
            float dist = 0;
            
            for (size_t sub = 0; sub < nsub; ++sub) {
                dist += distance_tables[sub][codes[i][sub]];
            }
            
            if (result.size() < k) {
                result.push({dist, i});
            } else if (dist < result.top().first) {
                result.pop();
                result.push({dist, i});
            }
        }
        
        return result;
    }
};

// PQ优化的ANNS搜索
std::priority_queue<std::pair<float, uint32_t>> pq_search(
    float* base, float* query, size_t base_number, size_t vecdim, size_t k) {
    
    // 构建PQ量化器
    ProductQuantizer pq(base, base_number, vecdim);
    
    // 使用PQ进行搜索
    return pq.search(query, base_number, k);
}