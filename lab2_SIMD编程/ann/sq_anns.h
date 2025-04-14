#pragma once
#include <queue>
#include <arm_neon.h>
#include <cstdint>
#include <vector>
#include <algorithm>

class ScalarQuantizer {
private:
    std::vector<float> scales;// 缩放因子
    std::vector<float> min_vals;// 每个维度的最小值，用于量化时的偏移
    // 存储量化后的基向量集合，每个向量的每个维度是一个 uint8_t。
    std::vector<std::vector<uint8_t>> quantized_base;
    size_t vecdim; // 向量维度
    size_t base_number; // 基向量数量

public:
    ScalarQuantizer(float* base, size_t base_number, size_t vecdim) 
        : vecdim(vecdim), base_number(base_number) {
        
        // 为每个维度计算缩放因子
        scales.resize(vecdim);
        min_vals.resize(vecdim);
        
        // 找到每个维度的最小值和最大值
        for (size_t d = 0; d < vecdim; ++d) {
            float min_val = base[d];
            float max_val = base[d];
            // 遍历每个基向量的当前维度，找到最小值和最大值
            for (size_t i = 1; i < base_number; ++i) {
                min_val = std::min(min_val, base[d + i*vecdim]);
                max_val = std::max(max_val, base[d + i*vecdim]);
            }
            
            min_vals[d] = min_val;
            scales[d] = (max_val - min_val) / 255.0f;  // 缩放到0-255范围
            if (scales[d] < 1e-6f) scales[d] = 1.0f;  // 避免除零
        }
        
        // 量化基础向量集
        quantized_base.resize(base_number, std::vector<uint8_t>(vecdim));
        for (size_t i = 0; i < base_number; ++i) {
            for (size_t d = 0; d < vecdim; ++d) {
                float val = base[d + i*vecdim];
                // 使用static_cast将float转换为uint8_t
                quantized_base[i][d] = static_cast<uint8_t>((val - min_vals[d]) / scales[d]);
            }
        }
    }
    
    // 使用NEON SIMD优化的量化内积计算，idx为基向量的索引
    float quantized_inner_product(const float* query, size_t idx) const {
        std::vector<uint8_t> quantized_query(vecdim);
        
        // 量化查询向量
        for (size_t d = 0; d < vecdim; ++d) {
            float val = query[d];
            val = std::max(val, min_vals[d]);
            val = std::min(val, min_vals[d] + scales[d] * 255);
            quantized_query[d] = static_cast<uint8_t>((val - min_vals[d]) / scales[d]);
        }
        
        // 使用SIMD计算内积
        int32_t sum = 0;
        size_t d = 0;
        
        // 每次处理16个uint8_t
        for (; d + 15 < vecdim; d += 16) {
            uint8x16_t a = vld1q_u8(&quantized_query[d]);
            uint8x16_t b = vld1q_u8(&quantized_base[idx][d]);
            
            // 使用SMLAL进行点乘累加
            // prod_low和prod_high分别存储低8位和高8位的乘积
            uint16x8_t prod_low = vmull_u8(vget_low_u8(a), vget_low_u8(b));
            uint16x8_t prod_high = vmull_u8(vget_high_u8(a), vget_high_u8(b));
            
            // 水平加和，与simd_anns中的水平加和过程类似
            uint32x4_t sum_low = vpaddlq_u16(prod_low);
            uint32x4_t sum_high = vpaddlq_u16(prod_high);
            uint32x4_t total = vaddq_u32(sum_low, sum_high);
            
            uint32x2_t total_low = vget_low_u32(total);
            uint32x2_t total_high = vget_high_u32(total);
            uint32x2_t total_sum = vadd_u32(total_low, total_high);
            total_sum = vpadd_u32(total_sum, total_sum);
            // 将16个元素的和加到sum中
            sum += vget_lane_u32(total_sum, 0);
        }
        
        // 处理剩余元素
        for (; d < vecdim; ++d) {
            sum += quantized_query[d] * quantized_base[idx][d];
        }
        
        // 反量化内积结果
        float scale_factor = 0.0f;
        float bias = 0.0f;
        for (size_t d = 0; d < vecdim; ++d) {
            scale_factor += scales[d] * scales[d];
            bias += min_vals[d] * min_vals[d];
        }
        return 1.0f - (sum * scale_factor + bias);
    }
    
    std::priority_queue<std::pair<float, uint32_t>> search(
        float* query, size_t k) const {
        
        std::priority_queue<std::pair<float, uint32_t>> q;
        
        for (size_t i = 0; i < base_number; ++i) {
            float dis = quantized_inner_product(query, i);
            
            if (q.size() < k) {
                q.push({dis, i});
            } else {
                if (dis < q.top().first) {
                    q.push({dis, i});
                    q.pop();
                }
            }
        }
        return q;
    }
};

// SQ优化的ANNS搜索
std::priority_queue<std::pair<float, uint32_t>> sq_search(
    float* base, float* query, size_t base_number, size_t vecdim, size_t k) {
    
    // 构建量化器
    ScalarQuantizer quantizer(base, base_number, vecdim);
    
    // 使用量化器进行搜索
    return quantizer.search(query, k);
}