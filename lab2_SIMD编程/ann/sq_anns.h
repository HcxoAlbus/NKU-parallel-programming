#pragma once
#include <queue>
#include <arm_neon.h>
#include <cstdint>
#include <vector>
#include <algorithm>
#include <limits> // for numeric_limits
#include <numeric> // for std::inner_product, std::accumulate (optional, for verification)

class ScalarQuantizer {
private:
    std::vector<float> scales;     // 缩放因子 (每个维度一个)
    std::vector<float> min_vals;   // 每个维度的最小值 (用于偏移)
    std::vector<uint8_t> quantized_base; // 扁平化存储量化后的基向量集合 (N * D)
    std::vector<uint32_t> quantized_base_norms; // 存储每个量化基向量的L2范数平方 (Σ qB[d]^2)
    size_t vecdim;                 // 向量维度
    size_t base_number;            // 基向量数量

    // 辅助函数：量化单个浮点值
    inline uint8_t quantize_value(float val, size_t dim) const {
        // 注意：需要确保 val 在量化范围内，调用者（quantize_query）负责处理
        float scaled_val = (val - min_vals[dim]) / scales[dim];
        // Clamp to [0, 255] before casting
        scaled_val = std::max(0.0f, std::min(255.0f, scaled_val));
        return static_cast<uint8_t>(scaled_val);
    }

    // 辅助函数：计算两个量化向量的点积 (NEON优化)
    // q_query: 指向量化查询向量的指针
    // base_idx: 基向量的索引
    uint32_t quantized_dot_product(const uint8_t* q_query, size_t base_idx) const {
        const uint8_t* q_base = quantized_base.data() + base_idx * vecdim;
        
        uint32x4_t sum_vec = vdupq_n_u32(0); // 累加器向量，初始化为0
        size_t d = 0;

        // 每次处理16个uint8_t
        for (; d + 15 < vecdim; d += 16) {
            uint8x16_t a = vld1q_u8(q_query + d);
            uint8x16_t b = vld1q_u8(q_base + d);

            // 计算点积: a[i]*b[i], 扩展到uint16
            uint16x8_t prod_low = vmull_u8(vget_low_u8(a), vget_low_u8(b));
            uint16x8_t prod_high = vmull_u8(vget_high_u8(a), vget_high_u8(b));

            // 累加到32位累加器
            // 将 uint16x8_t 两两配对相加到 uint32x4_t
            sum_vec = vpadalq_u16(sum_vec, prod_low); 
            sum_vec = vpadalq_u16(sum_vec, prod_high);
        }

        // 将累加器向量中的4个u32加起来得到最终和
        uint32_t sum = vaddvq_u32(sum_vec); // NEON 水平加和 (更简洁)
        // 如果 vaddvq_u32 不可用或性能不佳，可以使用原来的多步水平加和方法:
        // uint64x2_t temp_sum = vpaddlq_u32(sum_vec);
        // uint32_t sum = vgetq_lane_u64(temp_sum, 0) + vgetq_lane_u64(temp_sum, 1);


        // 处理剩余的元素 (少于16个)
        for (; d < vecdim; ++d) {
            sum += static_cast<uint32_t>(q_query[d]) * static_cast<uint32_t>(q_base[d]);
        }
        
        return sum;
    }


public:
    ScalarQuantizer(const float* base, size_t base_number, size_t vecdim)
        : vecdim(vecdim), base_number(base_number) {

        if (base_number == 0 || vecdim == 0) {
            // 处理空输入的情况
            return;
        }

        scales.resize(vecdim);
        min_vals.resize(vecdim);
        quantized_base.resize(base_number * vecdim);
        quantized_base_norms.resize(base_number, 0);

        // 1. 找到每个维度的最小值和最大值，并计算 scale 和 min_val
        for (size_t d = 0; d < vecdim; ++d) {
            float min_val = base[d];
            float max_val = base[d];
            // 遍历所有基向量的当前维度
            for (size_t i = 1; i < base_number; ++i) {
                float val = base[d + i * vecdim];
                min_val = std::min(min_val, val);
                max_val = std::max(max_val, val);
            }

            min_vals[d] = min_val;
            float range = max_val - min_val;
            // 避免除以零或极小值
            scales[d] = (range > 1e-6f) ? (range / 255.0f) : 1.0f; 
        }

        // 2. 量化基础向量集，并计算每个量化向量的 L2 范数平方
        for (size_t i = 0; i < base_number; ++i) {
            uint32_t current_norm_sq = 0;
            uint8_t* q_base_ptr = quantized_base.data() + i * vecdim;
            const float* base_ptr = base + i * vecdim;

            for (size_t d = 0; d < vecdim; ++d) {
                float val = base_ptr[d];
                // 量化前不需要 clamp，因为 scale 和 min_val 是基于 base 数据集计算的
                // 但量化函数内部会做 clamp 到 [0, 255]
                uint8_t q_val = quantize_value(val, d);
                q_base_ptr[d] = q_val;
                current_norm_sq += static_cast<uint32_t>(q_val) * static_cast<uint32_t>(q_val);
            }
            quantized_base_norms[i] = current_norm_sq;
        }
    }

    // 量化查询向量，并返回其 L2 范数平方 (Σ qQ[d]^2)
    // q_query: 输出参数，用于存储量化后的查询向量 (需要预先分配好内存, 大小为 vecdim)
    // 返回值: 量化查询向量的 L2 范数平方
    uint32_t quantize_query(const float* query, uint8_t* q_query) const {
        uint32_t query_norm_sq = 0;
        for (size_t d = 0; d < vecdim; ++d) {
            // 量化查询值时，需要将其 clamp 到基向量集的范围 [min_vals[d], max_vals[d]]
            // max_vals[d] = min_vals[d] + scales[d] * 255.0f
            float clamped_val = std::max(min_vals[d], 
                                    std::min(query[d], min_vals[d] + scales[d] * 255.0f));
            uint8_t q_val = quantize_value(clamped_val, d);
            q_query[d] = q_val;
            query_norm_sq += static_cast<uint32_t>(q_val) * static_cast<uint32_t>(q_val);
        }
        return query_norm_sq;
    }

    // 计算查询向量 q_query 与基向量 base_idx 之间的量化空间 L2 距离平方
    // D^2 = Σ(qQ[d] - qB[d])^2 = ΣqQ^2 + ΣqB^2 - 2 * Σ(qQ*qB)
    uint32_t compute_l2_sq_distance(const uint8_t* q_query, uint32_t q_query_norm_sq, size_t base_idx) const {
        uint32_t dot_product = quantized_dot_product(q_query, base_idx);
        uint32_t q_base_norm_sq = quantized_base_norms[base_idx];
        
        // 避免减法下溢（理论上 dot_product 不会超过范数之和的一半的两倍）
        // D^2 = q_query_norm_sq + q_base_norm_sq - 2 * dot_product
        // 由于都是uint32_t，需要确保 2 * dot_product 不会比 q_query_norm_sq + q_base_norm_sq 大太多
        // 但根据柯西不等式 (Σxy)^2 <= (Σx^2)(Σy^2)，dot_product 不会太大
        // D^2 理论上 >= 0
        uint64_t term1 = q_query_norm_sq;
        uint64_t term2 = q_base_norm_sq;
        uint64_t term3 = 2ULL * dot_product; // Use 64-bit intermediate to avoid overflow

        // D^2 = term1 + term2 - term3 
        // Cautious approach for potential underflow if using only uint32_t
        uint32_t distance_sq = 0;
        if (term1 + term2 >= term3) {
             distance_sq = static_cast<uint32_t>(term1 + term2 - term3);
        } else {
            // Should theoretically not happen if math is correct, but as safeguard
            distance_sq = 0; 
        }
        
        // 或者直接计算（如果确信不会溢出 uint32_t 且差值非负）
        // uint32_t distance_sq = q_query_norm_sq + q_base_norm_sq - 2 * dot_product;

        return distance_sq;
    }

    // 搜索 Top-K 最近邻 (基于量化空间的 L2^2 距离)
    // 返回的 pair 中 float 代表距离的平方 (uint32_t 转型而来)，值越小越近
    std::priority_queue<std::pair<float, uint32_t>> sq_search(
        const float* query, size_t k) const {
        
        // 使用最小堆（存储 <距离平方, 索引>），方便找到 Top-K 最近邻（距离最小）
        // C++ priority_queue 默认是最大堆，所以存储负距离或者自定义比较器
        // 这里用标准方式，存储 <距离, 索引>，让最大的距离在顶部，方便 pop
        std::priority_queue<std::pair<float, uint32_t>> top_k_heap;

        if (k == 0 || base_number == 0) {
            return top_k_heap; // Return empty heap if k=0 or no base vectors
        }

        // 1. 量化查询向量并计算其范数平方 (只需一次)
        std::vector<uint8_t> q_query(vecdim); // 在栈上或外部管理可能更好，但这里保持简单
        uint32_t q_query_norm_sq = quantize_query(query, q_query.data());

        // 2. 遍历所有基向量，计算距离并维护 Top-K 堆
        for (size_t i = 0; i < base_number; ++i) {
            uint32_t dist_sq = compute_l2_sq_distance(q_query.data(), q_query_norm_sq, i);
            float dist_sq_float = static_cast<float>(dist_sq); // 转为 float 以匹配 priority_queue

            if (top_k_heap.size() < k) {
                top_k_heap.push({dist_sq_float, static_cast<uint32_t>(i)});
            } else {
                // 如果当前距离小于堆顶的最大距离，则替换
                if (dist_sq_float < top_k_heap.top().first) {
                    top_k_heap.pop();
                    top_k_heap.push({dist_sq_float, static_cast<uint32_t>(i)});
                }
            }
        }

        return top_k_heap; // 返回包含 Top-K 结果的堆 (最大距离在顶部)
    }

    // 提供访问量化器参数的接口 (可选)
    size_t get_vecdim() const { return vecdim; }
    size_t get_base_number() const { return base_number; }
};

// ----- 使用示例 (重要：修改了使用方式) -----

/*
// 原来的 sq_search 函数是有问题的，因为它每次都重新构建索引
// 正确的使用方式是：

// 1. 构建索引 (一次性)
// float* base_vectors = ...; // 指向基向量数据的指针 (N * D)
// size_t n_base = ...;        // 基向量数量
// size_t dimension = ...;     // 向量维度
// ScalarQuantizer quantizer(base_vectors, n_base, dimension);

// 2. 执行搜索 (可多次调用)
// float* query_vector = ...; // 指向查询向量数据的指针 (D)
// size_t k = 10;             // 需要查找的近邻数量
// std::priority_queue<std::pair<float, uint32_t>> results = quantizer.search(query_vector, k);

// 3. 处理结果
// while (!results.empty()) {
//     float distance_sq = results.top().first;
//     uint32_t index = results.top().second;
//     // ... 处理结果 ...
//     results.pop();
// }
*/