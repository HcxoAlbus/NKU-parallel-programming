#pragma once
#include <queue>
#include <cstdint>

// 简化的内积距离计算（用于GPU代码兼容性）
inline float inner_product_distance_simd(const float* x, const float* y, size_t dim) {
    float dot = 0.0f;
    for (size_t i = 0; i < dim; ++i) {
        dot += x[i] * y[i];
    }
    return 1.0f - dot;
}

// 简化的ANNS搜索（用于兼容性）
std::priority_queue<std::pair<float, uint32_t>> simd_search(
    float* base, const float* query, size_t base_number, size_t vecdim, size_t k) {
    
    std::priority_queue<std::pair<float, uint32_t>> q;

    for (size_t i = 0; i < base_number; ++i) {
        float dis = inner_product_distance_simd(base + i * vecdim, query, vecdim);

        if (q.size() < k) {
            q.push({dis, static_cast<uint32_t>(i)});
        } else {
            if (dis < q.top().first) {
                q.push({dis, static_cast<uint32_t>(i)});
                q.pop();
            }
        }
    }
    return q;
}