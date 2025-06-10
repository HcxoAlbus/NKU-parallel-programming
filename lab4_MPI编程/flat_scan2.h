#pragma once

#include <vector>
#include <queue>
#include <cstdint>
#include "simd_anns.h"

inline std::priority_queue<std::pair<float, uint32_t>> flat_search(
    const float* base_data, const float* query, size_t num_base, size_t dim, size_t k) {
    
    std::priority_queue<std::pair<float, uint32_t>> top_k;
    
    for (size_t i = 0; i < num_base; ++i) {
        float dist = inner_product_distance_simd(query, base_data + i * dim, dim);
        
        if (top_k.size() < k) {
            top_k.push({dist, static_cast<uint32_t>(i)});
        } else if (dist < top_k.top().first) {
            top_k.pop();
            top_k.push({dist, static_cast<uint32_t>(i)});
        }
    }
    
    return top_k;
}