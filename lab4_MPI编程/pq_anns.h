#pragma once

#include <vector>
#include <queue>
#include <random>
#include <algorithm>
#include <limits>
#include <stdexcept>
#include <cmath>
#include <cstdint>
#include <arm_neon.h>
#include <omp.h>
#include <numeric>
#include <chrono>
#include <iostream>
#include <cstring>

// 辅助函数：使用 NEON 计算 L2 平方距离
inline float compute_l2_sq_neon(const float* a, const float* b, size_t dim) {
    float32x4_t sum_vec = vdupq_n_f32(0.0f);
    size_t d = 0;
    for (; d + 3 < dim; d += 4) {
        float32x4_t a_vec = vld1q_f32(a + d);
        float32x4_t b_vec = vld1q_f32(b + d);
        float32x4_t diff = vsubq_f32(a_vec, b_vec);
        float32x4_t diff_sq = vmulq_f32(diff, diff);
        sum_vec = vaddq_f32(sum_vec, diff_sq);
    }

    float32x2_t sum_pair = vpadd_f32(vget_low_f32(sum_vec), vget_high_f32(sum_vec));
    sum_pair = vpadd_f32(sum_pair, sum_pair);
    float total_sum = vget_lane_f32(sum_pair, 0);

    for (; d < dim; ++d) {
        float diff_scalar = a[d] - b[d];
        total_sum += diff_scalar * diff_scalar;
    }
    return total_sum;
}


class ProductQuantizer {
private:
    const size_t vecdim;
    const size_t nsub;
    const size_t dsub;
    const size_t ksub;
    const size_t nbase;

    std::vector<float> flat_centroids;
    std::vector<uint8_t> flat_codes;

    // ... 私有方法 (保持不变) ...
    void kmeans_subspace(const float* train_data_sub, size_t ntrain, size_t sub_idx, size_t n_iter = 100) {
        if (ntrain == 0 || ksub == 0) return;

        float* sub_centroids = flat_centroids.data() + sub_idx * ksub * dsub;

        std::vector<size_t> assign(ntrain);
        std::vector<float> min_dists_sq(ntrain, std::numeric_limits<float>::max());

        std::vector<size_t> initial_ids(ntrain);
        std::iota(initial_ids.begin(), initial_ids.end(), 0);
        std::random_device rd;
        std::mt19937 gen(rd());
        std::shuffle(initial_ids.begin(), initial_ids.end(), gen);

        size_t n_init_centroids = std::min(ntrain, ksub);
        for (size_t i = 0; i < n_init_centroids; ++i) {
            const float* src_vec = train_data_sub + initial_ids[i] * dsub;
            float* dst_centroid = sub_centroids + i * dsub;
            std::copy(src_vec, src_vec + dsub, dst_centroid);
        }
        for (size_t i = n_init_centroids; i < ksub; ++i) {
            float* dst_centroid = sub_centroids + i * dsub;
            std::fill(dst_centroid, dst_centroid + dsub, 0.0f);
        }

        std::vector<float> new_centroids(ksub * dsub);
        std::vector<size_t> nassign(ksub);
        std::vector<float> old_centroid_for_check(dsub);

        bool converged = false;
        for (size_t iter = 0; iter < n_iter && !converged; ++iter) {
            #pragma omp parallel for
            for (size_t i = 0; i < ntrain; ++i) {
                const float* point = train_data_sub + i * dsub;
                float current_min_dist_sq = std::numeric_limits<float>::max();
                size_t current_min_idx = 0;

                for (size_t k = 0; k < ksub; ++k) {
                    const float* centroid = sub_centroids + k * dsub;
                    float dist_sq = compute_l2_sq_neon(point, centroid, dsub);
                    if (dist_sq < current_min_dist_sq) {
                        current_min_dist_sq = dist_sq;
                        current_min_idx = k;
                    }
                }
                assign[i] = current_min_idx;
                min_dists_sq[i] = current_min_dist_sq;
            }

            std::fill(new_centroids.begin(), new_centroids.end(), 0.0f);
            std::fill(nassign.begin(), nassign.end(), 0);

            for (size_t i = 0; i < ntrain; ++i) {
                size_t cluster_idx = assign[i];
                nassign[cluster_idx]++;
                const float* point = train_data_sub + i * dsub;
                float* target_centroid_sum = new_centroids.data() + cluster_idx * dsub;
                size_t d = 0;
                for (; d + 3 < dsub; d += 4) {
                    float32x4_t sum_vec = vld1q_f32(target_centroid_sum + d);
                    float32x4_t point_vec = vld1q_f32(point + d);
                    sum_vec = vaddq_f32(sum_vec, point_vec);
                    vst1q_f32(target_centroid_sum + d, sum_vec);
                }
                for (; d < dsub; ++d) {
                    target_centroid_sum[d] += point[d];
                }
            }

            bool iteration_changed = false;
            size_t n_empty_clusters = 0;

            for (size_t k = 0; k < ksub; ++k) {
                float* current_centroid = sub_centroids + k * dsub;

                if (nassign[k] == 0) {
                    n_empty_clusters++;
                    continue;
                }
                std::copy(current_centroid, current_centroid + dsub, old_centroid_for_check.data());
                float* sum_centroid = new_centroids.data() + k * dsub;
                float count_inv = 1.0f / static_cast<float>(nassign[k]);
                float32x4_t count_inv_vec = vdupq_n_f32(count_inv);
                size_t d = 0;
                for (; d + 3 < dsub; d += 4) {
                    float32x4_t sum_vec = vld1q_f32(sum_centroid + d);
                    float32x4_t new_val_vec = vmulq_f32(sum_vec, count_inv_vec);
                    vst1q_f32(current_centroid + d, new_val_vec);
                }
                for (; d < dsub; ++d) {
                    current_centroid[d] = sum_centroid[d] * count_inv;
                }
                for (size_t dim_idx = 0; dim_idx < dsub; ++dim_idx) {
                    if (std::abs(current_centroid[dim_idx] - old_centroid_for_check[dim_idx]) > 1e-6f) {
                        iteration_changed = true;
                        break;
                    }
                }
            }

            if (n_empty_clusters > 0 && ntrain > ksub) {
                 std::vector<std::pair<float, size_t>> dist_idx_pairs(ntrain);
                 for(size_t i = 0; i < ntrain; ++i) {
                     dist_idx_pairs[i] = {min_dists_sq[i], i};
                 }
                 std::sort(dist_idx_pairs.rbegin(), dist_idx_pairs.rend());
                 size_t fill_count = 0;
                 std::vector<bool> point_used(ntrain, false);
                 for (size_t k = 0; k < ksub && fill_count < n_empty_clusters; ++k) {
                     if (nassign[k] == 0) {
                         size_t point_to_use_idx = ntrain;
                         for(size_t p_idx = 0; p_idx < ntrain; ++p_idx) {
                             size_t candidate_point_idx = dist_idx_pairs[p_idx].second;
                             if (!point_used[candidate_point_idx]) {
                                 point_to_use_idx = candidate_point_idx;
                                 point_used[candidate_point_idx] = true;
                                 break;
                             }
                         }
                         if (point_to_use_idx != ntrain) {
                             const float* src_point = train_data_sub + point_to_use_idx * dsub;
                             float* empty_centroid = sub_centroids + k * dsub;
                             std::copy(src_point, src_point + dsub, empty_centroid);
                             iteration_changed = true;
                             fill_count++;
                         } else {
                             float* empty_centroid = sub_centroids + k * dsub;
                             bool was_already_zero = true;
                             for(size_t d=0; d<dsub; ++d) { if (empty_centroid[d] != 0.0f) was_already_zero = false; }
                             if (!was_already_zero) {
                                 std::fill(empty_centroid, empty_centroid + dsub, 0.0f);
                                 iteration_changed = true;
                             }
                         }
                     }
                 }
            }
            if (!iteration_changed && iter > 0) {
                converged = true;
            }
        }
    }


    void train(const float* train_base, size_t ntrain) {
        flat_centroids.resize(nsub * ksub * dsub);
        #pragma omp parallel for schedule(dynamic)
        for (size_t sub = 0; sub < nsub; ++sub) {
            std::vector<float> subspace_train_data(ntrain * dsub);
            size_t d_start = sub * dsub;
            for (size_t i = 0; i < ntrain; ++i) {
                const float* src_full_vec = train_base + i * vecdim;
                float* dst_sub_vec = subspace_train_data.data() + i * dsub;
                std::memcpy(dst_sub_vec, src_full_vec + d_start, dsub * sizeof(float));
            }
            kmeans_subspace(subspace_train_data.data(), ntrain, sub);
        }
    }

    void encode(const float* base) {
        flat_codes.resize(nbase * nsub);
        #pragma omp parallel for schedule(dynamic)
        for (size_t i = 0; i < nbase; ++i) {
            const float* vec = base + i * vecdim;
            uint8_t* code_ptr = flat_codes.data() + i * nsub;
            for (size_t sub = 0; sub < nsub; ++sub) {
                const float* sub_vec = vec + sub * dsub;
                const float* sub_centroids = flat_centroids.data() + sub * ksub * dsub;
                float min_dist_sq = std::numeric_limits<float>::max();
                uint8_t best_code = 0;
                for (size_t k = 0; k < ksub; ++k) {
                    const float* centroid = sub_centroids + k * dsub;
                    float dist_sq = compute_l2_sq_neon(sub_vec, centroid, dsub);
                    if (dist_sq < min_dist_sq) {
                        min_dist_sq = dist_sq;
                        if (k >= 256) {
                           throw std::runtime_error("Centroid index k exceeds uint8_t range.");
                        }
                        best_code = static_cast<uint8_t>(k);
                    }
                }
                code_ptr[sub] = best_code;
            }
        }
    }

public:
    ProductQuantizer(const float* base, size_t nbase_, size_t vecdim_, size_t nsub_ = 8, double train_ratio = 1.0)
        : vecdim(vecdim_),
          nsub(nsub_),
          dsub( (nsub_ == 0 || vecdim_ % nsub_ != 0) ? 0 : vecdim_ / nsub_ ),
          ksub(256),
          nbase(nbase_)
    {
        if (dsub == 0 && vecdim > 0 && nsub > 0) {
             throw std::invalid_argument("ProductQuantizer: vecdim must be divisible by nsub.");
        }
        if (vecdim == 0 || nsub == 0) {
             throw std::invalid_argument("ProductQuantizer: vecdim and nsub must be greater than zero.");
        }

        size_t ntrain = static_cast<size_t>(static_cast<double>(nbase) * train_ratio);
        ntrain = std::max(static_cast<size_t>(ksub * 39), std::min(nbase, ntrain)); 
        
        if (ntrain > nbase) ntrain = nbase; 
        if (ntrain == 0 && nbase > 0) ntrain = nbase;
        if (ntrain == 0 && nbase == 0) {
            std::cout << "PQ Training: No base data to train on." << std::endl;
            return;
        }

        const float* train_data_ptr = base;
        std::vector<float> train_subset; 

        if (ntrain < nbase) {
            std::cout << "PQ Training using a subset of " << ntrain << " vectors..." << std::endl;
            train_subset.resize(ntrain * vecdim);
            std::vector<size_t> indices(nbase);
            std::iota(indices.begin(), indices.end(), 0);

            std::random_device rd;
            std::mt19937 gen(rd());
            std::shuffle(indices.begin(), indices.end(), gen);

            #pragma omp parallel for
            for(size_t i = 0; i < ntrain; ++i) {
                const float* src = base + indices[i] * vecdim;
                float* dst = train_subset.data() + i * vecdim;
                std::memcpy(dst, src, vecdim * sizeof(float));
            }
            train_data_ptr = train_subset.data();
        } else {
             std::cout << "PQ Training using all " << nbase << " base vectors..." << std::endl;
        }

        auto start_train = std::chrono::high_resolution_clock::now();
        train(train_data_ptr, ntrain);
        auto end_train = std::chrono::high_resolution_clock::now();
        auto duration_train = std::chrono::duration_cast<std::chrono::milliseconds>(end_train - start_train);
        std::cout << "PQ Training finished in " << duration_train.count() << " ms." << std::endl;

        if (nbase > 0) {
            std::cout << "PQ Encoding nbase=" << nbase << " vectors..." << std::endl;
            auto start_encode = std::chrono::high_resolution_clock::now();
            encode(base);
            auto end_encode = std::chrono::high_resolution_clock::now();
            auto duration_encode = std::chrono::duration_cast<std::chrono::milliseconds>(end_encode - start_encode);
            std::cout << "PQ Encoding finished in " << duration_encode.count() << " ms." << std::endl;
        } else {
            std::cout << "PQ Encoding: No base data to encode." << std::endl;
        }
    }

    // +++ 新增的公有方法：判断PQ是否已训练 +++
    bool is_trained() const {
        return !flat_centroids.empty();
    }

    // +++ 修正后的 encode_vector 方法 +++
    void encode_vector(const float* vec, uint8_t* code) const {
        // 使用 is_trained() 方法来检查
        if (!is_trained()) {
            // 如果PQ没有被训练，将code置为0并返回
            // This is important: ensures code is always initialized.
            if (code && nsub > 0) { // Check code is not null
                std::fill(code, code + nsub, 0);
            }
            return;
        }

        const float* sub_vec_ptr = vec;
        // 遍历每个子空间
        for (size_t m = 0; m < nsub; ++m) {
            float min_dist_sq = std::numeric_limits<float>::max();
            int best_centroid_idx = 0;

            // 获取当前子空间的码本（codebook）
            // 使用正确的成员变量: flat_centroids, ksub, dsub
            const float* current_codebook = flat_centroids.data() + m * ksub * dsub;

            // 在当前子空间的码本中寻找最近的质心
            for (size_t j = 0; j < ksub; ++j) {
                float dist_sq = compute_l2_sq_neon(sub_vec_ptr, current_codebook + j * dsub, dsub);
                if (dist_sq < min_dist_sq) {
                    min_dist_sq = dist_sq;
                    best_centroid_idx = static_cast<int>(j);
                }
            }
            // 将找到的最佳质心索引作为该子空间的编码
            code[m] = static_cast<uint8_t>(best_centroid_idx);
            
            // 移动指针到下一个子向量
            sub_vec_ptr += dsub;
        }
    }

    void compute_query_distance_table(const float* query, std::vector<float>& table_output) const {
        if (!is_trained()) {
            table_output.clear(); // Ensure table is empty if not trained
            return;
        }
        table_output.resize(nsub * ksub);
        for (size_t sub = 0; sub < nsub; ++sub) {
            const float* query_sub_vec = query + sub * dsub;
            const float* current_sub_centroids = flat_centroids.data() + sub * ksub * dsub;
            float* table_sub_ptr = table_output.data() + sub * ksub;

            for (size_t c = 0; c < ksub; ++c) {
                const float* centroid = current_sub_centroids + c * dsub;
                table_sub_ptr[c] = compute_l2_sq_neon(query_sub_vec, centroid, dsub);
            }
        }
    }

    float compute_asymmetric_distance_sq_with_table(const uint8_t* item_code, const std::vector<float>& query_dist_table) const {
        float approx_dist_sq = 0.0f;
        if (query_dist_table.size() != nsub * ksub) {
            return std::numeric_limits<float>::max();
        }
        for (size_t sub = 0; sub < nsub; ++sub) {
            approx_dist_sq += query_dist_table[sub * ksub + item_code[sub]];
        }
        return approx_dist_sq;
    }
    
    const uint8_t* get_code_for_item(uint32_t item_original_idx) const {
        if (item_original_idx >= nbase || flat_codes.empty()) {
            return nullptr;
        }
        return flat_codes.data() + static_cast<size_t>(item_original_idx) * nsub;
    }

    std::priority_queue<std::pair<float, uint32_t>> search(
        const float* query,
        const float* base_data, 
        size_t k,
        size_t rerank_k = 0    
    ) const {
        std::priority_queue<std::pair<float, uint32_t>> final_results_heap;
        if (k == 0 || !is_trained()) return final_results_heap;
        if (query == nullptr) throw std::invalid_argument("Query vector pointer cannot be null.");

        size_t pq_search_k = (rerank_k > k && base_data != nullptr) ? rerank_k : k;
        if (pq_search_k > nbase) pq_search_k = nbase;

        std::priority_queue<std::pair<float, uint32_t>> pq_heap;

        std::vector<float> current_query_distance_table;
        compute_query_distance_table(query, current_query_distance_table);
        if (current_query_distance_table.empty()) {
            return final_results_heap;
        }

        for (size_t i = 0; i < nbase; ++i) {
            const uint8_t* code_ptr = flat_codes.data() + i * nsub;
            float approx_dist_sq = compute_asymmetric_distance_sq_with_table(code_ptr, current_query_distance_table);

            if (pq_heap.size() < pq_search_k) {
                pq_heap.push({approx_dist_sq, static_cast<uint32_t>(i)});
            } else if (approx_dist_sq < pq_heap.top().first) {
                pq_heap.pop();
                pq_heap.push({approx_dist_sq, static_cast<uint32_t>(i)});
            }
        }

        bool perform_reranking = (rerank_k > k && base_data != nullptr);

        if (!perform_reranking) {
            return pq_heap;
        } else {
            std::vector<uint32_t> candidate_indices;
            candidate_indices.reserve(pq_heap.size());
            while (!pq_heap.empty()) {
                candidate_indices.push_back(pq_heap.top().second);
                pq_heap.pop();
            }

            for (uint32_t idx : candidate_indices) {
                const float* exact_vec = base_data + static_cast<size_t>(idx) * vecdim;
                float exact_dist_sq = compute_l2_sq_neon(query, exact_vec, vecdim);
                if (final_results_heap.size() < k) {
                    final_results_heap.push({exact_dist_sq, idx});
                } else if (exact_dist_sq < final_results_heap.top().first) {
                    final_results_heap.pop();
                    final_results_heap.push({exact_dist_sq, idx});
                }
            }

            return final_results_heap;
        }
    }

    // 访问器
    size_t get_vecdim() const { return vecdim; }
    size_t get_nsub() const { return nsub; }
    size_t get_dsub() const { return dsub; }
    size_t get_ksub() const { return ksub; }
    size_t get_nbase() const { return nbase; }
};