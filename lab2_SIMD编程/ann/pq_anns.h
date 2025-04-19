#pragma once

#include <vector>
#include <queue>
#include <random>
#include <algorithm>
#include <limits>
#include <stdexcept> // 用于异常处理
#include <cmath>    // 用于距离计算中的 std::sqrt (虽然我们使用平方距离)
#include <cstdint>
#include <arm_neon.h>
#include <omp.h>      // For potential OpenMP usage
#include <numeric>    // For std::iota

// 辅助函数：使用 NEON 计算 L2 平方距离 (保持不变)
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
    const size_t vecdim;           // 原始向量维度
    const size_t nsub;             // 子空间数量
    const size_t dsub;             // 每个子空间的维度 (vecdim / nsub)
    const size_t ksub;             // 每个子空间的聚类中心数量 (通常 256)
    const size_t nbase;            // 编码的基向量数量

    // 连续存储质心: [sub0_c0, ..., sub0_ck-1, sub1_c0, ...] (nsub * ksub * dsub)
    std::vector<float> flat_centroids;

    // 连续存储编码: [vec0_sub0, ..., vec0_subN-1, vec1_sub0, ...] (nbase * nsub)
    std::vector<uint8_t> flat_codes;

    // --- 私有辅助方法 ---

    // 对单个子空间进行 k-means 聚类
    // train_data_sub: 指向该子空间训练数据的指针 (ntrain * dsub)
    // ntrain: 训练向量的数量
    // sub_idx: 当前子空间的索引
    void kmeans_subspace(const float* train_data_sub, size_t ntrain, size_t sub_idx, size_t n_iter = 1000) {
        if (ntrain == 0 || ksub == 0) return; // 没有训练数据或没有簇

        float* sub_centroids = flat_centroids.data() + sub_idx * ksub * dsub;

        std::vector<size_t> assign(ntrain);       // 每个训练点的簇分配
        std::vector<float> min_dists_sq(ntrain);  // 每个点到其质心的最小 *平方* 距离

        // --- 初始化: 随机选择 ksub 个点作为初始质心 ---
        std::vector<size_t> initial_ids(ntrain);
        std::iota(initial_ids.begin(), initial_ids.end(), 0); // Fill with 0, 1, ..., ntrain-1
        std::random_device rd;
        std::mt19937 gen(rd());
        std::shuffle(initial_ids.begin(), initial_ids.end(), gen); // Shuffle indices

        // Pick the first ksub unique shuffled indices (or fewer if ntrain < ksub)
        size_t n_init_centroids = std::min(ntrain, ksub);
        for (size_t i = 0; i < n_init_centroids; ++i) {
            const float* src_vec = train_data_sub + initial_ids[i] * dsub;
            float* dst_centroid = sub_centroids + i * dsub;
            std::copy(src_vec, src_vec + dsub, dst_centroid);
        }
        // If ntrain < ksub, fill remaining centroids with zeros (or another strategy)
        for (size_t i = n_init_centroids; i < ksub; ++i) {
            float* dst_centroid = sub_centroids + i * dsub;
            std::fill(dst_centroid, dst_centroid + dsub, 0.0f);
        }

        // --- k-means 迭代 ---
        std::vector<float> new_centroids(ksub * dsub); // 存储新质心的临时空间
        std::vector<size_t> nassign(ksub);        // 分配给每个簇的点数
        bool changed = true; // Flag to track centroid changes (optional for early stop)

        for (size_t iter = 0; iter < n_iter && changed; ++iter) {
            // 1. 分配步骤
            #pragma omp parallel for // Parallelize assignment if useful
            for (size_t i = 0; i < ntrain; ++i) {
                const float* point = train_data_sub + i * dsub;
                float min_dist_sq = std::numeric_limits<float>::max();
                size_t min_idx = 0;

                for (size_t k = 0; k < ksub; ++k) {
                    const float* centroid = sub_centroids + k * dsub;
                    float dist_sq = compute_l2_sq_neon(point, centroid, dsub);
                    if (dist_sq < min_dist_sq) {
                        min_dist_sq = dist_sq;
                        min_idx = k;
                    }
                }
                assign[i] = min_idx;
                min_dists_sq[i] = min_dist_sq; // Store squared distance
            }

            // 2. 更新步骤
            std::fill(new_centroids.begin(), new_centroids.end(), 0.0f);
            std::fill(nassign.begin(), nassign.end(), 0);

            // Accumulate points per cluster
            for (size_t i = 0; i < ntrain; ++i) {
                size_t cluster_idx = assign[i];
                nassign[cluster_idx]++;
                const float* point = train_data_sub + i * dsub;
                float* target_centroid_sum = new_centroids.data() + cluster_idx * dsub;

                // Optimized accumulation (using NEON if beneficial for dsub)
                size_t d = 0;
                for (; d + 3 < dsub; d += 4) {
                    float32x4_t sum_vec = vld1q_f32(target_centroid_sum + d);
                    float32x4_t point_vec = vld1q_f32(point + d);
                    sum_vec = vaddq_f32(sum_vec, point_vec);
                    vst1q_f32(target_centroid_sum + d, sum_vec);
                }
                for (; d < dsub; ++d) { // Handle remainder
                    target_centroid_sum[d] += point[d];
                }
            }

            // 计算新的均值 (质心) 并处理空簇
            // changed = false; // Reset change flag for this iteration - Keep this line

            size_t n_empty_clusters = 0;

            for (size_t k = 0; k < ksub; ++k) {
                float* current_centroid = sub_centroids + k * dsub;

                if (nassign[k] == 0) {
                    n_empty_clusters++;
                    continue; // Or handle empty cluster
                }

                float* sum_centroid = new_centroids.data() + k * dsub;
                float count_inv = 1.0f / static_cast<float>(nassign[k]);
                float32x4_t count_inv_vec = vdupq_n_f32(count_inv);

                size_t d = 0;
                // NEON Part: Just calculate and store. REMOVE the change check here.
                for (; d + 3 < dsub; d += 4) {
                    float32x4_t sum_vec = vld1q_f32(sum_centroid + d);
                    float32x4_t new_val_vec = vmulq_f32(sum_vec, count_inv_vec);
                    // float32x4_t old_val_vec = vld1q_f32(current_centroid + d); // No longer needed for comparison here

                    // REMOVED the for(lane...) loop and check for 'changed'

                    vst1q_f32(current_centroid + d, new_val_vec); // Update centroid
                }

                // Scalar Part: Handles remainder AND correctly checks for changes.
                // The comparison here is against the value *before* the update in this loop iteration.
                // This is sufficient to detect changes overall for the centroid k.
                for (; d < dsub; ++d) {
                     float new_val = sum_centroid[d] * count_inv;
                     if (std::abs(new_val - current_centroid[d]) > 1e-6f) { // Compare BEFORE update
                         changed = true; // Set flag if scalar part changes
                     }
                     current_centroid[d] = new_val; // Update centroid
                 }
            } // End loop for k (clusters)

            // --- 空簇处理 (如之前实现) ---
            if (n_empty_clusters > 0 && ntrain > 0) {
                // ... (空簇处理逻辑保持不变, 它也可能设置 changed = true) ...
                 std::vector<size_t> point_indices(ntrain);
                 std::iota(point_indices.begin(), point_indices.end(), 0);
                 std::vector<size_t> shuffled_point_indices = point_indices;
                 std::random_device rd_reinit; // Use a different rd or seed if needed
                 std::mt19937 gen_reinit(rd_reinit());
                 std::shuffle(shuffled_point_indices.begin(), shuffled_point_indices.end(), gen_reinit); // Re-shuffle needed

                 size_t fill_count = 0;
                 for (size_t k = 0; k < ksub && fill_count < n_empty_clusters; ++k) {
                     if (nassign[k] == 0) {
                         if (fill_count < shuffled_point_indices.size()) {
                             size_t point_idx_to_use = shuffled_point_indices[fill_count];
                             const float* src_point = train_data_sub + point_idx_to_use * dsub;
                             float* empty_centroid = sub_centroids + k * dsub;
                             std::copy(src_point, src_point + dsub, empty_centroid);
                             changed = true; // Re-initializing means centroids changed
                             fill_count++;
                         } else {
                             float* empty_centroid = sub_centroids + k * dsub;
                             std::fill(empty_centroid, empty_centroid + dsub, 0.0f);
                         }
                     }
                 }
            }

             // Optional convergence check: if (!changed && iter > 0) break;
        } // end k-means iterations
    }


    // 使用基础数据的一个子集 (如果 ntrain = nbase 则使用全部) 来训练 PQ 码本
    // train_base: 指向训练数据的指针 (ntrain * vecdim)
    // ntrain: 训练向量的数量
    void train(const float* train_base, size_t ntrain) {
        flat_centroids.resize(nsub * ksub * dsub);

        // 使用 OpenMP 并行化子空间的训练过程
        #pragma omp parallel for schedule(dynamic)
        for (size_t sub = 0; sub < nsub; ++sub) {
            // 为每个线程创建独立的子空间数据副本，避免数据竞争
            // 注意：这会增加内存使用量，如果 ntrain 很大，可能需要优化
            // 优化方案：直接传递指向 train_base 中对应子空间数据的指针或偏移量，
            //          并在 kmeans_subspace 中处理非连续访问（可能降低缓存效率）
            //          或者使用 #pragma omp critical 进行数据提取（可能引入瓶颈）
            // 当前实现：为每个线程创建副本（简单但可能耗内存）
            std::vector<float> subspace_train_data(ntrain * dsub);
            size_t d_start = sub * dsub;

            for (size_t i = 0; i < ntrain; ++i) {
                const float* src_full_vec = train_base + i * vecdim;
                float* dst_sub_vec = subspace_train_data.data() + i * dsub;
                std::copy(src_full_vec + d_start, src_full_vec + d_start + dsub, dst_sub_vec);
            }

            // 对该子空间运行 k-means
            kmeans_subspace(subspace_train_data.data(), ntrain, sub);
            // 注意：如果 kmeans_subspace 内部也用了 OMP parallel for，需要考虑嵌套并行
            //       当前 kmeans_subspace 的 parallel for 只在 assignment 上，可能还好
        }
    }

    // 使用训练好的码本对整个基础数据集进行编码
    void encode(const float* base) {
        flat_codes.resize(nbase * nsub);

        #pragma omp parallel for schedule(dynamic) // 并行化编码过程
        for (size_t i = 0; i < nbase; ++i) {
            const float* vec = base + i * vecdim;
            uint8_t* code_ptr = flat_codes.data() + i * nsub;

            for (size_t sub = 0; sub < nsub; ++sub) {
                const float* sub_vec = vec + sub * dsub;
                const float* sub_centroids = flat_centroids.data() + sub * ksub * dsub;
                float min_dist_sq = std::numeric_limits<float>::max();
                uint8_t best_code = 0; // Correct type

                // 在该子空间中查找最近的质心
                // *** 修复: 使用 size_t 作为循环变量 ***
                for (size_t k = 0; k < ksub; ++k) {
                    const float* centroid = sub_centroids + k * dsub;
                    float dist_sq = compute_l2_sq_neon(sub_vec, centroid, dsub);

                    if (dist_sq < min_dist_sq) {
                        min_dist_sq = dist_sq;
                        // *** 确保 k 在 uint8_t 范围内 ***
                        // ksub 通常是 256，所以 k 的范围是 0-255，可以安全转换
                        if (k >= 256) {
                           // This should not happen if ksub <= 256, but as a safeguard
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
    // 构造函数: 训练并编码数据
    ProductQuantizer(const float* base, size_t nbase_, size_t vecdim_, size_t nsub_ = 8, double train_ratio = 1.0)
        : vecdim(vecdim_),              // Initialize const member
          nsub(nsub_),                  // Initialize const member
          // Initialize dsub after basic checks on inputs nsub_ and vecdim_
          dsub( (nsub_ == 0 || vecdim_ % nsub_ != 0) ?
                  // Assign a dummy value like 0 if invalid, validation will throw later
                  // Or throw immediately if preferred, but cleaner to validate below
                  0
                : vecdim_ / nsub_ ),
          ksub(256),                  // Initialize const member
          nbase(nbase_)               // Initialize const member
    {
        // --- 参数验证 (using original inputs and initialized members) ---
        if (base == nullptr) {
            throw std::invalid_argument("Base data pointer cannot be null.");
        }
        // Check original inputs for fundamental issues first
        if (vecdim_ == 0 || nbase_ == 0 || nsub_ == 0) {
            throw std::invalid_argument("Vector dim, base number, and subspace number must be positive.");
        }
        // Check divisibility using original inputs (determines dsub's validity)
        if (vecdim_ % nsub_ != 0) {
            throw std::invalid_argument("Vector dim must be divisible by the number of subspaces.");
        }
        // Now dsub calculated in the initializer list should be correct and non-zero.
        // We can proceed with other checks.

        if (ksub > 256) {
             throw std::invalid_argument("ksub > 256 not supported with uint8_t codes.");
        }
        if (train_ratio <= 0.0 || train_ratio > 1.0) {
            throw std::invalid_argument("Training ratio must be in (0.0, 1.0].");
        }


        // --- 训练 ---
        size_t ntrain = static_cast<size_t>(static_cast<double>(nbase) * train_ratio);
        ntrain = std::max(static_cast<size_t>(ksub), std::min(nbase, ntrain));
        if (ntrain < ksub) {
             std::cerr << "Warning: Number of training points (" << ntrain
                       << ") is less than ksub (" << ksub
                       << "). K-means might be unstable." << std::endl;
        }

        const float* train_data_ptr = base;
        std::vector<float> train_subset; // RAII

        if (ntrain < nbase) {
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
        }

        std::cout << "PQ Training with ntrain=" << ntrain << " vectors..." << std::endl;
        auto start_train = std::chrono::high_resolution_clock::now();
        train(train_data_ptr, ntrain);
        auto end_train = std::chrono::high_resolution_clock::now();
        auto duration_train = std::chrono::duration_cast<std::chrono::milliseconds>(end_train - start_train);
        std::cout << "PQ Training finished in " << duration_train.count() << " ms." << std::endl;


        // --- 编码 ---
        std::cout << "PQ Encoding nbase=" << nbase << " vectors..." << std::endl;
         auto start_encode = std::chrono::high_resolution_clock::now();
        encode(base);
        auto end_encode = std::chrono::high_resolution_clock::now();
        auto duration_encode = std::chrono::duration_cast<std::chrono::milliseconds>(end_encode - start_encode);
        std::cout << "PQ Encoding finished in " << duration_encode.count() << " ms." << std::endl;
    }

    // 使用 ADC L2 距离搜索 k 个最近邻 (保持不变)
    std::priority_queue<std::pair<float, uint32_t>> search(
        const float* query, size_t k) const {

        std::priority_queue<std::pair<float, uint32_t>> result_heap;
        if (k == 0 || nbase == 0) return result_heap;
        if (query == nullptr) throw std::invalid_argument("Query vector pointer cannot be null.");

        // 1. 计算距离表: dist_table[sub][code] = || query_sub - centroid[sub][code] ||^2
        // 使用 alignas 或 posix_memalign 确保对齐可能有助于 NEON 性能
        // std::vector<float, AlignedAllocator<float, 32>> distance_table(nsub * ksub); // Requires custom allocator
        std::vector<float> distance_table(nsub * ksub); // Standard vector

        // Parallelize distance table computation if beneficial (many subspaces)
        // #pragma omp parallel for // Be careful with nested parallelism if search is also parallelized externally
        for (size_t sub = 0; sub < nsub; ++sub) {
            const float* query_sub_vec = query + sub * dsub;
            const float* sub_centroids = flat_centroids.data() + sub * ksub * dsub;
            float* table_sub_ptr = distance_table.data() + sub * ksub;

            for (size_t c = 0; c < ksub; ++c) { // Use size_t here too
                const float* centroid = sub_centroids + c * dsub;
                table_sub_ptr[c] = compute_l2_sq_neon(query_sub_vec, centroid, dsub);
            }
        }

        // 2. 计算所有基向量的近似距离
        // Could potentially parallelize this outer loop if nbase is very large and k is small
        // #pragma omp parallel // Requires careful handling of the shared result_heap (e.g., reduction or thread-local heaps)
        for (size_t i = 0; i < nbase; ++i) {
            const uint8_t* code_ptr = flat_codes.data() + i * nsub;
            float approx_dist_sq = 0.0f;

            // Unroll loop slightly? (May or may not help depending on compiler)
            size_t sub = 0;
            for (; sub + 3 < nsub; sub += 4) {
                 approx_dist_sq += distance_table[sub * ksub + code_ptr[sub]];
                 approx_dist_sq += distance_table[(sub+1) * ksub + code_ptr[sub+1]];
                 approx_dist_sq += distance_table[(sub+2) * ksub + code_ptr[sub+2]];
                 approx_dist_sq += distance_table[(sub+3) * ksub + code_ptr[sub+3]];
            }
            for (; sub < nsub; ++sub) { // Handle remainder
                approx_dist_sq += distance_table[sub * ksub + code_ptr[sub]];
            }

            // 3. 维护 top-k 堆 (This part needs protection if the outer loop is parallelized)
            if (result_heap.size() < k) {
                result_heap.push({approx_dist_sq, static_cast<uint32_t>(i)});
            } else if (approx_dist_sq < result_heap.top().first) {
                result_heap.pop();
                result_heap.push({approx_dist_sq, static_cast<uint32_t>(i)});
            }
        }

        return result_heap;
    }

    // 访问器 (保持不变)
    size_t get_vecdim() const { return vecdim; }
    size_t get_nsub() const { return nsub; }
    size_t get_dsub() const { return dsub; }
    size_t get_ksub() const { return ksub; }
    size_t get_nbase() const { return nbase; }
};