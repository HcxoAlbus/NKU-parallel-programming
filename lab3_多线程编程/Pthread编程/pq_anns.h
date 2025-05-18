#pragma once

#include <vector>
#include <queue>
#include <random>
#include <algorithm>
#include <limits>
#include <stdexcept> // 用于异常处理
#include <cmath>    // 用于距离计算
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
    void kmeans_subspace(const float* train_data_sub, size_t ntrain, size_t sub_idx, size_t n_iter = 100) {
        if (ntrain == 0 || ksub == 0) return;

        float* sub_centroids = flat_centroids.data() + sub_idx * ksub * dsub;

        std::vector<size_t> assign(ntrain);       // 每个训练点的簇分配
        std::vector<float> min_dists_sq(ntrain, std::numeric_limits<float>::max()); // 初始化为最大值

        // --- 初始化: 随机选择 ksub 个点作为初始质心 ---
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

        // --- k-means 迭代 ---
        std::vector<float> new_centroids(ksub * dsub); // 存储新质心的临时空间
        std::vector<size_t> nassign(ksub);        // 分配给每个簇的点数
        std::vector<float> old_centroid_for_check(dsub); // 用于检查变化的临时空间

        bool converged = false;
        for (size_t iter = 0; iter < n_iter && !converged; ++iter) {
            // 1. 分配步骤
            #pragma omp parallel for // 并行化分配步骤
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
                min_dists_sq[i] = current_min_dist_sq; // 存储平方距离
            }

            // 2. 更新步骤
            std::fill(new_centroids.begin(), new_centroids.end(), 0.0f);
            std::fill(nassign.begin(), nassign.end(), 0);

            // 累加每个簇的点
            for (size_t i = 0; i < ntrain; ++i) {
                size_t cluster_idx = assign[i];
                nassign[cluster_idx]++;
                const float* point = train_data_sub + i * dsub;
                float* target_centroid_sum = new_centroids.data() + cluster_idx * dsub;

                // 优化累加 (如果 dsub 较大，NEON 可能有帮助)
                size_t d = 0;
                for (; d + 3 < dsub; d += 4) {
                    float32x4_t sum_vec = vld1q_f32(target_centroid_sum + d);
                    float32x4_t point_vec = vld1q_f32(point + d);
                    sum_vec = vaddq_f32(sum_vec, point_vec);
                    vst1q_f32(target_centroid_sum + d, sum_vec);
                }
                for (; d < dsub; ++d) { // 处理剩余部分
                    target_centroid_sum[d] += point[d];
                }
            }

            // 计算新的均值 (质心) 并检查收敛性
            bool iteration_changed = false; // 本轮迭代是否有质心变化
            size_t n_empty_clusters = 0;

            for (size_t k = 0; k < ksub; ++k) {
                float* current_centroid = sub_centroids + k * dsub;

                if (nassign[k] == 0) {
                    n_empty_clusters++;
                    // 保留旧质心，稍后处理空簇
                    continue;
                }

                // 修复收敛检查: 先保存旧质心
                std::copy(current_centroid, current_centroid + dsub, old_centroid_for_check.data());

                // 计算并更新质心
                float* sum_centroid = new_centroids.data() + k * dsub;
                float count_inv = 1.0f / static_cast<float>(nassign[k]);
                float32x4_t count_inv_vec = vdupq_n_f32(count_inv);

                size_t d = 0;
                for (; d + 3 < dsub; d += 4) {
                    float32x4_t sum_vec = vld1q_f32(sum_centroid + d);
                    float32x4_t new_val_vec = vmulq_f32(sum_vec, count_inv_vec);
                    vst1q_f32(current_centroid + d, new_val_vec); // 直接更新
                }
                for (; d < dsub; ++d) {
                    float new_val = sum_centroid[d] * count_inv;
                    current_centroid[d] = new_val; // 直接更新
                }

                // 修复收敛检查: 比较更新后的质心和旧质心
                for (size_t dim_idx = 0; dim_idx < dsub; ++dim_idx) {
                    if (std::abs(current_centroid[dim_idx] - old_centroid_for_check[dim_idx]) > 1e-6f) {
                        iteration_changed = true;
                        break; // 该质心已改变，无需继续比较
                    }
                }
            } 

            // --- 空簇处理 ---
            if (n_empty_clusters > 0 && ntrain > ksub) { // 确保有足够的点来处理
                 // 找到距离其当前质心最远的点
                 std::vector<std::pair<float, size_t>> dist_idx_pairs(ntrain);
                 for(size_t i = 0; i < ntrain; ++i) {
                     dist_idx_pairs[i] = {min_dists_sq[i], i}; // 存储 (距离平方, 点索引)
                 }
                 // 按距离降序排序，找到最远的点
                 std::sort(dist_idx_pairs.rbegin(), dist_idx_pairs.rend());

                 size_t fill_count = 0;
                 std::vector<bool> point_used(ntrain, false); // 跟踪点是否已被用于重新初始化

                 for (size_t k = 0; k < ksub && fill_count < n_empty_clusters; ++k) {
                     if (nassign[k] == 0) { // 这是一个空簇
                         // 尝试找到一个未被使用的最远点
                         size_t point_to_use_idx = ntrain; // 无效索引
                         for(size_t p_idx = 0; p_idx < ntrain; ++p_idx) {
                             size_t candidate_point_idx = dist_idx_pairs[p_idx].second;
                             if (!point_used[candidate_point_idx]) {
                                 point_to_use_idx = candidate_point_idx;
                                 point_used[candidate_point_idx] = true;
                                 break;
                             }
                         }

                         if (point_to_use_idx != ntrain) { // 找到了一个点
                             const float* src_point = train_data_sub + point_to_use_idx * dsub;
                             float* empty_centroid = sub_centroids + k * dsub;
                             std::copy(src_point, src_point + dsub, empty_centroid);
                             iteration_changed = true; // 重新初始化意味着质心改变了
                             fill_count++;
                         } else {
                             // 如果所有点都被用完了（理论上不应发生），或者找不到合适的点
                             // 可以选择填充 0 或保留上一次迭代的质心（如果需要）
                             // 这里我们填充 0，并认为这可能是一个变化
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
            } else if (n_empty_clusters > 0) {
                 // 如果 ntrain <= ksub，无法保证能找到足够的点重新初始化
                 // 可以选择保留空簇或填充0
                 for (size_t k = 0; k < ksub; ++k) {
                      if (nassign[k] == 0) {
                          // 保留或填充0
                          // float* empty_centroid = sub_centroids + k * dsub;
                          // std::fill(empty_centroid, empty_centroid + dsub, 0.0f);
                          // iteration_changed = true; // Consider if filling 0 is a change
                      }
                 }
            }


            // 检查是否收敛
            if (!iteration_changed && iter > 0) { // 如果本轮迭代没有任何质心改变
                converged = true;
            }
        } 
    }


    // 使用基础数据的一个子集 (如果 ntrain = nbase 则使用全部) 来训练 PQ 码本
    void train(const float* train_base, size_t ntrain) {
        flat_centroids.resize(nsub * ksub * dsub);

        // 使用 OpenMP 并行化子空间的训练过程
        #pragma omp parallel for schedule(dynamic)
        for (size_t sub = 0; sub < nsub; ++sub) {
            // 为每个线程创建独立的子空间数据副本 (注意内存开销)
            std::vector<float> subspace_train_data(ntrain * dsub);
            size_t d_start = sub * dsub;

            // 提取子空间数据
            // 可以考虑优化这里的内存拷贝，例如在 kmeans_subspace 中直接处理偏移量
            // 但需要小心缓存行伪共享等问题
            for (size_t i = 0; i < ntrain; ++i) {
                const float* src_full_vec = train_base + i * vecdim;
                float* dst_sub_vec = subspace_train_data.data() + i * dsub;
                // 使用 memcpy 可能比 std::copy 稍快
                std::memcpy(dst_sub_vec, src_full_vec + d_start, dsub * sizeof(float));
            }

            // 对该子空间运行 k-means
            kmeans_subspace(subspace_train_data.data(), ntrain, sub);
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
                uint8_t best_code = 0;

                // 在该子空间中查找最近的质心
                for (size_t k = 0; k < ksub; ++k) { // Use size_t for loop
                    const float* centroid = sub_centroids + k * dsub;
                    float dist_sq = compute_l2_sq_neon(sub_vec, centroid, dsub);

                    if (dist_sq < min_dist_sq) {
                        min_dist_sq = dist_sq;
                        // ksub 通常是 256，所以 k 的范围是 0-255，可以安全转换
                        if (k >= 256) {
                           // 理论上不应发生，但作为预防措施
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
        : vecdim(vecdim_),
          nsub(nsub_),
          dsub( (nsub_ == 0 || vecdim_ % nsub_ != 0) ? 0 : vecdim_ / nsub_ ),
          ksub(256),
          nbase(nbase_)
    {

        // --- 训练 ---
        size_t ntrain = static_cast<size_t>(static_cast<double>(nbase) * train_ratio);
        ntrain = std::max(static_cast<size_t>(ksub * 39), std::min(nbase, ntrain)); // FAISS 建议至少 ksub*39 个训练点
        
        if (ntrain > nbase) ntrain = nbase; // 不能超过基向量总数

        const float* train_data_ptr = base;
        std::vector<float> train_subset; // 用于存储训练数据的子集

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


        // --- 编码 ---
        std::cout << "PQ Encoding nbase=" << nbase << " vectors..." << std::endl;
         auto start_encode = std::chrono::high_resolution_clock::now();
        encode(base);
        auto end_encode = std::chrono::high_resolution_clock::now();
        auto duration_encode = std::chrono::duration_cast<std::chrono::milliseconds>(end_encode - start_encode);
        std::cout << "PQ Encoding finished in " << duration_encode.count() << " ms." << std::endl;
    }

    // 使用 ADC L2 距离搜索 k 个最近邻，并可选进行重排序
    // query: 查询向量 (vecdim)
    // base_data: 指向原始基向量数据的指针 (nbase * vecdim)，用于重排序
    // k: 最终需要的最近邻数量
    // rerank_k: PQ 搜索阶段返回的候选数量 (p)，用于重排序。如果 rerank_k <= k 或 base_data == nullptr，则不进行重排序。
    std::priority_queue<std::pair<float, uint32_t>> search(
        const float* query,
        const float* base_data, // 指向原始基向量数据的指针 (nbase * vecdim)，用于重排序
        size_t k,
        size_t rerank_k = 0    // PQ 搜索阶段返回的候选数量 (p)，用于重排序
    ) const {

        // 最小堆用于存储最终结果 <距离平方, 索引>
        std::priority_queue<std::pair<float, uint32_t>> final_results_heap;
        if (k == 0 || nbase == 0) return final_results_heap;
        if (query == nullptr) throw std::invalid_argument("Query vector pointer cannot be null.");

        // --- 阶段 1: PQ 近似搜索 ---
        // 确定 PQ 搜索阶段需要找多少个候选者
        size_t pq_search_k = (rerank_k > k && base_data != nullptr) ? rerank_k : k;
        if (pq_search_k > nbase) pq_search_k = nbase; // 不能超过基向量总数

        // 临时最大堆用于 PQ 搜索 <近似距离平方, 索引>
        std::priority_queue<std::pair<float, uint32_t>> pq_heap;

        // 1. 计算距离表: dist_table[sub][code] = || query_sub - centroid[sub][code] ||^2
        // 使用 alignas 或 posix_memalign 可能有助于 NEON 性能，但标准 vector 也可以
        std::vector<float> distance_table(nsub * ksub);

        // 并行化距离表计算 ( nsub 很大)
        // #pragma omp parallel for
        for (size_t sub = 0; sub < nsub; ++sub) {
            const float* query_sub_vec = query + sub * dsub;
            const float* sub_centroids = flat_centroids.data() + sub * ksub * dsub;
            float* table_sub_ptr = distance_table.data() + sub * ksub;

            for (size_t c = 0; c < ksub; ++c) {
                const float* centroid = sub_centroids + c * dsub;
                table_sub_ptr[c] = compute_l2_sq_neon(query_sub_vec, centroid, dsub);
            }
        }

        // 2. 计算所有基向量的近似距离并维护 PQ 堆
        for (size_t i = 0; i < nbase; ++i) {
            const uint8_t* code_ptr = flat_codes.data() + i * nsub;
            float approx_dist_sq = 0.0f;

            // 累加各子空间的距离
            // 可以考虑循环展开或 SIMD 优化这里的累加
            for (size_t sub = 0; sub < nsub; ++sub) {
                approx_dist_sq += distance_table[sub * ksub + code_ptr[sub]];
            }

            // 维护 top-pq_search_k 堆 (最大堆)
            if (pq_heap.size() < pq_search_k) {
                pq_heap.push({approx_dist_sq, static_cast<uint32_t>(i)});
            } else if (approx_dist_sq < pq_heap.top().first) {
                pq_heap.pop();
                pq_heap.push({approx_dist_sq, static_cast<uint32_t>(i)});
            }
        }

        // --- 阶段 2: 重排序 ---
        // 如果不需要重排序，直接返回 PQ 结果
        bool perform_reranking = (rerank_k > k && base_data != nullptr);

        if (!perform_reranking) {
            // 如果不重排序，直接返回 PQ 结果 (需要转换成最小堆形式，或者直接返回 pq_heap)
            // 为了统一返回类型，我们将 pq_heap (最大堆) 转换为 final_results_heap (最小堆)
             while (!pq_heap.empty()) {
                 final_results_heap.push(pq_heap.top()); // 结果顺序可能与预期相反，取决于如何使用
                 pq_heap.pop();
             }
             // 注意：上面的转换结果是按距离从大到小排列的。如果需要从小到大，需要反转或使用不同的结构。
             // 或者，如果 benchmark_search 处理最大堆没问题，可以直接返回 pq_heap。
             // 假设 benchmark_search 需要从小到大的结果（因为它用了 std::set），我们需要正确构建 final_results_heap。
             // 清空 final_results_heap 并重新构建
             std::vector<std::pair<float, uint32_t>> temp_results;
             while (!pq_heap.empty()) {
                 temp_results.push_back(pq_heap.top());
                 pq_heap.pop();
             }
             // 现在 temp_results 包含了 pq_search_k 个结果，距离最大的在前面
             // 我们只需要 k 个，并且需要按距离从小到大
             std::sort(temp_results.begin(), temp_results.end()); // 按距离升序排序
             // 取前 k 个放入最小堆（虽然直接用 vector 可能更简单）
             for(size_t i = 0; i < std::min((size_t)k, temp_results.size()); ++i) {
                 final_results_heap.push(temp_results[i]);
             }
             // 返回包含前 k 个近似结果的最小堆
             return final_results_heap;


        } else {
            // 执行重排序
            // 1. 从 pq_heap 中提取候选者索引
            std::vector<uint32_t> candidate_indices;
            candidate_indices.reserve(pq_heap.size());
            while (!pq_heap.empty()) {
                candidate_indices.push_back(pq_heap.top().second);
                pq_heap.pop();
            }
             // candidate_indices 现在包含了 pq_search_k 个候选者的索引

            // 2. 计算精确距离并维护最终的 top-k 最小堆
            for (uint32_t idx : candidate_indices) {
                const float* base_vec = base_data + static_cast<size_t>(idx) * vecdim; // 使用 static_cast
                float exact_dist_sq = compute_l2_sq_neon(query, base_vec, vecdim);

                if (final_results_heap.size() < k) {
                    final_results_heap.push({exact_dist_sq, idx});
                } else if (exact_dist_sq < final_results_heap.top().first) {
                    final_results_heap.pop();
                    final_results_heap.push({exact_dist_sq, idx});
                }
            }
            // 返回包含重排序后 top-k 结果的最小堆
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