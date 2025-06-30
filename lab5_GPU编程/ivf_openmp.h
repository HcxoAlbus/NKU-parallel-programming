#pragma once

#include <vector>
#include <queue>
#include <cstdint>
#include <algorithm>
#include <limits>
#include <cmath>
#include <numeric> // 用于 std::iota
#include <random>  // 用于 K-means++ 初始化
#include <chrono>  // 用于随机种子
#include <omp.h>   // 用于 OpenMP

#include "simd_anns.h" // 用于 inner_product_distance_simd

class IVFIndexOpenMP {
public:
    float* base_data_source_ptr;
    size_t num_base_vectors;
    size_t vector_dim;
    size_t num_target_clusters;
    int num_threads_to_use;

    std::vector<float> centroids_data; // 以扁平数组存储: num_clusters * dim
    std::vector<std::vector<uint32_t>> inverted_lists_data;

private:
    int kmeans_max_iter;

public:
    IVFIndexOpenMP(float* base_data, size_t n_base, size_t dim,
                   size_t n_clusters, int n_threads, int kmeans_iterations = 20)
        : base_data_source_ptr(base_data), num_base_vectors(n_base), vector_dim(dim),
          num_target_clusters(n_clusters), num_threads_to_use(std::max(1, n_threads)),
          kmeans_max_iter(kmeans_iterations) {

        if (num_base_vectors == 0 || vector_dim == 0 || num_target_clusters == 0) {
            std::cerr << "IVFIndexOpenMP: 无效参数 (num_base, dim, 或 num_clusters 为零)." << std::endl;
            return;
        }
        if (num_target_clusters > num_base_vectors) {
            std::cerr << "IVFIndexOpenMP: 警告, num_clusters > num_base. 设置 num_clusters = num_base." << std::endl;
            num_target_clusters = num_base_vectors;
        }
        
        centroids_data.resize(num_target_clusters * vector_dim);
        if (num_target_clusters > 0) {
            initialize_centroids_kmeans_plus_plus();
            run_kmeans_parallel();
            build_inverted_lists();
        }
    }

    ~IVFIndexOpenMP() = default;

    void initialize_centroids_kmeans_plus_plus() {
        if (num_base_vectors == 0 || num_target_clusters == 0) return;

        std::vector<float> min_dist_sq(num_base_vectors, std::numeric_limits<float>::max());
        std::vector<bool> chosen(num_base_vectors, false);
        std::mt19937 rng(std::chrono::system_clock::now().time_since_epoch().count());

        std::uniform_int_distribution<size_t> dist_idx(0, num_base_vectors - 1);
        size_t first_centroid_idx = dist_idx(rng);
        std::copy(base_data_source_ptr + first_centroid_idx * vector_dim,
                  base_data_source_ptr + (first_centroid_idx + 1) * vector_dim,
                  centroids_data.begin());
        chosen[first_centroid_idx] = true;
        min_dist_sq[first_centroid_idx] = 0.0f;

        int update_threads = std::max(1, (num_base_vectors == 0) ? 1 : std::min(num_threads_to_use, static_cast<int>(num_base_vectors)));
        
        #pragma omp parallel for num_threads(update_threads)
        for (size_t i = 0; i < num_base_vectors; ++i) {
            if (i == first_centroid_idx) continue;
            float d = inner_product_distance_simd(base_data_source_ptr + i * vector_dim, centroids_data.data(), vector_dim);
            min_dist_sq[i] = d;
        }

        for (size_t c_idx = 1; c_idx < num_target_clusters; ++c_idx) {
            std::vector<double> weights(num_base_vectors);
            double total_weight = 0.0;
            for (size_t i = 0; i < num_base_vectors; ++i) {
                if (chosen[i]) {
                    weights[i] = 0.0;
                } else {
                    weights[i] = static_cast<double>(min_dist_sq[i]);
                }
                total_weight += weights[i];
            }

            size_t next_centroid_base_idx = 0;
            if (total_weight == 0.0) { // 所有剩余点都相同或已被选择
                bool found_new = false;
                for (size_t i = 0; i < num_base_vectors; ++i) {
                    if (!chosen[i]) {
                        next_centroid_base_idx = i;
                        found_new = true;
                        break;
                    }
                }
                if (!found_new) { // 所有点都已选择，复制前一个
                     if (c_idx > 0) {
                        std::copy(centroids_data.data() + (c_idx-1)*vector_dim,
                                centroids_data.data() + c_idx*vector_dim,
                                centroids_data.data() + c_idx*vector_dim);
                     } else { // 如果 c_idx 从 1 开始，则不应发生
                        size_t rand_fallback_idx = dist_idx(rng);
                         std::copy(base_data_source_ptr + rand_fallback_idx * vector_dim,
                                  base_data_source_ptr + (rand_fallback_idx + 1) * vector_dim,
                                  centroids_data.data() + c_idx * vector_dim);
                        chosen[rand_fallback_idx] = true;
                     }
                     // 如果我们复制了或没有新点，则无需更新 min_dist_sq
                     continue;
                }
            } else {
                std::uniform_real_distribution<double> dist_prob_selection(0.0, total_weight);
                double rand_val = dist_prob_selection(rng);
                double current_sum = 0.0;
                for (size_t i = 0; i < num_base_vectors; ++i) {
                    current_sum += weights[i];
                    if (current_sum >= rand_val && !chosen[i]) {
                        next_centroid_base_idx = i;
                        break;
                    }
                }
                if (chosen[next_centroid_base_idx] || current_sum < rand_val) { // 回退机制
                     for(size_t i=0; i<num_base_vectors; ++i) if(!chosen[i]) {next_centroid_base_idx = i; break;}
                }
            }

            std::copy(base_data_source_ptr + next_centroid_base_idx * vector_dim,
                      base_data_source_ptr + (next_centroid_base_idx + 1) * vector_dim,
                      centroids_data.data() + c_idx * vector_dim);
            chosen[next_centroid_base_idx] = true;
            min_dist_sq[next_centroid_base_idx] = 0.0f;


            if (c_idx < num_target_clusters - 1) {
                const float* new_centroid_ptr = centroids_data.data() + c_idx * vector_dim;
                #pragma omp parallel for num_threads(update_threads)
                for (size_t i = 0; i < num_base_vectors; ++i) {
                    if (chosen[i]) continue;
                    float d = inner_product_distance_simd(base_data_source_ptr + i * vector_dim, new_centroid_ptr, vector_dim);
                    if (d < min_dist_sq[i]) {
                        min_dist_sq[i] = d;
                    }
                }
            }
        }
    }

    void run_kmeans_parallel() {
        if (num_target_clusters == 0) return;
        std::vector<int> assignments(num_base_vectors);

        int kmeans_threads = std::max(1, (num_base_vectors == 0) ? 1 : std::min(num_threads_to_use, static_cast<int>(num_base_vectors)));

        for (int iter = 0; iter < kmeans_max_iter; ++iter) {
            std::vector<float> global_new_centroids_sum(num_target_clusters * vector_dim, 0.0f);
            std::vector<int> global_new_centroids_count(num_target_clusters, 0);

            #pragma omp parallel num_threads(kmeans_threads)
            {
                std::vector<float> local_sums(num_target_clusters * vector_dim, 0.0f);
                std::vector<int> local_counts(num_target_clusters, 0);

                #pragma omp for schedule(static)
                for (size_t i = 0; i < num_base_vectors; ++i) {
                    const float* current_vector = base_data_source_ptr + i * vector_dim;
                    float min_dist = std::numeric_limits<float>::max();
                    int best_cluster_idx = 0;
                    for (size_t c = 0; c < num_target_clusters; ++c) {
                        const float* centroid_ptr = centroids_data.data() + c * vector_dim;
                        float dist = inner_product_distance_simd(current_vector, centroid_ptr, vector_dim);
                        if (dist < min_dist) {
                            min_dist = dist;
                            best_cluster_idx = static_cast<int>(c);
                        }
                    }
                    assignments[i] = best_cluster_idx;
                    local_counts[best_cluster_idx]++;
                    for (size_t d = 0; d < vector_dim; ++d) {
                        local_sums[best_cluster_idx * vector_dim + d] += current_vector[d];
                    }
                }

                #pragma omp critical
                {
                    for (size_t c = 0; c < num_target_clusters; ++c) {
                        if (local_counts[c] > 0) {
                            global_new_centroids_count[c] += local_counts[c];
                            for (size_t d = 0; d < vector_dim; ++d) {
                                global_new_centroids_sum[c * vector_dim + d] += local_sums[c * vector_dim + d];
                            }
                        }
                    }
                }
            } // 结束并行区域

            bool changed = false;
            for (size_t c = 0; c < num_target_clusters; ++c) {
                if (global_new_centroids_count[c] > 0) {
                    for (size_t d = 0; d < vector_dim; ++d) {
                        float new_val = global_new_centroids_sum[c * vector_dim + d] / global_new_centroids_count[c];
                        if (std::fabs(centroids_data[c * vector_dim + d] - new_val) > 1e-6) {
                            changed = true;
                        }
                        centroids_data[c * vector_dim + d] = new_val;
                    }
                } else { // 处理空簇
                    if (num_base_vectors > num_target_clusters) {
                        std::mt19937 rng_reinit(iter + c + omp_get_thread_num()); // 如果在并行中，添加 thread_num 以获得更多种子方差
                        std::uniform_int_distribution<size_t> dist_pt(0, num_base_vectors - 1);
                        size_t random_point_idx = dist_pt(rng_reinit);
                        
                        bool is_already_centroid_approx = false;
                        // 这个检查可以简化或使其更健壮
                        for(size_t cc=0; cc < num_target_clusters; ++cc) {
                            if (cc == c) continue;
                            float dist_to_other_centroid_sq = 0;
                            for(size_t dd=0; dd < vector_dim; ++dd) {
                                float diff = base_data_source_ptr[random_point_idx*vector_dim + dd] - centroids_data[cc*vector_dim + dd];
                                dist_to_other_centroid_sq += diff*diff;
                            }
                            if (dist_to_other_centroid_sq < 1e-12) { // 比较平方距离
                                is_already_centroid_approx = true;
                                break;
                            }
                        }

                        if(!is_already_centroid_approx){
                             std::copy(base_data_source_ptr + random_point_idx * vector_dim,
                                  base_data_source_ptr + (random_point_idx + 1) * vector_dim,
                                  centroids_data.data() + c * vector_dim);
                            changed = true;
                        }
                    }
                }
            }
            if (!changed && iter > 0) break;
        }
    }

    void build_inverted_lists() {
        if (num_target_clusters == 0) return;
        inverted_lists_data.assign(num_target_clusters, std::vector<uint32_t>());
        std::vector<int> assignments(num_base_vectors);

        int build_list_threads = std::max(1, (num_base_vectors == 0) ? 1 : std::min(num_threads_to_use, static_cast<int>(num_base_vectors)));

        #pragma omp parallel for num_threads(build_list_threads) schedule(static)
        for (size_t i = 0; i < num_base_vectors; ++i) {
            float min_dist = std::numeric_limits<float>::max();
            int best_cluster = 0;
            const float* current_vector = base_data_source_ptr + i * vector_dim;
            for (size_t c = 0; c < num_target_clusters; ++c) {
                const float* centroid_ptr = centroids_data.data() + c * vector_dim;
                float dist = inner_product_distance_simd(current_vector, centroid_ptr, vector_dim);
                if (dist < min_dist) {
                    min_dist = dist;
                    best_cluster = static_cast<int>(c);
                }
            }
            assignments[i] = best_cluster;
        }

        // 从 assignments 顺序填充倒排列表
        // 这部分通常不是瓶颈。并行化对向量的 push_back 很复杂。
        for (size_t i = 0; i < num_base_vectors; ++i) {
            if (assignments[i] >= 0 && static_cast<size_t>(assignments[i]) < num_target_clusters) {
                inverted_lists_data[assignments[i]].push_back(static_cast<uint32_t>(i));
            }
        }
    }

    std::priority_queue<std::pair<float, uint32_t>> search(
        const float* query, size_t k, size_t nprobe) {
        
        std::priority_queue<std::pair<float, uint32_t>> final_top_k;
        if (k == 0 || num_target_clusters == 0) return final_top_k;
        if (nprobe == 0) nprobe = 1;
        if (nprobe > num_target_clusters) nprobe = num_target_clusters;

        // 阶段 1: 找到 nprobe 个最近的质心
        std::vector<std::pair<float, int>> all_centroid_distances;
        all_centroid_distances.reserve(num_target_clusters);

        int stage1_threads = std::max(1, (num_target_clusters == 0) ? 1 : std::min(num_threads_to_use, static_cast<int>(num_target_clusters)));

        #pragma omp parallel num_threads(stage1_threads)
        {
            std::vector<std::pair<float, int>> local_centroid_distances;
            local_centroid_distances.reserve(num_target_clusters / stage1_threads + 1); // 预分配
            
            #pragma omp for schedule(static)
            for (size_t c_idx = 0; c_idx < num_target_clusters; ++c_idx) {
                const float* centroid_ptr = centroids_data.data() + c_idx * vector_dim;
                float dist = inner_product_distance_simd(query, centroid_ptr, vector_dim);
                local_centroid_distances.emplace_back(dist, static_cast<int>(c_idx));
            }

            #pragma omp critical
            {
                all_centroid_distances.insert(all_centroid_distances.end(),
                                              local_centroid_distances.begin(),
                                              local_centroid_distances.end());
            }
        }
        
        std::sort(all_centroid_distances.begin(), all_centroid_distances.end());

        std::vector<int> nprobe_cluster_indices;
        nprobe_cluster_indices.reserve(nprobe);
        for (size_t i = 0; i < std::min(nprobe, all_centroid_distances.size()); ++i) {
            nprobe_cluster_indices.push_back(all_centroid_distances[i].second);
        }

        if (nprobe_cluster_indices.empty()) {
            return final_top_k;
        }

        // 阶段 2: 在选定的 nprobe 个簇内搜索
        // 根据 nprobe_cluster_indices.size() 确定阶段 2 的有效线程数
        int stage2_threads = std::max(1, (nprobe_cluster_indices.empty()) ? 1 : std::min(num_threads_to_use, static_cast<int>(nprobe_cluster_indices.size())));
        std::vector<std::priority_queue<std::pair<float, uint32_t>>> per_thread_top_k_queues(stage2_threads);


        #pragma omp parallel num_threads(stage2_threads)
        {
            int tid = omp_get_thread_num();
            // 每个线程从向量中处理自己的优先队列
            // std::priority_queue<std::pair<float, uint32_t>> local_pq; // 这也是一个选项

            #pragma omp for schedule(dynamic) // 动态调度，因为列表大小可能不同
            for (size_t i = 0; i < nprobe_cluster_indices.size(); ++i) {
                int cluster_idx = nprobe_cluster_indices[i];
                if (cluster_idx < 0 || static_cast<size_t>(cluster_idx) >= inverted_lists_data.size()) continue;

                const auto& point_indices_in_cluster = inverted_lists_data[cluster_idx];
                for (uint32_t point_orig_idx : point_indices_in_cluster) {
                    const float* base_vector_ptr = base_data_source_ptr + point_orig_idx * vector_dim;
                    float dist = inner_product_distance_simd(query, base_vector_ptr, vector_dim);

                    if (per_thread_top_k_queues[tid].size() < k) {
                        per_thread_top_k_queues[tid].push({dist, point_orig_idx});
                    } else if (dist < per_thread_top_k_queues[tid].top().first) {
                        per_thread_top_k_queues[tid].pop();
                        per_thread_top_k_queues[tid].push({dist, point_orig_idx});
                    }
                }
            }
        } // 结束阶段 2 的并行区域

        // 合并来自 per_thread_top_k_queues 的结果
        for (int i = 0; i < stage2_threads; ++i) {
            while (!per_thread_top_k_queues[i].empty()) {
                std::pair<float, uint32_t> cand = per_thread_top_k_queues[i].top();
                per_thread_top_k_queues[i].pop();
                if (final_top_k.size() < k) {
                    final_top_k.push(cand);
                } else if (cand.first < final_top_k.top().first) {
                    final_top_k.pop();
                    final_top_k.push(cand);
                }
            }
        }
        return final_top_k;
    }
};
