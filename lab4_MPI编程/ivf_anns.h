#pragma once

#include <vector>
#include <queue>
#include <cstdint>
#include <algorithm>
#include <limits>
#include <cmath>
#include <pthread.h>
#include <numeric> // 用于 std::iota
#include <random>  // 用于 K-means++ 初始化
#include <map>     // 用于在搜索中合并结果
#include <chrono>  // 用于随机种子

#include "simd_anns.h" // 用于 inner_product_distance_simd

// 前向声明
class IVFIndex;

// --- Pthread 数据结构 ---

struct KMeansAssignArgs {
    IVFIndex* ivf_instance;
    size_t start_idx; // 要处理的 base_data 的起始索引
    size_t end_idx;   // 要处理的 base_data 的结束索引
    const std::vector<float>* current_centroids_ptr;
    std::vector<int>* assignments_output_ptr; // assignments[base_data_idx] = cluster_idx
    // 用于质心更新的部分和 (每个线程一个)
    std::vector<float>* local_sum_vectors_ptr; // 扁平数组: num_clusters * dim
    std::vector<int>* local_counts_ptr;        // 扁平数组: num_clusters
};

struct SearchCentroidsArgs {
    IVFIndex* ivf_instance;
    const float* query_ptr;
    size_t start_centroid_idx;
    size_t end_centroid_idx;
    std::vector<std::pair<float, int>>* thread_centroid_distances_output_ptr;
};

struct SearchListsArgs {
    IVFIndex* ivf_instance;
    const float* query_ptr;
    size_t k_neighbors;
    const std::vector<int>* candidate_cluster_indices_ptr; // 所有 nprobe 簇索引
    size_t task_start_idx_in_candidates; // candidate_cluster_indices_ptr 中的任务起始索引
    size_t task_end_idx_in_candidates;   // candidate_cluster_indices_ptr 中的任务结束索引
    std::priority_queue<std::pair<float, uint32_t>>* thread_top_k_output_ptr;
};


class IVFIndex {
public:
    float* base_data_source_ptr; // 指向原始基准数据的指针
    size_t num_base_vectors;
    size_t vector_dim;
    size_t num_target_clusters;
    int num_threads_to_use;

    std::vector<float> centroids_data; // 以扁平数组存储: num_clusters * dim
    std::vector<std::vector<uint32_t>> inverted_lists_data; // 存储 base_data_source_ptr 的原始索引

    IVFIndex(float* base_data, size_t n_base, size_t dim,
             size_t n_clusters, int n_threads, int kmeans_iterations = 20)
        : base_data_source_ptr(base_data), num_base_vectors(n_base), vector_dim(dim),
          num_target_clusters(n_clusters), num_threads_to_use(std::max(1, n_threads)),
          kmeans_max_iter(kmeans_iterations) {
        
        if (num_base_vectors == 0 || vector_dim == 0 || num_target_clusters == 0) {
            // 或者抛出异常
            std::cerr << "IVFIndex: 无效参数 (num_base, dim, 或 num_clusters 为零)." << std::endl;
            return;
        }
        if (num_target_clusters > num_base_vectors) {
            std::cerr << "IVFIndex: 警告, num_clusters > num_base. 设置 num_clusters = num_base." << std::endl;
            num_target_clusters = num_base_vectors;
        }

        centroids_data.resize(num_target_clusters * vector_dim);
        if (num_target_clusters > 0) { // 仅当有簇要形成时才初始化
            initialize_centroids_kmeans_plus_plus();
            run_kmeans_parallel();
            build_inverted_lists();
        }
    }

    ~IVFIndex() = default;

    std::priority_queue<std::pair<float, uint32_t>> search(
        const float* query, size_t k, size_t nprobe) {
        
        std::priority_queue<std::pair<float, uint32_t>> final_top_k;
        if (k == 0 || num_target_clusters == 0) return final_top_k; // 没有簇或 k=0
        if (nprobe == 0) nprobe = 1; // 默认至少搜索一个簇
        if (nprobe > num_target_clusters) nprobe = num_target_clusters;

        // 阶段 1: 找到 nprobe 个最近的质心
        std::vector<std::pair<float, int>> all_centroid_distances; // pair: <distance, cluster_idx>
        all_centroid_distances.reserve(num_target_clusters);

        std::vector<pthread_t> threads_stage1(num_threads_to_use -1);
        std::vector<SearchCentroidsArgs> args_stage1(num_threads_to_use);
        std::vector<std::vector<std::pair<float, int>>> per_thread_centroid_distances(num_threads_to_use);

        size_t items_per_thread_s1 = num_target_clusters / num_threads_to_use;
        size_t remainder_s1 = num_target_clusters % num_threads_to_use;
        size_t current_start_idx_s1 = 0;

        for (int i = 0; i < num_threads_to_use; ++i) {
            size_t current_chunk_size = items_per_thread_s1 + (i < static_cast<int>(remainder_s1) ? 1 : 0);
            if (current_chunk_size == 0 && current_start_idx_s1 < num_target_clusters) { // 如果 num_threads > num_target_clusters，确保有进展
                 current_chunk_size = 1;
            }
            if (current_start_idx_s1 >= num_target_clusters) { // 没有更多工作
                args_stage1[i].start_centroid_idx = num_target_clusters; // 标记为无工作
                args_stage1[i].end_centroid_idx = num_target_clusters;
            } else {
                args_stage1[i].ivf_instance = this;
                args_stage1[i].query_ptr = query;
                args_stage1[i].start_centroid_idx = current_start_idx_s1;
                args_stage1[i].end_centroid_idx = std::min(current_start_idx_s1 + current_chunk_size, num_target_clusters);
                args_stage1[i].thread_centroid_distances_output_ptr = &per_thread_centroid_distances[i];
            }
            
            if (i < num_threads_to_use - 1 && args_stage1[i].start_centroid_idx < args_stage1[i].end_centroid_idx) {
                pthread_create(&threads_stage1[i], nullptr, search_centroids_worker_static, &args_stage1[i]);
            } else if (args_stage1[i].start_centroid_idx < args_stage1[i].end_centroid_idx) { // 如果有工作，主线程执行其部分
                search_centroids_worker_static(&args_stage1[i]);
            }
            current_start_idx_s1 += current_chunk_size;
        }

        for (int i = 0; i < num_threads_to_use - 1; ++i) {
            if (args_stage1[i].start_centroid_idx < args_stage1[i].end_centroid_idx) { // 仅当为工作创建了线程时才 join
                 pthread_join(threads_stage1[i], nullptr);
            }
        }

        for (int i = 0; i < num_threads_to_use; ++i) {
            all_centroid_distances.insert(all_centroid_distances.end(),
                                          per_thread_centroid_distances[i].begin(),
                                          per_thread_centroid_distances[i].end());
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
        std::vector<pthread_t> threads_stage2(num_threads_to_use - 1);
        std::vector<SearchListsArgs> args_stage2(num_threads_to_use);
        std::vector<std::priority_queue<std::pair<float, uint32_t>>> per_thread_top_k(num_threads_to_use);

        size_t num_candidate_clusters_to_search = nprobe_cluster_indices.size();
        size_t clusters_per_thread_s2 = num_candidate_clusters_to_search / num_threads_to_use;
        size_t remainder_s2 = num_candidate_clusters_to_search % num_threads_to_use;
        size_t current_cluster_offset_s2 = 0;

        for (int i = 0; i < num_threads_to_use; ++i) {
            size_t current_chunk_size = clusters_per_thread_s2 + (i < static_cast<int>(remainder_s2) ? 1 : 0);
             if (current_chunk_size == 0 && current_cluster_offset_s2 < num_candidate_clusters_to_search) {
                current_chunk_size = 1;
            }
            if(current_cluster_offset_s2 >= num_candidate_clusters_to_search) {
                args_stage2[i].task_start_idx_in_candidates = num_candidate_clusters_to_search;
                args_stage2[i].task_end_idx_in_candidates = num_candidate_clusters_to_search;
            } else {
                args_stage2[i].ivf_instance = this;
                args_stage2[i].query_ptr = query;
                args_stage2[i].k_neighbors = k;
                args_stage2[i].candidate_cluster_indices_ptr = &nprobe_cluster_indices;
                args_stage2[i].task_start_idx_in_candidates = current_cluster_offset_s2;
                args_stage2[i].task_end_idx_in_candidates = std::min(current_cluster_offset_s2 + current_chunk_size, num_candidate_clusters_to_search);
                args_stage2[i].thread_top_k_output_ptr = &per_thread_top_k[i];
            }

            if (i < num_threads_to_use - 1 && args_stage2[i].task_start_idx_in_candidates < args_stage2[i].task_end_idx_in_candidates) {
                pthread_create(&threads_stage2[i], nullptr, search_lists_worker_static, &args_stage2[i]);
            } else if (args_stage2[i].task_start_idx_in_candidates < args_stage2[i].task_end_idx_in_candidates) {
                search_lists_worker_static(&args_stage2[i]);
            }
            current_cluster_offset_s2 += current_chunk_size;
        }

        for (int i = 0; i < num_threads_to_use - 1; ++i) {
            if (args_stage2[i].task_start_idx_in_candidates < args_stage2[i].task_end_idx_in_candidates) {
                 pthread_join(threads_stage2[i], nullptr);
            }
        }

        // 合并所有线程的结果
        for (int i = 0; i < num_threads_to_use; ++i) {
            while (!per_thread_top_k[i].empty()) {
                std::pair<float, uint32_t> cand = per_thread_top_k[i].top();
                per_thread_top_k[i].pop();
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


private:
    int kmeans_max_iter;

    void initialize_centroids_kmeans_plus_plus() {
        if (num_base_vectors == 0 || num_target_clusters == 0) return;

        std::vector<float> min_dist_sq(num_base_vectors, std::numeric_limits<float>::max());
        std::vector<bool> chosen(num_base_vectors, false);
        std::mt19937 rng(std::chrono::system_clock::now().time_since_epoch().count()); // 随机种子

        // 1. 均匀随机选择一个中心。
        std::uniform_int_distribution<size_t> dist_idx(0, num_base_vectors - 1);
        size_t first_centroid_idx = dist_idx(rng);
        std::copy(base_data_source_ptr + first_centroid_idx * vector_dim,
                  base_data_source_ptr + (first_centroid_idx + 1) * vector_dim,
                  centroids_data.begin());
        chosen[first_centroid_idx] = true;

        // 根据第一个质心更新距离
        for (size_t i = 0; i < num_base_vectors; ++i) {
            if (i == first_centroid_idx) {
                 min_dist_sq[i] = 0.0f;
                 continue;
            }
            float d = inner_product_distance_simd(base_data_source_ptr + i * vector_dim, centroids_data.data(), vector_dim);
            min_dist_sq[i] = d; 
        }


        for (size_t c_idx = 1; c_idx < num_target_clusters; ++c_idx) {
            std::vector<double> weights(num_base_vectors);
            double total_weight = 0.0;
            for(size_t i=0; i<num_base_vectors; ++i) {
                if (chosen[i]) {
                    weights[i] = 0.0;
                } else {
                    weights[i] = static_cast<double>(min_dist_sq[i]); 
                }
                total_weight += weights[i];
            }
            
            if (total_weight == 0.0) { 
                bool found_new = false;
                for(size_t i=0; i<num_base_vectors; ++i) { 
                    if(!chosen[i]) {
                        std::copy(base_data_source_ptr + i * vector_dim,
                                  base_data_source_ptr + (i + 1) * vector_dim,
                                  centroids_data.data() + c_idx * vector_dim);
                        chosen[i] = true;
                        found_new = true;
                        break; 
                    }
                }
                if (!found_new) { // 所有点都已选择或相同，复制前一个
                     if (c_idx > 0) { // 确保存在前一个质心
                        std::copy(centroids_data.data() + (c_idx-1)*vector_dim,
                                centroids_data.data() + c_idx*vector_dim,
                                centroids_data.data() + c_idx*vector_dim);
                     } else { // 如果 c_idx 从 1 开始，则不应发生，但作为安全措施
                        // 如果绝对必要，可以再次选择一个随机点
                        size_t rand_fallback_idx = dist_idx(rng);
                         std::copy(base_data_source_ptr + rand_fallback_idx * vector_dim,
                                  base_data_source_ptr + (rand_fallback_idx + 1) * vector_dim,
                                  centroids_data.data() + c_idx * vector_dim);
                        chosen[rand_fallback_idx] = true; // 标记它，可能是一个副本，但总比未初始化好
                     }
                }
                // 为 c_idx 循环的下一次迭代更新 min_dist_sq
                if (c_idx < num_target_clusters -1) {
                    const float* new_centroid_ptr = centroids_data.data() + c_idx * vector_dim;
                    for (size_t i = 0; i < num_base_vectors; ++i) {
                        if (chosen[i]) continue;
                        float d = inner_product_distance_simd(base_data_source_ptr + i * vector_dim, new_centroid_ptr, vector_dim);
                        if (d < min_dist_sq[i]) {
                            min_dist_sq[i] = d;
                        }
                    }
                }
                continue; // 跳到下一个 c_idx 迭代
            }

            std::uniform_real_distribution<double> dist_prob_selection(0.0, total_weight);
            double rand_val = dist_prob_selection(rng);
            
            size_t next_centroid_base_idx = 0;
            double current_sum = 0.0;
            for (size_t i = 0; i < num_base_vectors; ++i) {
                current_sum += weights[i];
                if (current_sum >= rand_val && !chosen[i]) { // 由于浮点精度问题，确保不选择已选择的点
                    next_centroid_base_idx = i;
                    break;
                }
            }
             // 如果未找到（例如，所有加权点都已选择，或数值问题），则回退
            if (current_sum < rand_val || chosen[next_centroid_base_idx]) {
                 for(size_t i=0; i<num_base_vectors; ++i) if(!chosen[i]) {next_centroid_base_idx = i; break;}
            }


            std::copy(base_data_source_ptr + next_centroid_base_idx * vector_dim,
                      base_data_source_ptr + (next_centroid_base_idx + 1) * vector_dim,
                      centroids_data.data() + c_idx * vector_dim);
            chosen[next_centroid_base_idx] = true;

            // 根据新质心更新所有点的 min_dist_sq
            if (c_idx < num_target_clusters -1) { 
                const float* new_centroid_ptr = centroids_data.data() + c_idx * vector_dim;
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
        if (num_target_clusters == 0) return; // 没有簇可运行 k-means
        std::vector<int> assignments(num_base_vectors); 

        for (int iter = 0; iter < kmeans_max_iter; ++iter) {
            std::vector<pthread_t> threads_assign(num_threads_to_use - 1);
            std::vector<KMeansAssignArgs> args_assign(num_threads_to_use);
            
            std::vector<std::vector<float>> per_thread_partial_sums(
                num_threads_to_use, std::vector<float>(num_target_clusters * vector_dim, 0.0f));
            std::vector<std::vector<int>> per_thread_partial_counts(
                num_threads_to_use, std::vector<int>(num_target_clusters, 0));

            size_t items_per_thread = num_base_vectors / num_threads_to_use;
            size_t remainder = num_base_vectors % num_threads_to_use;
            size_t current_start_idx = 0;

            for (int i = 0; i < num_threads_to_use; ++i) {
                size_t current_chunk_size = items_per_thread + (i < static_cast<int>(remainder) ? 1 : 0);
                 if (current_chunk_size == 0 && current_start_idx < num_base_vectors) {
                    current_chunk_size = 1;
                }
                if (current_start_idx >= num_base_vectors) {
                    args_assign[i].start_idx = num_base_vectors;
                    args_assign[i].end_idx = num_base_vectors;
                } else {
                    args_assign[i].ivf_instance = this;
                    args_assign[i].start_idx = current_start_idx;
                    args_assign[i].end_idx = std::min(current_start_idx + current_chunk_size, num_base_vectors);
                    args_assign[i].current_centroids_ptr = &centroids_data;
                    args_assign[i].assignments_output_ptr = &assignments;
                    args_assign[i].local_sum_vectors_ptr = &per_thread_partial_sums[i];
                    args_assign[i].local_counts_ptr = &per_thread_partial_counts[i];
                }
                
                if (i < num_threads_to_use - 1 && args_assign[i].start_idx < args_assign[i].end_idx) {
                    pthread_create(&threads_assign[i], nullptr, kmeans_assign_worker_static, &args_assign[i]);
                } else if (args_assign[i].start_idx < args_assign[i].end_idx) { 
                    kmeans_assign_worker_static(&args_assign[i]);
                }
                current_start_idx += current_chunk_size;
            }

            for (int i = 0; i < num_threads_to_use - 1; ++i) {
                if (args_assign[i].start_idx < args_assign[i].end_idx) {
                    pthread_join(threads_assign[i], nullptr);
                }
            }

            std::vector<float> new_centroids_sum(num_target_clusters * vector_dim, 0.0f);
            std::vector<int> new_centroids_count(num_target_clusters, 0);

            for (int thread_idx = 0; thread_idx < num_threads_to_use; ++thread_idx) {
                for (size_t c = 0; c < num_target_clusters; ++c) {
                    if (per_thread_partial_counts[thread_idx][c] > 0) {
                        new_centroids_count[c] += per_thread_partial_counts[thread_idx][c];
                        for (size_t d = 0; d < vector_dim; ++d) {
                            new_centroids_sum[c * vector_dim + d] += per_thread_partial_sums[thread_idx][c * vector_dim + d];
                        }
                    }
                }
            }
            
            bool changed = false;
            for (size_t c = 0; c < num_target_clusters; ++c) {
                if (new_centroids_count[c] > 0) {
                    for (size_t d = 0; d < vector_dim; ++d) {
                        float new_val = new_centroids_sum[c * vector_dim + d] / new_centroids_count[c];
                        if (std::fabs(centroids_data[c * vector_dim + d] - new_val) > 1e-6) { 
                            changed = true;
                        }
                        centroids_data[c * vector_dim + d] = new_val;
                    }
                } else {
                    // 处理空簇：重新初始化为一个尚未成为质心的随机数据点
                    // 这是一个简单的策略；存在更复杂的策略。
                    if (num_base_vectors > num_target_clusters) { // 仅当有备用点时
                        std::mt19937 rng_reinit(iter + c); // 不同种子
                        std::uniform_int_distribution<size_t> dist_pt(0, num_base_vectors - 1);
                        size_t random_point_idx = dist_pt(rng_reinit);
                        // 一个简单的检查，以尽可能避免选择现有的质心点，并非详尽无遗
                        bool is_already_centroid_approx = false;
                        for(size_t cc=0; cc < num_target_clusters; ++cc) {
                            if (cc == c) continue;
                            float dist_to_other_centroid = 0;
                            for(size_t dd=0; dd < vector_dim; ++dd) {
                                float diff = base_data_source_ptr[random_point_idx*vector_dim + dd] - centroids_data[cc*vector_dim + dd];
                                dist_to_other_centroid += diff*diff;
                            }
                            if (sqrt(dist_to_other_centroid) < 1e-6) {
                                is_already_centroid_approx = true;
                                break;
                            }
                        }
                        if(!is_already_centroid_approx){
                             std::copy(base_data_source_ptr + random_point_idx * vector_dim,
                                  base_data_source_ptr + (random_point_idx + 1) * vector_dim,
                                  centroids_data.data() + c * vector_dim);
                            changed = true; // 重新初始化算作更改
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

        for (size_t i = 0; i < num_base_vectors; ++i) {
            if (assignments[i] >=0 && static_cast<size_t>(assignments[i]) < num_target_clusters) {
                 inverted_lists_data[assignments[i]].push_back(static_cast<uint32_t>(i));
            }
        }
    }


public: 
    static void* kmeans_assign_worker_static(void* arg) {
        KMeansAssignArgs* data = static_cast<KMeansAssignArgs*>(arg);
        IVFIndex* self = data->ivf_instance;

        if (data->start_idx >= data->end_idx) return nullptr; // 此线程无工作

        std::fill(data->local_sum_vectors_ptr->begin(), data->local_sum_vectors_ptr->end(), 0.0f);
        std::fill(data->local_counts_ptr->begin(), data->local_counts_ptr->end(), 0);

        for (size_t i = data->start_idx; i < data->end_idx; ++i) {
            const float* current_vector = self->base_data_source_ptr + i * self->vector_dim;
            float min_dist = std::numeric_limits<float>::max();
            int best_cluster_idx = 0;

            for (size_t c = 0; c < self->num_target_clusters; ++c) {
                const float* centroid_ptr = data->current_centroids_ptr->data() + c * self->vector_dim;
                float dist = inner_product_distance_simd(current_vector, centroid_ptr, self->vector_dim);
                if (dist < min_dist) {
                    min_dist = dist;
                    best_cluster_idx = static_cast<int>(c);
                }
            }
            (*data->assignments_output_ptr)[i] = best_cluster_idx;
            
            (*data->local_counts_ptr)[best_cluster_idx]++;
            for (size_t d = 0; d < self->vector_dim; ++d) {
                (*data->local_sum_vectors_ptr)[best_cluster_idx * self->vector_dim + d] += current_vector[d];
            }
        }
        return nullptr;
    }

    static void* search_centroids_worker_static(void* arg) {
        SearchCentroidsArgs* data = static_cast<SearchCentroidsArgs*>(arg);
        IVFIndex* self = data->ivf_instance;
        
        if (data->start_centroid_idx >= data->end_centroid_idx) return nullptr;

        data->thread_centroid_distances_output_ptr->clear();
        data->thread_centroid_distances_output_ptr->reserve(data->end_centroid_idx - data->start_centroid_idx);

        for (size_t c_idx = data->start_centroid_idx; c_idx < data->end_centroid_idx; ++c_idx) {
            const float* centroid_ptr = self->centroids_data.data() + c_idx * self->vector_dim;
            float dist = inner_product_distance_simd(data->query_ptr, centroid_ptr, self->vector_dim);
            data->thread_centroid_distances_output_ptr->emplace_back(dist, static_cast<int>(c_idx));
        }
        return nullptr;
    }

    static void* search_lists_worker_static(void* arg) {
        SearchListsArgs* data = static_cast<SearchListsArgs*>(arg);
        IVFIndex* self = data->ivf_instance;

        if (data->task_start_idx_in_candidates >= data->task_end_idx_in_candidates) return nullptr;
        
        while(!data->thread_top_k_output_ptr->empty()) data->thread_top_k_output_ptr->pop();

        for (size_t i = data->task_start_idx_in_candidates; i < data->task_end_idx_in_candidates; ++i) {
            int cluster_idx = (*data->candidate_cluster_indices_ptr)[i];
            if (cluster_idx < 0 || static_cast<size_t>(cluster_idx) >= self->inverted_lists_data.size()) continue;

            const auto& point_indices_in_cluster = self->inverted_lists_data[cluster_idx];
            for (uint32_t point_orig_idx : point_indices_in_cluster) {
                const float* base_vector_ptr = self->base_data_source_ptr + point_orig_idx * self->vector_dim;
                float dist = inner_product_distance_simd(data->query_ptr, base_vector_ptr, self->vector_dim);

                if (data->thread_top_k_output_ptr->size() < data->k_neighbors) {
                    data->thread_top_k_output_ptr->push({dist, point_orig_idx});
                } else if (dist < data->thread_top_k_output_ptr->top().first) {
                    data->thread_top_k_output_ptr->pop();
                    data->thread_top_k_output_ptr->push({dist, point_orig_idx});
                }
            }
        }
        return nullptr;
    }
};
