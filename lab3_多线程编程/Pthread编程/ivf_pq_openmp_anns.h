#pragma once

#include <vector>
#include <queue>
#include <cstdint>
#include <algorithm>
#include <limits>
#include <cmath>
#include <numeric>
#include <random>
#include <map>
#include <chrono>
#include <iostream>
#include <cstring> // 用于 memcpy/memset
#include <stdexcept>
#include <omp.h>       // 引入 OpenMP

#include "simd_anns.h" // 用于 inner_product_distance_simd 和 compute_l2_sq_neon
#include "pq_anns.h"   // 用于 ProductQuantizer

class IVFPQIndexOpenMP {
private:
    size_t vecdim;
    size_t num_ivf_clusters;
    ProductQuantizer* pq_quantizer; 
    size_t pq_nsub_config;          
    int num_threads_omp;             // 用于 OpenMP 的线程数
    int ivf_kmeans_iterations;

    std::vector<float> ivf_centroids_data; 
    std::vector<std::vector<uint32_t>> ivf_inverted_lists_data; 

    // IVF 使用 L2 距离
    float compute_distance_ivf(const float* v1, const float* v2, size_t dim) const {
        return compute_l2_sq_neon(v1, v2, dim); 
    }

    // 重排序使用 IP 距离
    float compute_distance_reranking(const float* v1, const float* v2, size_t dim) const {
        return inner_product_distance_simd(v1, v2, dim); 
    }

public:
    IVFPQIndexOpenMP(size_t dim, size_t n_ivf_clusters, 
                     size_t pq_nsub, 
                     int threads = 1, int ivf_iter = 20)
        : vecdim(dim), num_ivf_clusters(n_ivf_clusters), 
          pq_quantizer(nullptr), 
          pq_nsub_config(pq_nsub),
          num_threads_omp(threads > 0 ? threads : 1), // 确保线程数至少为1
          ivf_kmeans_iterations(ivf_iter) {
        if (vecdim == 0) throw std::invalid_argument("IVFPQIndexOpenMP: 向量维度不能为零。");
        if (pq_nsub_config == 0) throw std::invalid_argument("IVFPQIndexOpenMP: pq_nsub 不能为零。");
        if (vecdim > 0 && pq_nsub_config > 0 && vecdim % pq_nsub_config != 0) {
            // PQ 构造函数会处理这个问题，但提前警告可能有用
            // std::cerr << "IVFPQIndexOpenMP 警告: vecdim 不能被 pq_nsub 整除。" << std::endl;
        }
    }

    ~IVFPQIndexOpenMP() {
        delete pq_quantizer; 
    }
    
    void build(const float* all_base_data, size_t num_all_base_data, 
               double pq_train_ratio_for_pq) { 
        if (!all_base_data || num_all_base_data == 0) {
            std::cerr << "IVFPQ-OpenMP: 基准数据为空，无法构建。" << std::endl;
            return;
        }
        if (num_ivf_clusters > 0 && num_all_base_data < num_ivf_clusters) {
             std::cerr << "IVFPQ-OpenMP: 警告 - 基准向量数量 (" << num_all_base_data 
                       << ") 小于 num_ivf_clusters (" << num_ivf_clusters << ")." << std::endl;
        }

        // 1. 构建 IVF 部分 (K-means)
        if (num_ivf_clusters > 0) {
            std::cout << "IVFPQ-OpenMP: 正在构建 IVF 部分 (基于 L2)... (簇数=" << num_ivf_clusters 
                      << ", 迭代次数=" << ivf_kmeans_iterations << ", 线程数=" << num_threads_omp << ")" << std::endl;
            ivf_centroids_data.assign(num_ivf_clusters * vecdim, 0.0f);
            
            std::mt19937 rng(std::chrono::system_clock::now().time_since_epoch().count() + 2); // 不同种子
            std::vector<size_t> initial_centroid_indices(num_all_base_data); 
            std::iota(initial_centroid_indices.begin(), initial_centroid_indices.end(), 0);
            std::shuffle(initial_centroid_indices.begin(), initial_centroid_indices.end(), rng);

            size_t num_initial_centroids_to_pick = std::min(num_all_base_data, num_ivf_clusters);
            for(size_t i=0; i<num_initial_centroids_to_pick; ++i) {
                memcpy(ivf_centroids_data.data() + i * vecdim, all_base_data + initial_centroid_indices[i] * vecdim, vecdim * sizeof(float));
            }

            std::vector<int> assignments(num_all_base_data);
            std::vector<float> iteration_centroids_sum_global(num_ivf_clusters * vecdim);
            std::vector<int> iteration_centroids_count_global(num_ivf_clusters);

            for (int iter = 0; iter < ivf_kmeans_iterations; ++iter) {
                std::fill(iteration_centroids_sum_global.begin(), iteration_centroids_sum_global.end(), 0.0f);
                std::fill(iteration_centroids_count_global.begin(), iteration_centroids_count_global.end(), 0);

                // 为每个线程准备局部累加空间
                std::vector<std::vector<float>> local_sums_per_thread(num_threads_omp, std::vector<float>(num_ivf_clusters * vecdim, 0.0f));
                std::vector<std::vector<int>> local_counts_per_thread(num_threads_omp, std::vector<int>(num_ivf_clusters, 0));
                
                #pragma omp parallel num_threads(num_threads_omp)
                {
                    int thread_id = omp_get_thread_num();
                    #pragma omp for schedule(dynamic) // 动态调度，因为计算每个点的最近质心耗时可能不同
                    for (size_t i = 0; i < num_all_base_data; ++i) {
                        const float* point = all_base_data + i * vecdim;
                        float min_dist = std::numeric_limits<float>::max();
                        int best_cluster = -1;

                        if (num_ivf_clusters == 0) continue; // 如果进入循环则不应发生

                        for (size_t c = 0; c < num_ivf_clusters; ++c) {
                            const float* centroid = ivf_centroids_data.data() + c * vecdim; // 使用当前迭代的全局质心
                            float dist = compute_distance_ivf(point, centroid, vecdim);
                            if (dist < min_dist) {
                                min_dist = dist;
                                best_cluster = c;
                            }
                        }
                        if (best_cluster != -1) {
                            assignments[i] = best_cluster; // assignments[i] 由单个线程写入，安全
                            for (size_t d = 0; d < vecdim; ++d) {
                                local_sums_per_thread[thread_id][best_cluster * vecdim + d] += point[d];
                            }
                            local_counts_per_thread[thread_id][best_cluster]++;
                        }
                    }
                } // 结束并行区域

                //串行归约每个线程的局部结果到全局
                for(int t=0; t < num_threads_omp; ++t) {
                    for(size_t c=0; c < num_ivf_clusters; ++c) {
                        if(local_counts_per_thread[t][c] > 0) {
                            iteration_centroids_count_global[c] += local_counts_per_thread[t][c];
                            for(size_t d=0; d < vecdim; ++d) {
                                iteration_centroids_sum_global[c * vecdim + d] += local_sums_per_thread[t][c * vecdim + d];
                            }
                        }
                    }
                }
                
                bool converged = true;
                // #pragma omp parallel for num_threads(num_threads_omp) schedule(static) reduction(&&:converged) // 如果质心更新也并行化
                for (size_t c = 0; c < num_ivf_clusters; ++c) {
                    if (iteration_centroids_count_global[c] > 0) {
                        for (size_t d = 0; d < vecdim; ++d) {
                            float new_val = iteration_centroids_sum_global[c * vecdim + d] / iteration_centroids_count_global[c];
                            if (std::abs(new_val - ivf_centroids_data[c * vecdim + d]) > 1e-5f) { 
                                converged = false;
                            }
                            ivf_centroids_data[c * vecdim + d] = new_val;
                        }
                    } else { // 处理空簇
                        if (num_all_base_data > 0) { 
                             size_t rand_idx = initial_centroid_indices[rng() % num_all_base_data]; 
                             memcpy(ivf_centroids_data.data() + c * vecdim, all_base_data + rand_idx * vecdim, vecdim * sizeof(float));
                             converged = false; 
                        }
                    }
                }
                if (converged && iter > 0) {
                    // std::cout << "IVF K-means 在迭代 " << iter + 1 << " 次后收敛" << std::endl;
                    break; 
                }
            }

            ivf_inverted_lists_data.assign(num_ivf_clusters, std::vector<uint32_t>());
            // 填充倒排列表 (串行，或者也可以并行化，但通常较快)
            for (size_t i = 0; i < num_all_base_data; ++i) {
                 if (assignments[i] >=0 && static_cast<size_t>(assignments[i]) < num_ivf_clusters) {
                    ivf_inverted_lists_data[assignments[i]].push_back(static_cast<uint32_t>(i));
                 }
            }
            std::cout << "IVFPQ-OpenMP: IVF 部分构建完成。" << std::endl;
        }

        // 2. 构建 PQ 部分 (ProductQuantizer 内部已使用 OpenMP)
        std::cout << "IVFPQ-OpenMP: 正在构建 PQ 部分 (基于 L2)..." << std::endl;
        delete pq_quantizer; 
        try {
            pq_quantizer = new ProductQuantizer(all_base_data, num_all_base_data, vecdim, 
                                                this->pq_nsub_config, pq_train_ratio_for_pq);
        } catch (const std::exception& e) {
            std::cerr << "IVFPQ-OpenMP: 创建 ProductQuantizer 时出错: " << e.what() << std::endl;
            pq_quantizer = nullptr; 
            throw; 
        }
        
        if (pq_quantizer) { 
            std::cout << "IVFPQ-OpenMP: PQ 部分已构建，数据已由 ProductQuantizer 编码。" << std::endl;
        } else {
            std::cerr << "IVFPQ-OpenMP: PQ 部分无法构建。" << std::endl;
        }
    }


    std::priority_queue<std::pair<float, uint32_t>> search(
        const float* query,
        const float* base_data_for_reranking, 
        size_t k,
        size_t nprobe,
        size_t rerank_k_candidates = 0
    ) {
        std::priority_queue<std::pair<float, uint32_t>> final_results_heap; 

        if (k == 0) return final_results_heap;
        if (!pq_quantizer) {
            std::cerr << "IVFPQ-OpenMP 搜索: ProductQuantizer 不可用。" << std::endl;
            return final_results_heap;
        }

        bool ivf_active = (num_ivf_clusters > 0 && !ivf_centroids_data.empty() && !ivf_inverted_lists_data.empty());
        if (ivf_active && nprobe == 0) nprobe = 1;
        if (ivf_active && nprobe > num_ivf_clusters) nprobe = num_ivf_clusters;

        std::vector<int> nprobe_cluster_indices;
        if (ivf_active) {
            std::vector<std::pair<float, int>> all_ivf_centroid_distances;
            all_ivf_centroid_distances.reserve(num_ivf_clusters);
            
            // 使用 OpenMP 并行计算查询向量到所有 IVF 中心的距离
            #pragma omp parallel num_threads(num_threads_omp)
            {
                std::vector<std::pair<float, int>> local_centroid_distances;
                local_centroid_distances.reserve(num_ivf_clusters / num_threads_omp + 1); // 预估大小

                #pragma omp for schedule(dynamic)
                for (size_t i = 0; i < num_ivf_clusters; ++i) {
                    const float* centroid_vec = ivf_centroids_data.data() + i * vecdim;
                    float dist = compute_distance_ivf(query, centroid_vec, vecdim);
                    local_centroid_distances.push_back({dist, static_cast<int>(i)});
                }

                #pragma omp critical // 合并局部结果到全局列表
                {
                    all_ivf_centroid_distances.insert(all_ivf_centroid_distances.end(),
                                                      local_centroid_distances.begin(),
                                                      local_centroid_distances.end());
                }
            }
            
            std::sort(all_ivf_centroid_distances.begin(), all_ivf_centroid_distances.end());
            nprobe_cluster_indices.reserve(nprobe);
            for (size_t i = 0; i < std::min(nprobe, all_ivf_centroid_distances.size()); ++i) {
                nprobe_cluster_indices.push_back(all_ivf_centroid_distances[i].second);
            }

            if (nprobe_cluster_indices.empty() && num_ivf_clusters > 0 && nprobe > 0) {
                std::cerr << "IVFPQ-OpenMP 搜索: 未找到 IVF 候选簇。返回空。" << std::endl;
                return final_results_heap;
            }
        } else {
             std::cerr << "IVFPQ-OpenMP 搜索: IVF 部分未激活。标准 IVFPQ 搜索无法继续。" << std::endl;
             return final_results_heap;
        }
        if (nprobe_cluster_indices.empty() && ivf_active) {
             std::cerr << "IVFPQ-OpenMP 搜索: nprobe_cluster_indices 为空。返回空。" << std::endl;
             return final_results_heap;
        }

        bool perform_reranking = (rerank_k_candidates > k && base_data_for_reranking != nullptr);
        size_t k_for_pq_stage = perform_reranking ? rerank_k_candidates : k;
        if (k_for_pq_stage == 0 && k > 0) k_for_pq_stage = k;
        if (k_for_pq_stage == 0) return final_results_heap;

        std::priority_queue<std::pair<float, uint32_t>> merged_pq_candidates_heap;
        std::vector<float> query_pq_dist_table;
        pq_quantizer->compute_query_distance_table(query, query_pq_dist_table); 

        if (query_pq_dist_table.empty()) {
            std::cerr << "IVFPQ-OpenMP 搜索: 计算 PQ 距离表失败。返回空。" << std::endl;
            return final_results_heap;
        }

        size_t num_candidate_clusters_to_search = nprobe_cluster_indices.size();
        if (num_candidate_clusters_to_search == 0 && ivf_active) return final_results_heap;

        // 为每个线程准备一个局部的 top-k 优先队列
        std::vector<std::priority_queue<std::pair<float, uint32_t>>> per_thread_top_k_pq(num_threads_omp);

        #pragma omp parallel num_threads(num_threads_omp)
        {
            int thread_id = omp_get_thread_num();
            #pragma omp for schedule(dynamic) // 动态调度，因为每个簇内的点数不同
            for (size_t i = 0; i < num_candidate_clusters_to_search; ++i) {
                int cluster_idx = nprobe_cluster_indices[i];
                if (cluster_idx < 0 || static_cast<size_t>(cluster_idx) >= ivf_inverted_lists_data.size()) continue;

                const auto& point_indices_in_cluster = ivf_inverted_lists_data[cluster_idx];
                for (uint32_t point_orig_idx : point_indices_in_cluster) {
                    const uint8_t* item_code = pq_quantizer->get_code_for_item(point_orig_idx);
                    if (item_code) {
                        float approx_dist_sq = pq_quantizer->compute_asymmetric_distance_sq_with_table(item_code, query_pq_dist_table);
                        
                        if (per_thread_top_k_pq[thread_id].size() < k_for_pq_stage) {
                            per_thread_top_k_pq[thread_id].push({approx_dist_sq, point_orig_idx});
                        } else if (approx_dist_sq < per_thread_top_k_pq[thread_id].top().first) {
                            per_thread_top_k_pq[thread_id].pop();
                            per_thread_top_k_pq[thread_id].push({approx_dist_sq, point_orig_idx});
                        }
                    }
                }
            }
        } // 结束并行区域

        // 串行合并所有线程的局部 PQ 结果
        for (int t = 0; t < num_threads_omp; ++t) {
            while (!per_thread_top_k_pq[t].empty()) {
                std::pair<float, uint32_t> cand = per_thread_top_k_pq[t].top();
                per_thread_top_k_pq[t].pop();
                if (merged_pq_candidates_heap.size() < k_for_pq_stage) {
                    merged_pq_candidates_heap.push(cand);
                } else if (cand.first < merged_pq_candidates_heap.top().first) {
                    merged_pq_candidates_heap.pop();
                    merged_pq_candidates_heap.push(cand);
                }
            }
        }

        if (perform_reranking) {
            std::vector<std::pair<float, uint32_t>> pq_candidates_vec;
            pq_candidates_vec.reserve(merged_pq_candidates_heap.size());
            while(!merged_pq_candidates_heap.empty()){
                pq_candidates_vec.push_back(merged_pq_candidates_heap.top()); 
                merged_pq_candidates_heap.pop(); 
            }
            // 注意：从最大堆中取出时是降序的，如果需要升序处理，可以反转或直接使用
            // 这里我们直接迭代，final_results_heap 会保持正确的 top-k

            // #pragma omp parallel for num_threads(num_threads_omp) schedule(dynamic) if(pq_candidates_vec.size() > 1000) // 仅当候选项足够多时并行化重排序
            // 上面的并行化重排序需要对 final_results_heap 的访问加锁，或者每个线程有局部堆最后合并
            // 为简单起见，暂时保持重排序部分为串行，或者使用临界区
            for(const auto& pq_cand_pair : pq_candidates_vec){
                uint32_t original_idx = pq_cand_pair.second;
                const float* exact_vec = base_data_for_reranking + static_cast<size_t>(original_idx) * vecdim;
                float exact_ip_dist = compute_distance_reranking(query, exact_vec, vecdim); 

                // #pragma omp critical // 如果并行化重排序，需要这个
                // {
                    if (final_results_heap.size() < k) {
                        final_results_heap.push({exact_ip_dist, original_idx});
                    } else if (exact_ip_dist < final_results_heap.top().first) {
                        final_results_heap.pop();
                        final_results_heap.push({exact_ip_dist, original_idx});
                    }
                // }
            }
            return final_results_heap; 
        } else {
            return merged_pq_candidates_heap; 
        }
    }
};
