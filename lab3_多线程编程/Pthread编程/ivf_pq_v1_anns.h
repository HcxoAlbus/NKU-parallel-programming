#pragma once

#include <vector>
#include <queue>
#include <cstdint>
#include <algorithm>
#include <limits>
#include <cmath>
#include <pthread.h>
#include <numeric>
#include <random>
#include <map>
#include <chrono>
#include <iostream>
#include <cstring> // For memcpy/memset
#include <stdexcept>

#include "simd_anns.h" // For inner_product_distance_simd (for reranking) and compute_l2_sq_neon
#include "pq_anns.h"   // For ProductQuantizer

// Forward declaration
class IVFPQIndexV1;

// --- Pthread Data Structures for IVFPQIndexV1 ---

struct IVFPQV1_KMeansAssignArgs {
    IVFPQIndexV1* ivfpq_instance;
    const float* all_reconstructed_data_ptr; // Now points to original base data for IVF training
    size_t start_idx_data;    
    size_t end_idx_data;      
    const std::vector<float>* current_ivf_centroids_ptr; // Centroids are trained on original data
    std::vector<int>* assignments_output_ptr;    // Stores original data indices' assignments
    std::vector<float>* local_sum_vectors_ptr;    
    std::vector<int>* local_counts_ptr;        
};

struct IVFPQV1_SearchIVFCentroidsArgs {
    IVFPQIndexV1* ivfpq_instance;
    const float* query_ptr;
    size_t start_centroid_idx;
    size_t end_centroid_idx;
    std::vector<std::pair<float, int>>* thread_centroid_distances_output_ptr;
};

struct IVFPQV1_SearchListsPQArgs {
    IVFPQIndexV1* ivfpq_instance;
    size_t k_to_collect; 
    const std::vector<int>* candidate_cluster_indices_ptr; 
    size_t task_start_idx_in_candidates;
    size_t task_end_idx_in_candidates;
    std::priority_queue<std::pair<float, uint32_t>>* thread_top_k_output_ptr; 
    const ProductQuantizer* pq_quantizer_ptr; 
    const std::vector<float>* query_pq_dist_table_ptr; 
};


class IVFPQIndexV1 {
private:
    size_t vecdim;
    size_t num_ivf_clusters;
    ProductQuantizer* pq_quantizer; // PQ is built first on original data
    size_t pq_nsub_config;
    size_t num_threads;
    int ivf_kmeans_iterations;

    std::vector<float> ivf_centroids_data; // Centroids for IVF, trained on original base data
    std::vector<std::vector<uint32_t>> ivf_inverted_lists_data; // Stores original data indices

    // IVF uses L2 distance. For V1, this is on original vectors.
    float compute_distance_ivf(const float* v1, const float* v2, size_t dim) const {
        return compute_l2_sq_neon(v1, v2, dim);
    }

    // Reranking uses IP distance on original data
    float compute_distance_reranking(const float* v1, const float* v2, size_t dim) const {
        return inner_product_distance_simd(v1, v2, dim);
    }

    static void* kmeans_assign_worker_v1_static(void* arg) {
        auto data = static_cast<IVFPQV1_KMeansAssignArgs*>(arg);
        IVFPQIndexV1* self = data->ivfpq_instance;

        data->local_sum_vectors_ptr->assign(self->num_ivf_clusters * self->vecdim, 0.0f);
        data->local_counts_ptr->assign(self->num_ivf_clusters, 0);

        for (size_t i = data->start_idx_data; i < data->end_idx_data; ++i) {
            // 'i' is the original index of the data point.
            // We use the original data point for IVF clustering.
            const float* point = data->all_reconstructed_data_ptr + i * self->vecdim; // This now points to original data
            float min_dist = std::numeric_limits<float>::max();
            int best_cluster = -1;

            if (self->num_ivf_clusters == 0) continue;

            for (size_t c = 0; c < self->num_ivf_clusters; ++c) {
                const float* centroid = data->current_ivf_centroids_ptr->data() + c * self->vecdim;
                float dist = self->compute_distance_ivf(point, centroid, self->vecdim);
                if (dist < min_dist) {
                    min_dist = dist;
                    best_cluster = static_cast<int>(c);
                }
            }
            if (best_cluster != -1) {
                (*data->assignments_output_ptr)[i] = best_cluster; // Store assignment for original index i
                for (size_t d = 0; d < self->vecdim; ++d) {
                    (*data->local_sum_vectors_ptr)[best_cluster * self->vecdim + d] += point[d];
                }
                (*data->local_counts_ptr)[best_cluster]++;
            }
        }
        return nullptr;
    }

    static void* search_ivf_centroids_worker_v1_static(void* arg) {
        IVFPQV1_SearchIVFCentroidsArgs* data = static_cast<IVFPQV1_SearchIVFCentroidsArgs*>(arg);
        IVFPQIndexV1* self = data->ivfpq_instance;
        data->thread_centroid_distances_output_ptr->clear();

        for (size_t i = data->start_centroid_idx; i < data->end_centroid_idx; ++i) {
            const float* centroid_vec = self->ivf_centroids_data.data() + i * self->vecdim;
            // Query is original, IVF centroids are based on reconstructed data. L2 distance.
            float dist = self->compute_distance_ivf(data->query_ptr, centroid_vec, self->vecdim);
            data->thread_centroid_distances_output_ptr->push_back({dist, static_cast<int>(i)});
        }
        return nullptr;
    }

    static void* search_lists_pq_worker_v1_static(void* arg) {
        IVFPQV1_SearchListsPQArgs* data = static_cast<IVFPQV1_SearchListsPQArgs*>(arg);
        IVFPQIndexV1* self = data->ivfpq_instance; 
        const ProductQuantizer* pq = data->pq_quantizer_ptr; // This is the globally trained PQ
        const std::vector<float>* query_dist_table = data->query_pq_dist_table_ptr;

        if (data->task_start_idx_in_candidates >= data->task_end_idx_in_candidates || !pq ) {
            return nullptr;
        }
        if (!query_dist_table || query_dist_table->empty()) {
             if (pq) {
                // std::cerr << "Worker V1: Invalid or empty query_pq_dist_table_ptr." << std::endl;
             }
            return nullptr;
        }
    
        // Clear the thread's local heap before filling
        while(!data->thread_top_k_output_ptr->empty()) data->thread_top_k_output_ptr->pop();
    
        for (size_t i = data->task_start_idx_in_candidates; i < data->task_end_idx_in_candidates; ++i) {
            int cluster_idx = (*data->candidate_cluster_indices_ptr)[i];
            if (cluster_idx < 0 || static_cast<size_t>(cluster_idx) >= self->ivf_inverted_lists_data.size()) continue;

            const auto& point_indices_in_cluster = self->ivf_inverted_lists_data[cluster_idx]; // These are original indices
            for (uint32_t point_orig_idx : point_indices_in_cluster) {
                const uint8_t* item_code = pq->get_code_for_item(point_orig_idx); // Get PQ code for original item
                if (item_code) {
                    float dist_pq = pq->compute_asymmetric_distance_sq_with_table(item_code, *query_dist_table);
                    if (data->thread_top_k_output_ptr->size() < data->k_to_collect || dist_pq < data->thread_top_k_output_ptr->top().first) {
                        if (data->thread_top_k_output_ptr->size() == data->k_to_collect) {
                            data->thread_top_k_output_ptr->pop();
                        }
                        data->thread_top_k_output_ptr->push({dist_pq, point_orig_idx});
                    }
                }
            }
        }
        return nullptr;
    }

public:
    IVFPQIndexV1(size_t dim, size_t n_ivf_clusters, 
                 size_t pq_nsub, 
                 size_t threads = 1, int ivf_iter = 20)
        : vecdim(dim), num_ivf_clusters(n_ivf_clusters), 
          pq_quantizer(nullptr), 
          pq_nsub_config(pq_nsub),
          num_threads(threads), ivf_kmeans_iterations(ivf_iter) {
        if (vecdim == 0) throw std::invalid_argument("IVFPQIndexV1: Vector dimension cannot be zero.");
        if (pq_nsub_config == 0) throw std::invalid_argument("IVFPQIndexV1: pq_nsub cannot be zero.");
        if (num_ivf_clusters == 0) std::cout << "IVFPQIndexV1 Warning: num_ivf_clusters is 0. IVF part will be inactive." << std::endl;
    }

    ~IVFPQIndexV1() {
        delete pq_quantizer;
    }
    
    void build(const float* all_base_data, size_t num_all_base_data, 
               double pq_train_ratio_for_pq) { 
        if (!all_base_data || num_all_base_data == 0) {
            std::cerr << "IVFPQ_V1: Base data is empty, cannot build." << std::endl;
            return;
        }

        // 1. Build PQ part first: Train PQ on original data and encode all original data
        std::cout << "IVFPQ_V1: Building PQ part (L2-based)..." << std::endl;
        delete pq_quantizer; 
        try {
            pq_quantizer = new ProductQuantizer(all_base_data, num_all_base_data, vecdim, 
                                                this->pq_nsub_config, pq_train_ratio_for_pq);
        } catch (const std::exception& e) {
            std::cerr << "IVFPQ_V1: Error creating ProductQuantizer: " << e.what() << std::endl;
            pq_quantizer = nullptr;
            throw; 
        }
        if (!pq_quantizer) { // Should be caught by throw above, but defensive
             std::cerr << "IVFPQ_V1: PQ part could not be built. Aborting build." << std::endl;
             return;
        }
        std::cout << "IVFPQ_V1: PQ part built and all base data encoded by ProductQuantizer." << std::endl;

        // 2. Build IVF part on original base data
        if (num_ivf_clusters == 0) {
            std::cout << "IVFPQ_V1: num_ivf_clusters is 0, skipping IVF part construction." << std::endl;
            return;
        }
        if (num_all_base_data < num_ivf_clusters) {
             std::cerr << "IVFPQ_V1: Warning - number of base vectors (" << num_all_base_data 
                       << ") is less than num_ivf_clusters (" << num_ivf_clusters << ")." << std::endl;
        }

        std::cout << "IVFPQ_V1: Building IVF part (L2-based on original data)..."
                  << " (clusters=" << num_ivf_clusters << ", iters=" << ivf_kmeans_iterations << ")" << std::endl;

        ivf_centroids_data.assign(num_ivf_clusters * vecdim, 0.0f);
        
        std::mt19937 rng(std::chrono::system_clock::now().time_since_epoch().count() + 2); // Seed + 2
        std::vector<size_t> initial_centroid_indices(num_all_base_data); 
        std::iota(initial_centroid_indices.begin(), initial_centroid_indices.end(), 0);
        std::shuffle(initial_centroid_indices.begin(), initial_centroid_indices.end(), rng);

        size_t num_initial_centroids_to_pick = std::min(num_all_base_data, num_ivf_clusters);
        for(size_t i=0; i<num_initial_centroids_to_pick; ++i) {
            // Initialize IVF centroids from the original base data
            memcpy(ivf_centroids_data.data() + i * vecdim, 
                   all_base_data + initial_centroid_indices[i] * vecdim, 
                   vecdim * sizeof(float));
        }

        std::vector<int> assignments(num_all_base_data); // Stores cluster ID for each original data point
        std::vector<float> iteration_centroids_sum(num_ivf_clusters * vecdim);
        std::vector<int> iteration_centroids_count(num_ivf_clusters);

        std::vector<pthread_t> kmeans_threads(num_threads > 1 ? num_threads -1 : 0);
        std::vector<IVFPQV1_KMeansAssignArgs> kmeans_args(num_threads);
        std::vector<std::vector<float>> per_thread_sums(num_threads, std::vector<float>(num_ivf_clusters * vecdim));
        std::vector<std::vector<int>> per_thread_counts(num_threads, std::vector<int>(num_ivf_clusters));

        for (int iter = 0; iter < ivf_kmeans_iterations; ++iter) {
            std::fill(iteration_centroids_sum.begin(), iteration_centroids_sum.end(), 0.0f);
            std::fill(iteration_centroids_count.begin(), iteration_centroids_count.end(), 0);

            size_t data_per_thread = num_all_base_data / num_threads;
            size_t data_remainder = num_all_base_data % num_threads;
            size_t current_data_start_idx = 0; // This is an index for original data points

            for(size_t t=0; t < num_threads; ++t) {
                kmeans_args[t].ivfpq_instance = this;
                kmeans_args[t].all_reconstructed_data_ptr = all_base_data; // Use original data
                kmeans_args[t].start_idx_data = current_data_start_idx;
                size_t chunk_size = data_per_thread + (t < data_remainder ? 1 : 0);
                kmeans_args[t].end_idx_data = current_data_start_idx + chunk_size;
                kmeans_args[t].current_ivf_centroids_ptr = &ivf_centroids_data;
                kmeans_args[t].assignments_output_ptr = &assignments;
                kmeans_args[t].local_sum_vectors_ptr = &per_thread_sums[t];
                kmeans_args[t].local_counts_ptr = &per_thread_counts[t];
                
                if (t < num_threads - 1 && num_threads > 1) {
                    if (chunk_size > 0) pthread_create(&kmeans_threads[t], nullptr, kmeans_assign_worker_v1_static, &kmeans_args[t]);
                } else { // Last thread or single thread
                    if (chunk_size > 0) kmeans_assign_worker_v1_static(&kmeans_args[t]);
                }
                current_data_start_idx += chunk_size;
            }

            for(size_t t=0; t < num_threads -1 && num_threads > 1; ++t) {
                if (kmeans_args[t].start_idx_data < kmeans_args[t].end_idx_data) pthread_join(kmeans_threads[t], nullptr);
            }

            for(size_t t=0; t < num_threads; ++t) {
                 if (kmeans_args[t].start_idx_data < kmeans_args[t].end_idx_data) {
                    for (size_t c = 0; c < num_ivf_clusters; ++c) {
                        for (size_t d = 0; d < vecdim; ++d) {
                            iteration_centroids_sum[c * vecdim + d] += per_thread_sums[t][c * vecdim + d];
                        }
                        iteration_centroids_count[c] += per_thread_counts[t][c];
                    }
                }
            }
            
            bool converged = true;
            for (size_t c = 0; c < num_ivf_clusters; ++c) {
                if (iteration_centroids_count[c] > 0) {
                    std::vector<float> new_centroid(vecdim);
                    for (size_t d = 0; d < vecdim; ++d) {
                        new_centroid[d] = iteration_centroids_sum[c * vecdim + d] / iteration_centroids_count[c];
                    }
                    // Check for convergence (optional, simple check here)
                    float diff_sum = 0.0f;
                     for (size_t d = 0; d < vecdim; ++d) {
                        diff_sum += std::abs(ivf_centroids_data[c*vecdim+d] - new_centroid[d]);
                     }
                    if (diff_sum > 1e-5 * vecdim) converged = false; // Simple convergence check
                    memcpy(ivf_centroids_data.data() + c * vecdim, new_centroid.data(), vecdim * sizeof(float));
                } else { // Handle empty cluster - reinitialize or mark
                    // std::cerr << "IVFPQ_V1: K-means iter " << iter << ", cluster " << c << " is empty." << std::endl;
                    // Could re-initialize this centroid from a random (reconstructed) point or split a large cluster
                    // For simplicity, leave it (might lead to fewer effective clusters)
                }
            }
            if (converged && iter > 0) { // Min 1 iter
                std::cout << "IVFPQ_V1: K-means converged after " << iter + 1 << " iterations." << std::endl;
                break;
            }
             if (iter == ivf_kmeans_iterations -1) std::cout << "IVFPQ_V1: K-means finished max iterations." << std::endl;
        }

        ivf_inverted_lists_data.assign(num_ivf_clusters, std::vector<uint32_t>());
        for (size_t i = 0; i < num_all_base_data; ++i) { // Iterate through original data indices
             if (assignments[i] >=0 && static_cast<size_t>(assignments[i]) < num_ivf_clusters) {
                ivf_inverted_lists_data[assignments[i]].push_back(static_cast<uint32_t>(i));
             }
        }
        std::cout << "IVFPQ_V1: IVF part built on original data." << std::endl;
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
            std::cerr << "IVFPQ_V1 search: ProductQuantizer not available. Build failed or not called." << std::endl;
            return final_results_heap;
        }
        
        bool ivf_active = (num_ivf_clusters > 0 && !ivf_centroids_data.empty() && !ivf_inverted_lists_data.empty());
        if (!ivf_active) { // If IVF is not active (e.g. num_ivf_clusters was 0)
            // Fallback to full PQ scan if IVF is not active
            // std::cout << "IVFPQ_V1 search: IVF not active, performing full PQ scan." << std::endl;
            std::vector<float> query_pq_dist_table;
            pq_quantizer->compute_query_distance_table(query, query_pq_dist_table);
            if (query_pq_dist_table.empty()) {
                 std::cerr << "IVFPQ_V1 search: Failed to compute PQ distance table for full scan." << std::endl;
                 return final_results_heap;
            }
            for (uint32_t i = 0; i < pq_quantizer->get_nbase(); ++i) { // Changed to get_nbase()
                const uint8_t* code = pq_quantizer->get_code_for_item(i);
                if (code) {
                    float dist = pq_quantizer->compute_asymmetric_distance_sq_with_table(code, query_pq_dist_table);
                     if (final_results_heap.size() < k || dist < final_results_heap.top().first) {
                        if (final_results_heap.size() == k) final_results_heap.pop();
                        final_results_heap.push({dist, i});
                    }
                }
            }
            // Reranking for this non-IVF path is not implemented here for brevity,
            // but would be similar to the main path's reranking.
            // The current `base_data_for_reranking` is for the main path.
            // If reranking is essential for this fallback, it needs to be added.
            // For now, returning PQ results directly.
            return final_results_heap; 
        }


        if (nprobe == 0) nprobe = 1; 
        if (nprobe > num_ivf_clusters) nprobe = num_ivf_clusters;

        std::vector<int> nprobe_cluster_indices;
        // Stage 1: Find nprobe closest IVF centroids (L2 distance between query and IVF centroids)
        std::vector<std::pair<float, int>> all_ivf_centroid_distances;
        all_ivf_centroid_distances.reserve(num_ivf_clusters);

        std::vector<pthread_t> threads_stage1(num_threads > 1 ? num_threads - 1 : 0);
        std::vector<IVFPQV1_SearchIVFCentroidsArgs> args_stage1(num_threads);
        std::vector<std::vector<std::pair<float, int>>> per_thread_centroid_distances(num_threads);

        size_t items_per_thread_s1 = num_ivf_clusters / num_threads;
        size_t remainder_s1 = num_ivf_clusters % num_threads;
        size_t current_start_idx_s1 = 0;

        for (size_t i = 0; i < num_threads; ++i) {
            args_stage1[i].ivfpq_instance = this;
            args_stage1[i].query_ptr = query;
            args_stage1[i].start_centroid_idx = current_start_idx_s1;
            size_t chunk = items_per_thread_s1 + (i < remainder_s1 ? 1 : 0);
            args_stage1[i].end_centroid_idx = current_start_idx_s1 + chunk;
            args_stage1[i].thread_centroid_distances_output_ptr = &per_thread_centroid_distances[i];

            if (i < num_threads - 1 && num_threads > 1) {
                if (chunk > 0) pthread_create(&threads_stage1[i], nullptr, search_ivf_centroids_worker_v1_static, &args_stage1[i]);
            } else {
                if (chunk > 0) search_ivf_centroids_worker_v1_static(&args_stage1[i]);
            }
            current_start_idx_s1 += chunk;
        }

        for (size_t i = 0; i < num_threads - 1 && num_threads > 1; ++i) {
             if (args_stage1[i].start_centroid_idx < args_stage1[i].end_centroid_idx) {
                pthread_join(threads_stage1[i], nullptr);
             }
        }

        for (size_t i = 0; i < num_threads; ++i) {
            all_ivf_centroid_distances.insert(all_ivf_centroid_distances.end(),
                                              per_thread_centroid_distances[i].begin(),
                                              per_thread_centroid_distances[i].end());
        }

        std::sort(all_ivf_centroid_distances.begin(), all_ivf_centroid_distances.end());

        nprobe_cluster_indices.reserve(nprobe);
        for (size_t i = 0; i < std::min(nprobe, all_ivf_centroid_distances.size()); ++i) {
            nprobe_cluster_indices.push_back(all_ivf_centroid_distances[i].second);
        }

        if (nprobe_cluster_indices.empty() && num_ivf_clusters > 0 && nprobe > 0) {
            std::cerr << "IVFPQ_V1 search: No IVF candidate clusters found after Stage 1. Returning empty." << std::endl;
            return final_results_heap;
        }
        
        // Stage 2: Search within selected nprobe clusters using PQ codes (L2 distance)
        bool perform_reranking = (rerank_k_candidates > k && base_data_for_reranking != nullptr);
        size_t k_for_pq_stage = perform_reranking ? rerank_k_candidates : k;
        if (k_for_pq_stage == 0 && k > 0) k_for_pq_stage = k;
        if (k_for_pq_stage == 0) return final_results_heap;

        std::priority_queue<std::pair<float, uint32_t>> merged_pq_candidates_heap;

        std::vector<pthread_t> threads_stage2(num_threads > 1 ? num_threads - 1 : 0);
        std::vector<IVFPQV1_SearchListsPQArgs> args_stage2(num_threads);
        std::vector<std::priority_queue<std::pair<float, uint32_t>>> per_thread_top_k_pq(num_threads);

        std::vector<float> query_pq_dist_table;
        pq_quantizer->compute_query_distance_table(query, query_pq_dist_table); 

        if (query_pq_dist_table.empty()) {
            std::cerr << "IVFPQ_V1 search: Failed to compute PQ distance table. Returning empty." << std::endl;
            return final_results_heap;
        }

        size_t num_candidate_clusters_to_search = nprobe_cluster_indices.size();
         if (num_candidate_clusters_to_search == 0 && ivf_active) {
            return final_results_heap;
        }

        size_t clusters_per_thread_s2 = num_candidate_clusters_to_search / num_threads;
        size_t remainder_s2 = num_candidate_clusters_to_search % num_threads;
        size_t current_cluster_offset_s2 = 0; 

        for (size_t i = 0; i < num_threads; ++i) {
            args_stage2[i].ivfpq_instance = this;
            args_stage2[i].k_to_collect = k_for_pq_stage; 
            args_stage2[i].candidate_cluster_indices_ptr = &nprobe_cluster_indices;
            args_stage2[i].task_start_idx_in_candidates = current_cluster_offset_s2;
            size_t chunk = clusters_per_thread_s2 + (i < remainder_s2 ? 1 : 0);
            args_stage2[i].task_end_idx_in_candidates = current_cluster_offset_s2 + chunk;
            args_stage2[i].thread_top_k_output_ptr = &per_thread_top_k_pq[i];
            args_stage2[i].pq_quantizer_ptr = pq_quantizer;
            args_stage2[i].query_pq_dist_table_ptr = &query_pq_dist_table;

            if (i < num_threads - 1 && num_threads > 1) {
                if (args_stage2[i].task_start_idx_in_candidates < args_stage2[i].task_end_idx_in_candidates) {
                    pthread_create(&threads_stage2[i], nullptr, search_lists_pq_worker_v1_static, &args_stage2[i]);
                }
            } else { 
                if (args_stage2[i].task_start_idx_in_candidates < args_stage2[i].task_end_idx_in_candidates) {
                    search_lists_pq_worker_v1_static(&args_stage2[i]);
                }
            }
            current_cluster_offset_s2 += chunk;
        }

        for (size_t i = 0; i < num_threads - 1 && num_threads > 1; ++i) {
            if (args_stage2[i].task_start_idx_in_candidates < args_stage2[i].task_end_idx_in_candidates) {
                pthread_join(threads_stage2[i], nullptr);
            }
        }

        for (size_t i = 0; i < num_threads; ++i) {
            while(!per_thread_top_k_pq[i].empty()){
                std::pair<float, uint32_t> cand = per_thread_top_k_pq[i].top();
                per_thread_top_k_pq[i].pop();
                if(merged_pq_candidates_heap.size() < k_for_pq_stage || cand.first < merged_pq_candidates_heap.top().first){
                    if(merged_pq_candidates_heap.size() == k_for_pq_stage) merged_pq_candidates_heap.pop();
                    merged_pq_candidates_heap.push(cand);
                }
            }
        }

        // Stage 3: Reranking (if enabled)
        if (perform_reranking) {
            std::vector<std::pair<float, uint32_t>> pq_candidates_vec;
            pq_candidates_vec.reserve(merged_pq_candidates_heap.size());
            while(!merged_pq_candidates_heap.empty()){
                pq_candidates_vec.push_back(merged_pq_candidates_heap.top());
                merged_pq_candidates_heap.pop();
            }
            // pq_candidates_vec is sorted by L2 dist (descending). We need ascending for std::sort.
            std::sort(pq_candidates_vec.begin(), pq_candidates_vec.end(), 
                      [](const std::pair<float, uint32_t>&a, const std::pair<float, uint32_t>&b){ return a.first < b.first; });


            for(const auto& pq_cand_pair : pq_candidates_vec){
                uint32_t original_idx = pq_cand_pair.second;
                const float* original_vec = base_data_for_reranking + static_cast<size_t>(original_idx) * vecdim;
                float rerank_dist = compute_distance_reranking(query, original_vec, vecdim); // IP distance

                if (final_results_heap.size() < k || rerank_dist < final_results_heap.top().first) {
                    if (final_results_heap.size() == k) final_results_heap.pop();
                    final_results_heap.push({rerank_dist, original_idx});
                }
            }
            return final_results_heap;
        } else {
            return merged_pq_candidates_heap; // Contains L2-based PQ candidates
        }
    }
};