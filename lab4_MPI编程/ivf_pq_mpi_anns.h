#ifndef IVF_PQ_MPI_ANNS_H
#define IVF_PQ_MPI_ANNS_H

#include <mpi.h>
#include <vector>
#include <string>
#include <queue>
#include <algorithm> // For std::random_shuffle, std::min, std::sort, std::reverse, std::unique
#include <iostream>
#include <cmath>     // For sqrt, pow if needed, fabsf
#include <limits>    // For std::numeric_limits
#include <omp.h>
#include <numeric>   // For std::iota
#include <set>       // Added for std::set in search
#include <memory>    // For std::unique_ptr
#include <random>    // For std::default_random_engine and std::shuffle with C++11
#include <chrono>    // For seeding random number generator

#include "pq_anns.h" // Assuming ProductQuantizer is defined here
// It seems compute_l2_sq_neon is in pq_anns.h, let's ensure it's usable or provide a generic one.

// Helper function for k-means, typically run by rank 0 for IVF centroids
void kmeans_for_ivf_centroids_mpi(
    const float* train_data, size_t n_train_vectors, size_t vecdim,
    size_t n_clusters, int n_iters, std::vector<float>& centroids_out, int num_threads_for_kmeans, bool verbose_kmeans, int rank_for_log = 0) {
    
    if (rank_for_log == 0 && verbose_kmeans) {
        std::cout << "IVFPQMPI Kmeans: Starting IVF centroid k-means. N_train=" << n_train_vectors
                  << ", D=" << vecdim << ", K=" << n_clusters << ", Iters=" << n_iters 
                  << ", OMP_threads=" << num_threads_for_kmeans << std::endl;
    }

    if (n_train_vectors == 0 || n_clusters == 0 || vecdim == 0) {
        centroids_out.clear();
        if (rank_for_log == 0 && verbose_kmeans) std::cerr << "IVFPQMPI Kmeans: Invalid parameters for k-means, exiting." << std::endl;
        return;
    }
    centroids_out.assign(n_clusters * vecdim, 0.0f); // Initialize with zeros
    
    std::vector<size_t> assignments(n_train_vectors);
    std::vector<float> min_dists_sq(n_train_vectors, std::numeric_limits<float>::max());

    std::vector<size_t> initial_ids(n_train_vectors);
    std::iota(initial_ids.begin(), initial_ids.end(), 0);
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::shuffle(initial_ids.begin(), initial_ids.end(), std::default_random_engine(seed));

    size_t n_init_centroids = std::min(n_train_vectors, n_clusters);
    for (size_t i = 0; i < n_init_centroids; ++i) {
        memcpy(centroids_out.data() + i * vecdim,
               train_data + initial_ids[i] * vecdim,
               vecdim * sizeof(float));
    }
    // Remaining centroids (if n_clusters > n_init_centroids) are already zero

    std::vector<float> new_centroids_sum(n_clusters * vecdim);
    std::vector<size_t> cluster_counts(n_clusters);
    std::vector<float> old_centroid_check(vecdim); // For convergence check

    bool converged = false;
    for (int iter = 0; iter < n_iters && !converged; ++iter) {
        if (rank_for_log == 0 && verbose_kmeans && (iter % 5 == 0 || iter == n_iters -1) ) {
            std::cout << "IVFPQMPI Kmeans: IVF Centroid Iteration " << iter << std::endl;
        }
        // Assignment step
        #pragma omp parallel for num_threads(num_threads_for_kmeans) schedule(dynamic)
        for (size_t i = 0; i < n_train_vectors; ++i) {
            const float* point = train_data + i * vecdim;
            float current_min_dist_sq = std::numeric_limits<float>::max();
            size_t best_cluster_idx = 0; // Default to 0
            if (n_clusters > 0) { // Ensure there's at least one cluster
                 for (size_t c = 0; c < n_clusters; ++c) {
                    float dist_sq = compute_l2_sq_neon(point, centroids_out.data() + c * vecdim, vecdim);
                    if (dist_sq < current_min_dist_sq) {
                        current_min_dist_sq = dist_sq;
                        best_cluster_idx = c;
                    }
                }
            }
            assignments[i] = best_cluster_idx;
            min_dists_sq[i] = current_min_dist_sq;
        }

        // Update step
        std::fill(new_centroids_sum.begin(), new_centroids_sum.end(), 0.0f);
        std::fill(cluster_counts.begin(), cluster_counts.end(), 0);

        for (size_t i = 0; i < n_train_vectors; ++i) {
            size_t cluster_idx = assignments[i];
            cluster_counts[cluster_idx]++;
            const float* point = train_data + i * vecdim;
            float* target_sum = new_centroids_sum.data() + cluster_idx * vecdim;
            for (size_t d = 0; d < vecdim; ++d) { // SIMD could be used here too
                target_sum[d] += point[d];
            }
        }
        
        bool iteration_changed_centroids = false;
        size_t n_empty_clusters_this_iter = 0;

        for (size_t c = 0; c < n_clusters; ++c) {
            float* current_centroid_ptr = centroids_out.data() + c * vecdim;
            if (cluster_counts[c] > 0) {
                std::copy(current_centroid_ptr, current_centroid_ptr + vecdim, old_centroid_check.data());
                float inv_count = 1.0f / static_cast<float>(cluster_counts[c]);
                float* sum_ptr = new_centroids_sum.data() + c * vecdim;
                for (size_t d = 0; d < vecdim; ++d) {
                    float new_val = sum_ptr[d] * inv_count;
                    if (std::abs(new_val - old_centroid_check[d]) > 1e-6f) { // Check against old_centroid_check
                        iteration_changed_centroids = true;
                    }
                    current_centroid_ptr[d] = new_val;
                }
            } else {
                n_empty_clusters_this_iter++;
            }
        }
        
        // Handle empty clusters (simplified: re-initialize from farthest points from existing centroids)
        if (n_empty_clusters_this_iter > 0 && n_train_vectors >= n_clusters) {
            if (rank_for_log == 0 && verbose_kmeans) std::cout << "IVFPQMPI Kmeans: Handling " << n_empty_clusters_this_iter << " empty clusters." << std::endl;
            std::vector<std::pair<float, size_t>> dist_idx_pairs_for_empty(n_train_vectors);
            #pragma omp parallel for num_threads(num_threads_for_kmeans)
            for(size_t i=0; i<n_train_vectors; ++i) {
                 dist_idx_pairs_for_empty[i] = {min_dists_sq[i], i}; // min_dists_sq was updated in assignment
            }
            std::sort(dist_idx_pairs_for_empty.rbegin(), dist_idx_pairs_for_empty.rend()); // Sort by distance descending

            size_t filled_empty_count = 0;
            std::vector<bool> point_selected_for_empty_cluster(n_train_vectors, false);

            for (size_t c = 0; c < n_clusters && filled_empty_count < n_empty_clusters_this_iter; ++c) {
                if (cluster_counts[c] == 0) { 
                    size_t point_to_use_idx = n_train_vectors; // Sentinel for not found
                    for(size_t p_idx = 0; p_idx < n_train_vectors; ++p_idx) { // Iterate through sorted points
                        size_t candidate_original_idx = dist_idx_pairs_for_empty[p_idx].second;
                        if (!point_selected_for_empty_cluster[candidate_original_idx]) {
                            point_to_use_idx = candidate_original_idx;
                            point_selected_for_empty_cluster[candidate_original_idx] = true; 
                            break;
                        }
                    }
                    if (point_to_use_idx < n_train_vectors) {
                         memcpy(centroids_out.data() + c * vecdim,
                               train_data + point_to_use_idx * vecdim,
                               vecdim * sizeof(float));
                        iteration_changed_centroids = true; 
                        filled_empty_count++;
                    } else { // Not enough unique points, or all points already used
                        // Keep it zero or assign a random point if desperate, here just keep as zero (already initialized)
                         if (rank_for_log == 0 && verbose_kmeans) std::cerr << "IVFPQMPI Kmeans: Could not find a unique point for empty cluster " << c << std::endl;
                    }
                }
            }
        }
        if (!iteration_changed_centroids && iter > 0) { // If no centroid moved significantly
            converged = true;
            if (rank_for_log == 0 && verbose_kmeans) std::cout << "IVFPQMPI Kmeans: IVF Centroid K-means converged at iteration " << iter << std::endl;
        }
    }
    if (rank_for_log == 0 && verbose_kmeans && !converged) {
        std::cout << "IVFPQMPI Kmeans: IVF Centroid K-means finished max iterations without full convergence." << std::endl;
    }
}


inline float l2_distance_sq_generic(const float* a, const float* b, size_t dim) {
    float dist_sq = 0.0f;
    for (size_t i = 0; i < dim; ++i) {
        float diff = a[i] - b[i];
        dist_sq += diff * diff;
    }
    return dist_sq;
}


class IVFPQMPIIndex {
public:
    MPI_Comm comm_;
    int rank_;
    int world_size_;

    size_t vecdim_;
    size_t num_ivf_clusters_;
    size_t pq_nsub_;
    int pq_kmeans_iters_; 
    int ivf_kmeans_iters_;
    int num_omp_threads_;

    std::vector<float> ivf_centroids_; 
    std::unique_ptr<ProductQuantizer> pq_;

    std::vector<std::vector<uint32_t>> local_inverted_lists_global_indices_; 
    std::vector<std::vector<uint8_t*>> local_pq_codes_for_lists_;      

    const float* full_base_data_ptr_; 
    size_t total_base_vectors_;

    const float* local_base_data_ptr_member_;
    size_t num_local_base_vectors_member_;
    std::vector<int> local_data_global_indices_member_;

    bool verbose_ = false;

public:
    IVFPQMPIIndex(MPI_Comm comm,
                  const float* full_base_data, 
                  const float* local_base_data_ptr_param, 
                  size_t num_local_base_vectors_param,   
                  const std::vector<int>& local_data_global_indices_param, 
                  size_t total_base_vectors_in, 
                  size_t vecdim_param,
                  size_t num_ivf_clusters_param,
                  size_t pq_nsub_param,
                  int pq_kmeans_iters, // This parameter seems unused in current ProductQuantizer constructor
                  int ivf_kmeans_iters,
                  int num_omp_threads,
                  bool verbose = false)
        : comm_(comm), vecdim_(vecdim_param), num_ivf_clusters_(num_ivf_clusters_param),
          pq_nsub_(pq_nsub_param), pq_kmeans_iters_(pq_kmeans_iters),
          ivf_kmeans_iters_(ivf_kmeans_iters), num_omp_threads_(num_omp_threads),
          full_base_data_ptr_(full_base_data), total_base_vectors_(total_base_vectors_in),
          local_base_data_ptr_member_(local_base_data_ptr_param),
          num_local_base_vectors_member_(num_local_base_vectors_param),
          local_data_global_indices_member_(local_data_global_indices_param),
          verbose_(verbose)
    {
        MPI_Comm_rank(comm_, &rank_);
        MPI_Comm_size(comm_, &world_size_);
        omp_set_num_threads(num_omp_threads_);

        // <<< 修改: 使用 new T(...) 替换 std::make_unique<T>(...)
        pq_.reset(new ProductQuantizer(
            nullptr, 
            0, 
            (vecdim_ == 0 ? 1 : vecdim_), // Ensure non-zero dimension for dummy PQ
            (pq_nsub_ == 0 ? 1 : pq_nsub_), // Ensure non-zero nsub for dummy PQ
            1.0 // train_ratio, not directly used by PQ constructor if base is nullptr
        ));

        if (num_ivf_clusters_ > 0) { // Only resize if IVF is active
            local_inverted_lists_global_indices_.resize(num_ivf_clusters_);
            local_pq_codes_for_lists_.resize(num_ivf_clusters_);
        } else { // If no IVF clusters, conceptually one list (list 0)
            local_inverted_lists_global_indices_.resize(1);
            local_pq_codes_for_lists_.resize(1);
        }
    }

    ~IVFPQMPIIndex() {
        for (auto& cluster_codes : local_pq_codes_for_lists_) {
            for (uint8_t* code : cluster_codes) {
                delete[] code; 
            }
        }
        // pq_ unique_ptr will auto-delete
    }

    void build(double pq_train_ratio = 0.1) { 
        // --- Stage 1: IVF Centroid Training (Rank 0) and Broadcast ---
        if (num_ivf_clusters_ > 0) { // Only if IVF is active
            if (rank_ == 0) {
                if (verbose_) std::cout << "IVFPQMPI: Rank 0 starting IVF centroid training. Num_clusters=" << num_ivf_clusters_ << std::endl;
                size_t ivf_train_n = total_base_vectors_; // Use all data or a large subset for IVF training
                if (total_base_vectors_ > 256000 && num_ivf_clusters_ > 0) { // Heuristic: if base is very large, sample
                    ivf_train_n = std::max(num_ivf_clusters_ * 100, (size_t)256000);
                    ivf_train_n = std::min(ivf_train_n, total_base_vectors_);
                }
                if (ivf_train_n < num_ivf_clusters_ && total_base_vectors_ >= num_ivf_clusters_) ivf_train_n = num_ivf_clusters_; // Ensure at least K points if possible
                
                std::vector<float> ivf_training_data;
                if (ivf_train_n > 0 && total_base_vectors_ > 0 && full_base_data_ptr_ != nullptr) {
                    if (ivf_train_n == total_base_vectors_) { // Use all data
                        // No copy needed if full_base_data_ptr_ can be used directly by kmeans_for_ivf_centroids_mpi
                        // However, kmeans_for_ivf_centroids_mpi expects non-const, so a copy might be safer or modify kmeans
                        // For simplicity here, let's assume full_base_data_ptr_ is usable or kmeans handles const
                        // If kmeans needs to shuffle or select, it should take a copy or indices.
                        // The provided kmeans_for_ivf_centroids_mpi shuffles indices from a copy.
                        ivf_training_data.assign(full_base_data_ptr_, full_base_data_ptr_ + ivf_train_n * vecdim_);

                    } else { // Sample a subset
                        ivf_training_data.resize(ivf_train_n * vecdim_);
                        std::vector<size_t> indices(total_base_vectors_);
                        std::iota(indices.begin(), indices.end(), 0);
                        unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
                        std::shuffle(indices.begin(), indices.end(), std::default_random_engine(seed));
                        for (size_t i = 0; i < ivf_train_n; ++i) {
                            memcpy(ivf_training_data.data() + i * vecdim_,
                                   full_base_data_ptr_ + indices[i] * vecdim_,
                                   vecdim_ * sizeof(float));
                        }
                    }
                     if (verbose_) std::cout << "IVFPQMPI: Rank 0 prepared IVF training data. N=" << ivf_train_n << std::endl;
                     kmeans_for_ivf_centroids_mpi(ivf_training_data.data(), ivf_train_n, vecdim_,
                                             num_ivf_clusters_, ivf_kmeans_iters_,
                                             ivf_centroids_, num_omp_threads_, verbose_, rank_);
                     if (verbose_) std::cout << "IVFPQMPI: Rank 0 finished IVF centroid training. Centroids size: " << ivf_centroids_.size() << std::endl;

                } else {
                    if (verbose_) std::cerr << "IVFPQMPI: Rank 0 cannot prepare IVF training data (N_total=" << total_base_vectors_ << ")." << std::endl;
                    ivf_centroids_.clear(); // Ensure it's empty if training fails
                }
            }
            // Broadcast IVF centroids
            size_t ivf_centroids_size = (rank_ == 0) ? ivf_centroids_.size() : 0;
            MPI_Bcast(&ivf_centroids_size, 1, MPI_UNSIGNED_LONG, 0, comm_);
            if (rank_ != 0) {
                ivf_centroids_.resize(ivf_centroids_size);
            }
            if (ivf_centroids_size > 0) {
                MPI_Bcast(ivf_centroids_.data(), ivf_centroids_size, MPI_FLOAT, 0, comm_);
            }
            if (verbose_) std::cout << "IVFPQMPI: Rank " << rank_ << " received IVF centroids. Size: " << ivf_centroids_.size() << std::endl;
            if (ivf_centroids_.empty() && num_ivf_clusters_ > 0) {
                 if (rank_ == 0 && verbose_) std::cerr << "IVFPQMPI: CRITICAL - IVF centroids are empty after broadcast, but num_ivf_clusters > 0. Search will likely fail." << std::endl;
                 // Potentially abort or handle this error to prevent further issues.
                 // For now, execution will continue, but search quality will be zero.
            }
        } else { // num_ivf_clusters_ == 0
            if (verbose_) std::cout << "IVFPQMPI: Rank " << rank_ << " IVF is disabled (num_ivf_clusters_ = 0)." << std::endl;
            ivf_centroids_.clear(); // Ensure it's clear if IVF is off
        }


        // --- Stage 2: PQ Training (Rank 0) and Broadcast ---
        size_t pq_train_actual_n = static_cast<size_t>(total_base_vectors_ * pq_train_ratio);
        if (pq_train_actual_n == 0 && total_base_vectors_ > 0) pq_train_actual_n = std::min((size_t)1000, total_base_vectors_); 
        if (pq_train_actual_n > total_base_vectors_) pq_train_actual_n = total_base_vectors_;
        
        std::vector<float> pq_training_data_flat;
        if (rank_ == 0) {
            if (verbose_) std::cout << "IVFPQMPI: Rank 0 preparing PQ training data. Requested ratio: " << pq_train_ratio << ", actual N: " << pq_train_actual_n << std::endl;
            if (pq_train_actual_n > 0 && total_base_vectors_ > 0 && full_base_data_ptr_ != nullptr) {
                pq_training_data_flat.resize(pq_train_actual_n * vecdim_);
                std::vector<size_t> indices(total_base_vectors_);
                std::iota(indices.begin(), indices.end(), 0);
                
                unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
                std::shuffle(indices.begin(), indices.end(), std::default_random_engine(seed)); 
                
                for (size_t i = 0; i < pq_train_actual_n; ++i) {
                    memcpy(pq_training_data_flat.data() + i * vecdim_,
                           full_base_data_ptr_ + indices[i] * vecdim_,
                           vecdim_ * sizeof(float));
                }
            } else if (pq_train_actual_n > 0) { // total_base_vectors_ is 0 or full_base_data_ptr_ is null
                 pq_train_actual_n = 0; // Cannot create training data
                 if(verbose_) std::cerr << "IVFPQMPI: Rank 0 cannot prepare PQ training data as base data is insufficient or null." << std::endl;
            }
        }

        MPI_Bcast(&pq_train_actual_n, 1, MPI_UNSIGNED_LONG, 0, comm_);
        if (rank_ != 0 && pq_train_actual_n > 0) { 
            pq_training_data_flat.resize(pq_train_actual_n * vecdim_);
        }
        if (pq_train_actual_n > 0) { 
             MPI_Bcast(pq_training_data_flat.data(), pq_train_actual_n * vecdim_, MPI_FLOAT, 0, comm_);
        }
        if (verbose_) std::cout << "IVFPQMPI: Rank " << rank_ << " received PQ training data. Size: " << pq_train_actual_n << std::endl;

        if (pq_train_actual_n > 0 && vecdim_ > 0 && pq_nsub_ > 0 && vecdim_ % pq_nsub_ == 0) {
            if (verbose_) std::cout << "IVFPQMPI: Rank " << rank_ << " training PQ with N=" << pq_train_actual_n << ", D=" << vecdim_ << ", M=" << pq_nsub_ << std::endl;
            // <<< 修改: 使用 new T(...) 替换 std::make_unique<T>(...)
            pq_.reset(new ProductQuantizer(pq_training_data_flat.data(), pq_train_actual_n, vecdim_, pq_nsub_, 1.0)); // train_ratio is 1.0 here as data is already sampled
        } else {
            if (rank_ == 0 && verbose_ && (vecdim_ == 0 || pq_nsub_ == 0 || (vecdim_ % pq_nsub_ !=0 && pq_nsub_ > 0) )) { // Only log error if params were bad
                std::cerr << "IVFPQMPI: PQ training skipped due to invalid parameters (train_n="
                          << pq_train_actual_n << ", D=" << vecdim_ << ", M=" << pq_nsub_
                          << ", D%M=" << (pq_nsub_ > 0 ? vecdim_ % pq_nsub_ : -1) << ")." << std::endl;
            }
            // pq_ remains the dummy one from constructor if training is skipped
        }

        if (verbose_) std::cout << "IVFPQMPI: Rank " << rank_ << " finished PQ setup. PQ nbase: " << (pq_ ? pq_->get_nbase() : 0) << std::endl;

        for(auto& list_indices : local_inverted_lists_global_indices_) list_indices.clear();
        for(auto& list_codes : local_pq_codes_for_lists_) {
            for(auto* code_ptr : list_codes) delete[] code_ptr;
            list_codes.clear();
        }

        #pragma omp parallel for schedule(dynamic)
        for (size_t i = 0; i < num_local_base_vectors_member_; ++i) {
            const float* current_vector = local_base_data_ptr_member_ + i * vecdim_;
            int global_original_idx = local_data_global_indices_member_[i];
            
            int best_ivf_cluster = 0; 
            if (num_ivf_clusters_ > 0 && !ivf_centroids_.empty()) { // Check ivf_centroids_ is not empty
                float min_dist_sq = std::numeric_limits<float>::max();
                best_ivf_cluster = -1; 
                for (size_t c = 0; c < num_ivf_clusters_; ++c) {
                    // Ensure c * vecdim_ does not go out of bounds for ivf_centroids_
                    if ((c + 1) * vecdim_ <= ivf_centroids_.size()) {
                        float dist = compute_l2_sq_neon(current_vector, ivf_centroids_.data() + c * vecdim_, vecdim_);
                        if (dist < min_dist_sq) {
                            min_dist_sq = dist;
                            best_ivf_cluster = c;
                        }
                    } else {
                        // This case should ideally not happen if ivf_centroids_ is correctly sized
                        if (verbose_ && rank_ == 0) std::cerr << "IVFPQMPI Build: ivf_centroids_ access out of bounds for cluster " << c << std::endl;
                        break; 
                    }
                }
                 if (best_ivf_cluster == -1 && num_ivf_clusters_ > 0) {
                     best_ivf_cluster = 0; // Fallback if somehow not assigned
                     if (verbose_ && rank_ == 0) std::cerr << "IVFPQMPI Build: Vector " << global_original_idx << " could not be assigned, falling back to cluster 0." << std::endl;
                 }
            } else if (num_ivf_clusters_ == 0) {
                best_ivf_cluster = 0; // All data goes to the single conceptual list 0
            }


            if (static_cast<size_t>(best_ivf_cluster) < local_inverted_lists_global_indices_.size()) {
                // --- 这里是关键修改 ---
                uint8_t* pq_code = nullptr;
                // 检查PQ是否被成功训练
                if (pq_ && pq_->get_nbase() > 0 && pq_nsub_ > 0) { 
                    pq_code = new uint8_t[pq_nsub_];
                    // 使用我们新的、正确的方法来为当前向量编码
                    pq_->encode_vector(current_vector, pq_code);
                }
                
                #pragma omp critical
        {
            local_inverted_lists_global_indices_[best_ivf_cluster].push_back(global_original_idx);
            // 因为现在 pq_code 总是被成功创建，所以我们总是 push_back 有效的 code
            // 不再需要处理 nullptr 的情况
            if (pq_code) { 
                 local_pq_codes_for_lists_[best_ivf_cluster].push_back(pq_code);
            } else {
                 // 这个分支理论上不应该进入了，但为了安全可以保留一个警告
                 // 并且 push nullptr 以保持数据对齐，尽管这仍是不理想的
                 local_pq_codes_for_lists_[best_ivf_cluster].push_back(nullptr);
                 if(verbose_ && rank_ == 0) std::cerr << "IVFPQMPI: CRITICAL - Failed to generate PQ code for " << global_original_idx << std::endl;
            }
        }
            } else if (num_ivf_clusters_ > 0) { 
                 #pragma omp critical
                 if(verbose_) std::cerr << "IVFPQMPI: Rank " << rank_ << " failed to assign vector " << global_original_idx 
                                       << " to any IVF cluster or cluster index " << best_ivf_cluster << " is out of bounds." << std::endl;
            }
        }
        if (verbose_) std::cout << "IVFPQMPI: Rank " << rank_ << " finished populating local inverted lists." << std::endl;
    }

    std::priority_queue<std::pair<float, uint32_t>> search(
        const float* query_vector, size_t k, size_t nprobe, size_t rerank_k_adc) {
        
        std::priority_queue<std::pair<float, uint32_t>, std::vector<std::pair<float, uint32_t>>, std::greater<std::pair<float, uint32_t>>> local_top_results_min_heap; 

        std::vector<std::pair<float, size_t>> query_to_centroid_dists;
        if (num_ivf_clusters_ > 0 && !ivf_centroids_.empty()) { // Check ivf_centroids_ is not empty
            for (size_t i = 0; i < num_ivf_clusters_; ++i) {
                // Ensure i * vecdim_ does not go out of bounds for ivf_centroids_
                if ((i + 1) * vecdim_ <= ivf_centroids_.size()) {
                    // <<< 修改: 使用 compute_l2_sq_neon
                    // <<< 修改: Explicitly create std::pair for push_back with C++11
                    query_to_centroid_dists.push_back(std::make_pair(compute_l2_sq_neon(query_vector, ivf_centroids_.data() + i * vecdim_, vecdim_), i));
                } else {
                     if (verbose_ && rank_ == 0) std::cerr << "IVFPQMPI Search: ivf_centroids_ access out of bounds for cluster " << i << " during nprobe selection." << std::endl;
                     break;
                }
            }
            std::sort(query_to_centroid_dists.begin(), query_to_centroid_dists.end());
        } else if (num_ivf_clusters_ > 0 && ivf_centroids_.empty() && verbose_ && rank_ == 0) {
            std::cerr << "IVFPQMPI Search: IVF is enabled but centroids are empty. Nprobe selection will be ineffective." << std::endl;
        }
        
        size_t actual_nprobe = 0;
        if (num_ivf_clusters_ > 0) {
            if (!ivf_centroids_.empty() && !query_to_centroid_dists.empty()) { // Only use nprobe if centroids and dists are valid
                actual_nprobe = std::min(nprobe, query_to_centroid_dists.size());
                actual_nprobe = std::min(actual_nprobe, num_ivf_clusters_); // Cap by total clusters
            } else {
                actual_nprobe = 0; // Cannot determine probes if centroids are missing
                 if (verbose_ && rank_ == 0 && num_ivf_clusters_ > 0) std::cerr << "IVFPQMPI Search: actual_nprobe set to 0 due to missing IVF centroids or dists." << std::endl;
            }
        } else { // No IVF clusters (num_ivf_clusters_ == 0)
            actual_nprobe = 1; // Search the single conceptual list 0
        }
        if (actual_nprobe == 0 && num_ivf_clusters_ > 0) { // If IVF was intended but no probes selected
             if (verbose_ && rank_ == 0) std::cerr << "IVFPQMPI Search: actual_nprobe is 0, ADC search will be skipped for IVF." << std::endl;
        }


        std::vector<float> query_pq_dist_table;
        if (pq_ && pq_nsub_ > 0 && pq_->get_nbase() > 0) { 
            pq_->compute_query_distance_table(query_vector, query_pq_dist_table);
        }

        std::priority_queue<std::pair<float, uint32_t>> local_adc_candidates_max_heap;

        // Proceed if (IVF is active AND we have probes AND PQ table is ready for ADC) OR (no IVF AND PQ table is ready for ADC)
        bool can_do_adc_search = !query_pq_dist_table.empty() && pq_ && pq_->get_nbase() > 0 && pq_->is_trained();

        if (actual_nprobe > 0 && can_do_adc_search) {
            size_t num_clusters_to_search = actual_nprobe; // Already determined based on IVF or no-IVF

            #pragma omp parallel for schedule(dynamic)
            for (size_t i = 0; i < num_clusters_to_search; ++i) {
                int cluster_idx;
                if (num_ivf_clusters_ > 0) {
                    if (i < query_to_centroid_dists.size()) { // query_to_centroid_dists might be smaller than actual_nprobe if nprobe > num_ivf_clusters
                        cluster_idx = query_to_centroid_dists[i].second;
                    } else {
                         if (verbose_ && rank_ == 0) std::cerr << "IVFPQMPI Search: Index i=" << i << " out of bounds for query_to_centroid_dists (size=" << query_to_centroid_dists.size() << ")" << std::endl;
                        continue; 
                    }
                } else {
                    cluster_idx = 0; 
                }

                if (static_cast<size_t>(cluster_idx) >= local_inverted_lists_global_indices_.size() ||
                    static_cast<size_t>(cluster_idx) >= local_pq_codes_for_lists_.size() ) continue;

                const auto& list_global_indices = local_inverted_lists_global_indices_[cluster_idx];
                const auto& list_pq_codes = local_pq_codes_for_lists_[cluster_idx];

                for (size_t j = 0; j < list_global_indices.size(); ++j) {
                    uint32_t global_idx = list_global_indices[j];
                    float dist_adc = std::numeric_limits<float>::max();

                    if (j < list_pq_codes.size() && list_pq_codes[j] != nullptr) { // Check code exists and is not null
                        const uint8_t* code = list_pq_codes[j];
                        dist_adc = pq_->compute_asymmetric_distance_sq_with_table(code, query_pq_dist_table);
                    } else {
                        // Code missing or null, cannot compute ADC distance for this item.
                        // This item should ideally not have been added or handled differently in build.
                        if (verbose_ && rank_ == 0) {
                            std::cerr << "IVFPQMPI Search: Skipping ADC for global_idx " << global_idx 
                                      << " in cluster " << cluster_idx << " due to missing/null PQ code." << std::endl;
                        }
                        continue; // Skip this item if no valid PQ code
                    }
                    
                    #pragma omp critical
                    {
                        if (local_adc_candidates_max_heap.size() < rerank_k_adc || dist_adc < local_adc_candidates_max_heap.top().first) {
                            local_adc_candidates_max_heap.push({dist_adc, global_idx}); // C++11 might need std::make_pair
                            if (local_adc_candidates_max_heap.size() > rerank_k_adc) {
                                local_adc_candidates_max_heap.pop();
                            }
                        }
                    }
                }
            }
        } else if (actual_nprobe > 0 && !can_do_adc_search && verbose_ && rank_ == 0) {
             std::cerr << "IVFPQMPI Search: Skipping ADC search phase because PQ distance table is not available or PQ not trained." << std::endl;
        }


        std::vector<std::pair<float, uint32_t>> collected_local_best;
        while(!local_adc_candidates_max_heap.empty()){
            collected_local_best.push_back(local_adc_candidates_max_heap.top());
            local_adc_candidates_max_heap.pop();
        }
        std::reverse(collected_local_best.begin(), collected_local_best.end()); 


        std::vector<std::pair<float, uint32_t>> all_processes_top_results_adc;
        
        int num_results_local = collected_local_best.size();
        
        std::vector<int> recv_counts_adc(world_size_);
        MPI_Gather(&num_results_local, 1, MPI_INT, recv_counts_adc.data(), 1, MPI_INT, 0, comm_);

        std::vector<int> displs_adc(world_size_);
        int total_gathered_adc = 0;
        if (rank_ == 0) {
            displs_adc[0] = 0;
            total_gathered_adc = recv_counts_adc[0];
            for (int j = 1; j < world_size_; ++j) {
                displs_adc[j] = displs_adc[j-1] + recv_counts_adc[j-1];
                total_gathered_adc += recv_counts_adc[j];
            }
            if (total_gathered_adc > 0) { // Only resize if there's data
                all_processes_top_results_adc.resize(total_gathered_adc);
            }
        }

        MPI_Datatype pair_type_adc;
        int blocklengths_adc[2] = {1, 1};
        MPI_Aint displacements_pair_adc[2];
        MPI_Datatype types_adc[2] = {MPI_FLOAT, MPI_UINT32_T};
        MPI_Aint float_lb_adc, float_extent_adc;
        MPI_Type_get_extent(MPI_FLOAT, &float_lb_adc, &float_extent_adc);
        displacements_pair_adc[0] = (MPI_Aint)0; // Explicit cast
        displacements_pair_adc[1] = float_extent_adc;
        MPI_Type_create_struct(2, blocklengths_adc, displacements_pair_adc, types_adc, &pair_type_adc);
        MPI_Type_commit(&pair_type_adc);

        
        MPI_Gatherv(collected_local_best.data(), num_results_local, pair_type_adc,
                (total_gathered_adc > 0 ? all_processes_top_results_adc.data() : nullptr), // MPI standard says recvbuf is "significant only at root"
                recv_counts_adc.data(), displs_adc.data(), pair_type_adc,
                0, comm_);
        
        
        MPI_Type_free(&pair_type_adc);


        std::priority_queue<std::pair<float, uint32_t>> final_top_k; 
        if (rank_ == 0) {
            std::sort(all_processes_top_results_adc.begin(), all_processes_top_results_adc.end(), 
                [](const std::pair<float, uint32_t>&a, const std::pair<float, uint32_t>&b){
                    if (a.first != b.first) return a.first < b.first;
                    return a.second < b.second; 
            });
            if (!all_processes_top_results_adc.empty()) { 
                all_processes_top_results_adc.erase(
                    std::unique(all_processes_top_results_adc.begin(), all_processes_top_results_adc.end(),
                        [](const std::pair<float, uint32_t>&a, const std::pair<float, uint32_t>&b){
                            return a.second == b.second; // Keep unique by global ID
                    }),
                    all_processes_top_results_adc.end()
                );
            }
            // After unique, sort again if reranking, as unique doesn't preserve order perfectly for reranking needs
            // if (needs_reranking) {
            //    std::sort(all_processes_top_results_adc.begin(), all_processes_top_results_adc.end(), ...);
            // }
            // The current sort is by distance, which is fine for taking top N for reranking.

            bool needs_reranking = (rerank_k_adc > k && k > 0 && full_base_data_ptr_ != nullptr && total_base_vectors_ > 0 && !all_processes_top_results_adc.empty());
            if (needs_reranking) {
                 if (verbose_) std::cout << "IVFPQMPI: Rank 0 reranking " << std::min(all_processes_top_results_adc.size(), rerank_k_adc) << " ADC candidates (target k=" << k << ")" << std::endl;
                // Take only up to rerank_k_adc candidates for actual reranking
                size_t num_to_rerank = std::min(all_processes_top_results_adc.size(), rerank_k_adc);

                for (size_t rerank_idx = 0; rerank_idx < num_to_rerank; ++rerank_idx) {
                    const auto& cand_pair = all_processes_top_results_adc[rerank_idx];
                    uint32_t global_idx_to_rerank = cand_pair.second;
                    if (global_idx_to_rerank < total_base_vectors_) {
                        // <<< 修改: 使用 compute_l2_sq_neon
                        float exact_dist = compute_l2_sq_neon(query_vector, full_base_data_ptr_ + global_idx_to_rerank * vecdim_, vecdim_);
                        
                        if (final_top_k.size() < k || exact_dist < final_top_k.top().first) {
                            final_top_k.push(std::make_pair(exact_dist, global_idx_to_rerank)); // C++11 make_pair
                            if (final_top_k.size() > k) {
                                final_top_k.pop();
                            }
                        }
                    }
                }
            } else { 
                for (const auto& cand_pair : all_processes_top_results_adc) {
                     if (final_top_k.size() < k || cand_pair.first < final_top_k.top().first) {
                        final_top_k.push(cand_pair); // C++11 make_pair might be safer: std::make_pair(cand_pair.first, cand_pair.second)
                        if (final_top_k.size() > k) {
                            final_top_k.pop();
                        }
                    } else if (final_top_k.size() >= k && cand_pair.first >= final_top_k.top().first) {
                        break;
                    }
                }
            }
             if (verbose_) std::cout << "IVFPQMPI: Rank 0 final_top_k size: " << final_top_k.size() << std::endl;
        }
        return final_top_k; 
    }
};

#endif // IVF_PQ_MPI_ANNS_H