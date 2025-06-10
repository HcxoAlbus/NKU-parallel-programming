#pragma once

#include <vector>
#include <queue>
#include <cstdint>
#include <algorithm>
#include <limits>
#include <cmath>
#include <numeric>
#include <random>
#include <chrono>
#include <mpi.h>
#include <omp.h>
#include <memory>
#include <iostream>

#include "hnswlib/hnswlib/hnswlib.h"
#include "simd_anns.h"

using namespace hnswlib;

class HNSWIVFIndexMPI {
public:
    HNSWIVFIndexMPI(MPI_Comm comm,
                    const float* full_base_data_for_rank0_init,
                    const float* local_data_chunk_for_this_rank,
                    size_t num_vectors_in_local_chunk,
                    const std::vector<int>& local_data_global_indices,
                    size_t total_num_base_vectors,
                    size_t vector_dim,
                    size_t n_clusters,
                    int n_threads_omp,
                    size_t hnsw_M = 32,                    // 增加M参数提高连接性
                    size_t hnsw_ef_construction = 400,     // 增加ef_construction提高索引质量
                    int kmeans_iterations = 30)            // 增加聚类迭代次数
        : comm_(comm),
          local_base_data_ptr_(local_data_chunk_for_this_rank),
          num_local_base_vectors_(num_vectors_in_local_chunk),
          local_data_global_indices_(local_data_global_indices),
          total_base_vectors_(total_num_base_vectors),
          vector_dim_(vector_dim),
          num_target_clusters_(n_clusters),
          num_threads_omp_(std::max(1, n_threads_omp)),
          hnsw_M_(hnsw_M),
          hnsw_ef_construction_(hnsw_ef_construction),
          kmeans_max_iter_(kmeans_iterations) {

        MPI_Comm_rank(comm_, &rank_);
        MPI_Comm_size(comm_, &world_size_);

        omp_set_num_threads(num_threads_omp_);

        if (total_base_vectors_ == 0 || vector_dim_ == 0) {
            if (rank_ == 0) {
                std::cerr << "HNSWIVFIndexMPI: Error, total_base_vectors or vector_dim is zero." << std::endl;
            }
            num_target_clusters_ = 0;
            return;
        }
        
        if (num_target_clusters_ > total_base_vectors_) {
            if (rank_ == 0) {
                std::cerr << "HNSWIVFIndexMPI: Warning, num_clusters > total_base_vectors. Setting num_clusters = total_base_vectors." << std::endl;
            }
            num_target_clusters_ = total_base_vectors_;
        }

        centroids_data_.resize(num_target_clusters_ * vector_dim_);
        full_base_data_for_rank0_init_ptr_ = full_base_data_for_rank0_init;

        if (num_target_clusters_ > 0) {
            initialize_centroids_kmeans_plus_plus(full_base_data_for_rank0_init); // Rank 0 initializes, Bcasts
            run_kmeans_parallel_mpi(); // All ranks participate, centroids_data_ is updated and consistent
            build_inverted_lists_and_hnsw_indices_mpi(); // Each rank builds HNSW for its local data
        }
    }

    ~HNSWIVFIndexMPI() {
        // Clean up HNSW indices and spaces
        for (size_t i = 0; i < hnsw_indices_.size(); ++i) {
            if (hnsw_indices_[i]) {
                delete hnsw_indices_[i];
                hnsw_indices_[i] = nullptr;
            }
        }
        for (size_t i = 0; i < inner_product_spaces_.size(); ++i) {
            if (inner_product_spaces_[i]) {
                delete inner_product_spaces_[i];
                inner_product_spaces_[i] = nullptr;
            }
        }
    }

    std::priority_queue<std::pair<float, uint32_t>> search(
        const float* query, size_t k, size_t nprobe, size_t hnsw_ef = 200) {
        
        std::priority_queue<std::pair<float, uint32_t>> final_top_k;
        if (k == 0 || num_target_clusters_ == 0 || !query) return final_top_k;

        // 自适应调整nprobe：对于较小的k值，使用更多的探测簇
        size_t adaptive_nprobe = nprobe; // Keep user-defined nprobe, or apply adaptive logic if preferred
        // size_t adaptive_nprobe = std::max(nprobe, std::min(k * 2, num_target_clusters_ / 4));
        if (adaptive_nprobe == 0 && num_target_clusters_ > 0) adaptive_nprobe = 1; // Ensure at least one probe if possible
        if (adaptive_nprobe > num_target_clusters_) adaptive_nprobe = num_target_clusters_;


        // Stage 1: Find nprobe nearest centroids (优化：减少重复计算)
        std::vector<int> nprobe_cluster_indices;
        nprobe_cluster_indices.reserve(adaptive_nprobe);

        if (rank_ == 0) {
            std::vector<std::pair<float, int>> all_centroid_distances;
            all_centroid_distances.reserve(num_target_clusters_);
            
            // 使用SIMD优化的距离计算
            #pragma omp parallel for num_threads(num_threads_omp_)
            for (size_t c = 0; c < num_target_clusters_; ++c) {
                float dist = inner_product_distance_simd(query, centroids_data_.data() + c * vector_dim_, vector_dim_);
                #pragma omp critical
                {
                    all_centroid_distances.push_back(std::make_pair(dist, static_cast<int>(c)));
                }
            }
            
            std::sort(all_centroid_distances.begin(), all_centroid_distances.end());
            for (size_t i = 0; i < std::min(adaptive_nprobe, all_centroid_distances.size()); ++i) {
                nprobe_cluster_indices.push_back(all_centroid_distances[i].second);
            }
        }
        
        // 优化MPI通信：只广播必要信息
        int num_nprobe_indices = nprobe_cluster_indices.size();
        MPI_Bcast(&num_nprobe_indices, 1, MPI_INT, 0, comm_);
        if (rank_ != 0) {
            nprobe_cluster_indices.resize(num_nprobe_indices);
        }
        if (num_nprobe_indices > 0) {
            MPI_Bcast(nprobe_cluster_indices.data(), num_nprobe_indices, MPI_INT, 0, comm_);
        }

        if (nprobe_cluster_indices.empty() && num_target_clusters_ > 0 && adaptive_nprobe > 0) {
             // Fallback if rank 0 didn't produce any (e.g. adaptive_nprobe was 0 initially)
             // This case should ideally be handled by ensuring adaptive_nprobe > 0 if num_target_clusters_ > 0
             if (rank_ == 0) {
                for(size_t i=0; i<std::min(adaptive_nprobe, num_target_clusters_); ++i) nprobe_cluster_indices.push_back(i);
             }
             int num_fallback_indices = nprobe_cluster_indices.size();
             MPI_Bcast(&num_fallback_indices, 1, MPI_INT, 0, comm_);
             if (rank_ != 0) nprobe_cluster_indices.resize(num_fallback_indices);
             if (num_fallback_indices > 0) MPI_Bcast(nprobe_cluster_indices.data(), num_fallback_indices, MPI_INT, 0, comm_);
        }


        if (nprobe_cluster_indices.empty()) {
            return final_top_k;
        }

        // Stage 2: Parallel search in selected clusters using local HNSW indices
        std::priority_queue<std::pair<float, uint32_t>> local_top_k;
        
        // REMOVED: Distribution of clusters to specific ranks. All ranks search all nprobe_cluster_indices.
        // std::vector<int> clusters_for_this_rank;
        // for (size_t i = 0; i < nprobe_cluster_indices.size(); ++i) {
        // if (i % world_size_ == static_cast<size_t>(rank_)) {
        // clusters_for_this_rank.push_back(nprobe_cluster_indices[i]);
        // }
        // }
        
        #pragma omp parallel
        {
            std::priority_queue<std::pair<float, uint32_t>> thread_local_pq;
            // MODIFIED: Iterate over all nprobe_cluster_indices, not a subset.
            #pragma omp for schedule(dynamic)
            for (size_t i = 0; i < nprobe_cluster_indices.size(); ++i) {
                int cluster_idx = nprobe_cluster_indices[i];
                if (cluster_idx < 0 || static_cast<size_t>(cluster_idx) >= num_target_clusters_) continue; // Safety check
                
                // Check this rank's local HNSW index for the current cluster_idx
                if (hnsw_indices_[cluster_idx] && cluster_global_indices_[cluster_idx].size() > 0) {
                    try {
                        size_t cluster_size_local = cluster_global_indices_[cluster_idx].size(); // Size of local HNSW
                        size_t current_hnsw_ef = std::max(hnsw_ef, std::min(k * 4, cluster_size_local / 2));
                        if (current_hnsw_ef == 0 && cluster_size_local > 0) current_hnsw_ef = std::max(k, (size_t)1); // Ensure ef is at least 1 or k
                        hnsw_indices_[cluster_idx]->setEf(current_hnsw_ef);
                        
                        size_t search_k_local = std::min(k * 3, cluster_size_local);
                        if (search_k_local == 0 && cluster_size_local > 0) search_k_local = std::min(k, cluster_size_local);


                        auto hnsw_results = hnsw_indices_[cluster_idx]->searchKnn(query, search_k_local);
                        
                        while (!hnsw_results.empty()) {
                            auto result = hnsw_results.top();
                            hnsw_results.pop();
                            
                            // result.second is the local index within this rank's HNSW for cluster_idx
                            size_t local_hnsw_label = static_cast<size_t>(result.second);
                            if (local_hnsw_label < cluster_global_indices_[cluster_idx].size()) {
                                uint32_t global_idx = cluster_global_indices_[cluster_idx][local_hnsw_label];
                                float distance = result.first;
                                
                                if (thread_local_pq.size() < k * 2) {
                                    thread_local_pq.push(std::make_pair(distance, global_idx));
                                } else if (distance < thread_local_pq.top().first) {
                                    thread_local_pq.pop();
                                    thread_local_pq.push(std::make_pair(distance, global_idx));
                                }
                            }
                        }
                    } catch (const std::exception& e) {
                        // Error during search in local HNSW
                        #pragma omp critical
                        {
                            // std::cerr << "Rank " << rank_ << " Error searching local HNSW for cluster " << cluster_idx << ": " << e.what() << std::endl;
                        }
                        continue;
                    }
                } else if (!cluster_data_[cluster_idx].empty()) {
                    // Fallback: Brute-force search on this rank's local data for this cluster
                    // This happens if HNSW wasn't built (e.g. too few points) but local data exists for this rank in this cluster.
                    size_t num_local_points_in_cluster = cluster_global_indices_[cluster_idx].size();
                    for (size_t point_k_idx = 0; point_k_idx < num_local_points_in_cluster; ++point_k_idx) {
                        const float* base_vector_ptr = cluster_data_[cluster_idx].data() + point_k_idx * vector_dim_;
                        uint32_t global_idx = cluster_global_indices_[cluster_idx][point_k_idx]; // Get global ID
                        float dist = inner_product_distance_simd(query, base_vector_ptr, vector_dim_);

                        if (thread_local_pq.size() < k * 2) {
                            thread_local_pq.push(std::make_pair(dist, global_idx));
                        } else if (dist < thread_local_pq.top().first) {
                            thread_local_pq.pop();
                            thread_local_pq.push(std::make_pair(dist, global_idx));
                        }
                    }
                }
            }
            
            // 合并线程本地结果
            #pragma omp critical
            {
                while(!thread_local_pq.empty()){
                    if(local_top_k.size() < k * 2){
                        local_top_k.push(thread_local_pq.top());
                    } else if (thread_local_pq.top().first < local_top_k.top().first) {
                        local_top_k.pop();
                        local_top_k.push(thread_local_pq.top());
                    }
                    thread_local_pq.pop();
                }
            }
        }

        // 优化MPI通信：减少数据传输量
        std::vector<std::pair<float, uint32_t>> local_top_k_vec;
        size_t max_local_results = std::min(static_cast<size_t>(local_top_k.size()), k * 2);
        local_top_k_vec.reserve(max_local_results);
        
        while(!local_top_k.empty() && local_top_k_vec.size() < max_local_results){
            local_top_k_vec.push_back(local_top_k.top());
            local_top_k.pop();
        }
        std::reverse(local_top_k_vec.begin(), local_top_k_vec.end());

        int local_size = local_top_k_vec.size();
        std::vector<int> recv_counts(world_size_);
        std::vector<int> displs(world_size_);
        
        MPI_Gather(&local_size, 1, MPI_INT, recv_counts.data(), 1, MPI_INT, 0, comm_);

        std::vector<float> gathered_distances;
        std::vector<uint32_t> gathered_indices;
        int total_gathered_count = 0;

        if (rank_ == 0) {
            displs[0] = 0;
            total_gathered_count = recv_counts[0];
            for (int r = 1; r < world_size_; ++r) {
                displs[r] = displs[r-1] + recv_counts[r-1];
                total_gathered_count += recv_counts[r];
            }
            gathered_distances.resize(total_gathered_count);
            gathered_indices.resize(total_gathered_count);
        }

        // 分离距离和索引数组进行MPI通信
        std::vector<float> local_distances;
        std::vector<uint32_t> local_indices;
        for (const auto& pair : local_top_k_vec) {
            local_distances.push_back(pair.first);
            local_indices.push_back(pair.second);
        }

        MPI_Gatherv(local_distances.data(), local_size, MPI_FLOAT,
                    gathered_distances.data(), recv_counts.data(), displs.data(), MPI_FLOAT,
                    0, comm_);
        MPI_Gatherv(local_indices.data(), local_size, MPI_UINT32_T,
                    gathered_indices.data(), recv_counts.data(), displs.data(), MPI_UINT32_T,
                    0, comm_);

        if (rank_ == 0) {
            for(int i = 0; i < total_gathered_count; ++i) {
                std::pair<float, uint32_t> p = std::make_pair(gathered_distances[i], gathered_indices[i]);
                if (final_top_k.size() < k) {
                    final_top_k.push(p);
                } else if (p.first < final_top_k.top().first) {
                    final_top_k.pop();
                    final_top_k.push(p);
                }
            }
        }
        return final_top_k;
    }

private:
    MPI_Comm comm_;
    int rank_;
    int world_size_;

    const float* local_base_data_ptr_;
    size_t num_local_base_vectors_;
    std::vector<int> local_data_global_indices_;
    
    size_t total_base_vectors_;
    size_t vector_dim_;
    size_t num_target_clusters_;
    int num_threads_omp_;
    size_t hnsw_M_;
    size_t hnsw_ef_construction_;
    int kmeans_max_iter_;
    const float* full_base_data_for_rank0_init_ptr_;

    std::vector<float> centroids_data_;
    std::vector<std::vector<uint32_t>> inverted_lists_data_;
    
    // HNSW indices for each cluster
    std::vector<HierarchicalNSW<float>*> hnsw_indices_;
    std::vector<InnerProductSpace*> inner_product_spaces_;  // Store spaces separately
    std::vector<std::vector<float>> cluster_data_;
    std::vector<std::vector<uint32_t>> cluster_global_indices_;

    void initialize_centroids_kmeans_plus_plus(const float* full_base_data_for_rank0) {
        if (rank_ == 0) {
            if (total_base_vectors_ == 0 || num_target_clusters_ == 0 || !full_base_data_for_rank0) {
                if (num_target_clusters_ > 0) {
                    std::cerr << "K-means++ init error on rank 0: No base data or no clusters." << std::endl;
                }
                MPI_Bcast(centroids_data_.data(), num_target_clusters_ * vector_dim_, MPI_FLOAT, 0, comm_);
                return;
            }

            std::vector<float> min_dist_sq(total_base_vectors_, std::numeric_limits<float>::max());
            std::vector<bool> chosen(total_base_vectors_, false);
            std::mt19937 rng(std::chrono::system_clock::now().time_since_epoch().count());

            std::uniform_int_distribution<size_t> dist_idx(0, total_base_vectors_ - 1);
            size_t first_centroid_idx = dist_idx(rng);
            std::copy(full_base_data_for_rank0 + first_centroid_idx * vector_dim_,
                      full_base_data_for_rank0 + (first_centroid_idx + 1) * vector_dim_,
                      centroids_data_.begin());
            chosen[first_centroid_idx] = true;

            #pragma omp parallel for num_threads(num_threads_omp_)
            for (size_t i = 0; i < total_base_vectors_; ++i) {
                if (i == first_centroid_idx) {
                    min_dist_sq[i] = 0.0f;
                    continue;
                }
                float d = inner_product_distance_simd(full_base_data_for_rank0 + i * vector_dim_, centroids_data_.data(), vector_dim_);
                min_dist_sq[i] = d;
            }

            for (size_t c_idx = 1; c_idx < num_target_clusters_; ++c_idx) {
                std::vector<double> weights(total_base_vectors_);
                double total_weight = 0.0;
                for (size_t i = 0; i < total_base_vectors_; ++i) {
                    if (chosen[i]) {
                        weights[i] = 0.0;
                    } else {
                        weights[i] = static_cast<double>(min_dist_sq[i]);
                    }
                    total_weight += weights[i];
                }
                
                size_t next_centroid_base_idx = 0;
                if (total_weight == 0.0) {
                    bool found_new = false;
                    for (size_t i = 0; i < total_base_vectors_; ++i) {
                        if (!chosen[i]) {
                            next_centroid_base_idx = i;
                            found_new = true;
                            break;
                        }
                    }
                    if (!found_new) {
                        if (c_idx > 0) {
                             std::copy(centroids_data_.data() + (c_idx-1)*vector_dim_,
                                   centroids_data_.data() + c_idx*vector_dim_,
                                   centroids_data_.data() + c_idx*vector_dim_);
                        } else {
                             size_t fallback_idx = dist_idx(rng);
                             std::copy(full_base_data_for_rank0 + fallback_idx * vector_dim_,
                                  full_base_data_for_rank0 + (fallback_idx + 1) * vector_dim_,
                                  centroids_data_.data() + c_idx * vector_dim_);
                             chosen[fallback_idx] = true;
                        }
                        continue;
                    }
                } else {
                    std::uniform_real_distribution<double> dist_prob_selection(0.0, total_weight);
                    double rand_val = dist_prob_selection(rng);
                    double current_sum = 0.0;
                    for (size_t i = 0; i < total_base_vectors_; ++i) {
                        current_sum += weights[i];
                        if (current_sum >= rand_val && !chosen[i]) {
                            next_centroid_base_idx = i;
                            break;
                        }
                    }
                     if (chosen[next_centroid_base_idx] || current_sum < rand_val && total_weight > 0) {
                        for(size_t i=0; i<total_base_vectors_; ++i) if(!chosen[i]) {next_centroid_base_idx = i; break;}
                    }
                }

                std::copy(full_base_data_for_rank0 + next_centroid_base_idx * vector_dim_,
                          full_base_data_for_rank0 + (next_centroid_base_idx + 1) * vector_dim_,
                          centroids_data_.data() + c_idx * vector_dim_);
                chosen[next_centroid_base_idx] = true;

                if (c_idx < num_target_clusters_ - 1) {
                    const float* new_centroid_ptr = centroids_data_.data() + c_idx * vector_dim_;
                    #pragma omp parallel for num_threads(num_threads_omp_)
                    for (size_t i = 0; i < total_base_vectors_; ++i) {
                        if (chosen[i]) continue;
                        float d = inner_product_distance_simd(full_base_data_for_rank0 + i * vector_dim_, new_centroid_ptr, vector_dim_);
                        if (d < min_dist_sq[i]) {
                            min_dist_sq[i] = d;
                        }
                    }
                }
            }
        }
        MPI_Bcast(centroids_data_.data(), num_target_clusters_ * vector_dim_, MPI_FLOAT, 0, comm_);
    }

    void run_kmeans_parallel_mpi() {
        if (num_target_clusters_ == 0) return;

        std::vector<float> local_centroids_sum(num_target_clusters_ * vector_dim_, 0.0f); // Corrected initialization
        std::vector<int> local_centroids_count(num_target_clusters_, 0); // Corrected initialization
        std::vector<float> global_centroids_sum(num_target_clusters_ * vector_dim_);
        std::vector<int> global_centroids_count(num_target_clusters_);

        for (int iter = 0; iter < kmeans_max_iter_; ++iter) {
            std::fill(local_centroids_sum.begin(), local_centroids_sum.end(), 0.0f);
            std::fill(local_centroids_count.begin(), local_centroids_count.end(), 0);

            #pragma omp parallel
            {
                std::vector<float> thread_sums(num_target_clusters_ * vector_dim_, 0.0f);
                std::vector<int> thread_counts(num_target_clusters_, 0);

                #pragma omp for schedule(static)
                for (size_t i = 0; i < num_local_base_vectors_; ++i) {
                    const float* current_vector = local_base_data_ptr_ + i * vector_dim_;
                    float min_dist = std::numeric_limits<float>::max();
                    int best_cluster_idx = 0;
                    for (size_t c = 0; c < num_target_clusters_; ++c) {
                        const float* centroid_ptr = centroids_data_.data() + c * vector_dim_;
                        float dist = inner_product_distance_simd(current_vector, centroid_ptr, vector_dim_);
                        if (dist < min_dist) {
                            min_dist = dist;
                            best_cluster_idx = static_cast<int>(c);
                        }
                    }
                    thread_counts[best_cluster_idx]++;
                    for (size_t d = 0; d < vector_dim_; ++d) {
                        thread_sums[best_cluster_idx * vector_dim_ + d] += current_vector[d];
                    }
                }
                #pragma omp critical
                {
                    for(size_t c=0; c<num_target_clusters_; ++c){
                        local_centroids_count[c] += thread_counts[c];
                        for(size_t d=0; d<vector_dim_; ++d){
                            local_centroids_sum[c*vector_dim_ + d] += thread_sums[c*vector_dim_ + d];
                        }
                    }
                }
            }

            MPI_Allreduce(local_centroids_sum.data(), global_centroids_sum.data(),
                          num_target_clusters_ * vector_dim_, MPI_FLOAT, MPI_SUM, comm_);
            MPI_Allreduce(local_centroids_count.data(), global_centroids_count.data(),
                          num_target_clusters_, MPI_INT, MPI_SUM, comm_);
            
            bool changed_on_rank0 = false;
            if (rank_ == 0) {
                for (size_t c = 0; c < num_target_clusters_; ++c) {
                    if (global_centroids_count[c] > 0) {
                        for (size_t d = 0; d < vector_dim_; ++d) {
                            float new_val = global_centroids_sum[c * vector_dim_ + d] / global_centroids_count[c];
                            if (std::fabs(centroids_data_[c * vector_dim_ + d] - new_val) > 1e-6) {
                                changed_on_rank0 = true;
                            }
                            centroids_data_[c * vector_dim_ + d] = new_val;
                        }
                    } else {
                        if (total_base_vectors_ > num_target_clusters_ && full_base_data_for_rank0_init_ptr_) {
                             std::mt19937 rng_reinit(iter + c + rank_); 
                             std::uniform_int_distribution<size_t> dist_pt(0, total_base_vectors_ - 1);
                             size_t random_point_idx = dist_pt(rng_reinit);
                             std::copy(full_base_data_for_rank0_init_ptr_ + random_point_idx * vector_dim_,
                                       full_base_data_for_rank0_init_ptr_ + (random_point_idx + 1) * vector_dim_,
                                       centroids_data_.data() + c * vector_dim_);
                             changed_on_rank0 = true;
                        }
                    }
                }
            }
            MPI_Bcast(centroids_data_.data(), num_target_clusters_ * vector_dim_, MPI_FLOAT, 0, comm_);
            MPI_Bcast(&changed_on_rank0, 1, MPI_C_BOOL, 0, comm_);

            if (!changed_on_rank0 && iter > 0) break;
        }
    }

    void build_inverted_lists_and_hnsw_indices_mpi() {
        if (num_target_clusters_ == 0) return;

        // Initialize data structures for local contributions to each cluster
        hnsw_indices_.assign(num_target_clusters_, nullptr);
        inner_product_spaces_.assign(num_target_clusters_, nullptr);
        cluster_data_.assign(num_target_clusters_, std::vector<float>());
        cluster_global_indices_.assign(num_target_clusters_, std::vector<uint32_t>());
        // Removed: inverted_lists_data_ global population and broadcast.

        // Step 1: Assign local points to clusters and collect data for local HNSW indices.
        // Each OpenMP thread will collect data for clusters independently.
        std::vector<std::vector<std::vector<float>>> per_thread_cluster_vectors(num_threads_omp_);
        std::vector<std::vector<std::vector<uint32_t>>> per_thread_cluster_indices(num_threads_omp_);

        for (int t = 0; t < num_threads_omp_; ++t) {
            per_thread_cluster_vectors[t].resize(num_target_clusters_);
            per_thread_cluster_indices[t].resize(num_target_clusters_);
        }

        #pragma omp parallel
        {
            int thread_id = omp_get_thread_num();
            #pragma omp for schedule(static)
            for (size_t i = 0; i < num_local_base_vectors_; ++i) { // Iterate over points local to this MPI rank
                const float* current_vector = local_base_data_ptr_ + i * vector_dim_;
                float min_dist = std::numeric_limits<float>::max();
                int best_cluster_idx = 0;
                for (size_t c_idx = 0; c_idx < num_target_clusters_; ++c_idx) {
                    const float* centroid_ptr = centroids_data_.data() + c_idx * vector_dim_;
                    float dist = inner_product_distance_simd(current_vector, centroid_ptr, vector_dim_);
                    if (dist < min_dist) {
                        min_dist = dist;
                        best_cluster_idx = static_cast<int>(c_idx);
                    }
                }
                
                uint32_t global_original_idx = static_cast<uint32_t>(local_data_global_indices_[i]);

                if (best_cluster_idx >= 0 && static_cast<size_t>(best_cluster_idx) < num_target_clusters_) {
                    per_thread_cluster_indices[thread_id][best_cluster_idx].push_back(global_original_idx);
                    per_thread_cluster_vectors[thread_id][best_cluster_idx].insert(
                        per_thread_cluster_vectors[thread_id][best_cluster_idx].end(),
                        current_vector, current_vector + vector_dim_
                    );
                }
            }
        }

        // Merge per-thread lists into the rank's main cluster_data_ and cluster_global_indices_
        for (size_t c = 0; c < num_target_clusters_; ++c) {
            for (int t = 0; t < num_threads_omp_; ++t) {
                if (!per_thread_cluster_indices[t][c].empty()) {
                    cluster_global_indices_[c].insert(cluster_global_indices_[c].end(),
                                                      per_thread_cluster_indices[t][c].begin(),
                                                      per_thread_cluster_indices[t][c].end());
                    cluster_data_[c].insert(cluster_data_[c].end(),
                                             per_thread_cluster_vectors[t][c].begin(),
                                             per_thread_cluster_vectors[t][c].end());
                }
            }
        }
        
        // MPI Barrier might be useful here if there's a concern about some ranks finishing
        // local assignment much faster and proceeding, though not strictly necessary for correctness.
        // MPI_Barrier(comm_); 

        // Step 2: Build HNSW indices for local data in each cluster
        // This loop iterates over all possible clusters.
        // Each rank builds HNSW only for its portion of data in that cluster.
        #pragma omp parallel for schedule(dynamic) // Parallelize HNSW building across clusters for this rank
        for (size_t c = 0; c < num_target_clusters_; ++c) {
            if (cluster_data_[c].empty() || cluster_global_indices_[c].empty()) {
                continue; // No local data for this cluster on this rank
            }

            size_t num_points_in_cluster_local = cluster_global_indices_[c].size();
            
            if (num_points_in_cluster_local > 0) {
                try {
                    inner_product_spaces_[c] = new InnerProductSpace(vector_dim_);
                    hnsw_indices_[c] = new HierarchicalNSW<float>(inner_product_spaces_[c], 
                                                                  num_points_in_cluster_local, 
                                                                  hnsw_M_, hnsw_ef_construction_);
                    
                    // Add points to this rank's HNSW index for cluster c
                    // Labels for HNSW are local indices (0 to num_points_in_cluster_local-1)
                    // These map to global IDs via cluster_global_indices_[c]
                    for (size_t i = 0; i < num_points_in_cluster_local; ++i) {
                        hnsw_indices_[c]->addPoint(cluster_data_[c].data() + i * vector_dim_, i); 
                    }
                } catch (const std::exception& e) {
                    #pragma omp critical // Ensure thread-safe logging/cleanup
                    {
                        // Log only from one rank or summarize errors to avoid excessive output
                        // std::cerr << "Rank " << rank_ << ": Failed to build HNSW for local part of cluster " << c << ": " << e.what() << std::endl;
                    }
                    if (hnsw_indices_[c]) { delete hnsw_indices_[c]; hnsw_indices_[c] = nullptr; }
                    if (inner_product_spaces_[c]) { delete inner_product_spaces_[c]; inner_product_spaces_[c] = nullptr; }
                }
            }
        }
    }
};