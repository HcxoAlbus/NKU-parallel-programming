#pragma once

#include <vector>
#include <queue>
#include <cstdint>
#include <algorithm>
#include <limits>
#include <cmath>
#include <numeric> // std::iota
#include <random>  // K-means++
#include <chrono>  // K-means++ seed
#include <mpi.h>
#include <omp.h>

#include "simd_anns.h" // inner_product_distance_simd

class IVFIndexMPI {
public:
    IVFIndexMPI(MPI_Comm comm,
                const float* full_base_data_for_rank0_init, // Rank 0 uses this for K-means++
                const float* local_data_chunk_for_this_rank,
                size_t num_vectors_in_local_chunk,
                const std::vector<int>& local_data_global_indices, // Global indices of local_data_chunk_for_this_rank
                size_t total_num_base_vectors,
                size_t vector_dim,
                size_t n_clusters,
                int n_threads_omp,
                int kmeans_iterations = 20)
        : comm_(comm),
          local_base_data_ptr_(local_data_chunk_for_this_rank),
          num_local_base_vectors_(num_vectors_in_local_chunk),
          local_data_global_indices_(local_data_global_indices),
          total_base_vectors_(total_num_base_vectors),
          vector_dim_(vector_dim),
          num_target_clusters_(n_clusters),
          num_threads_omp_(std::max(1, n_threads_omp)),
          kmeans_max_iter_(kmeans_iterations) {

        MPI_Comm_rank(comm_, &rank_);
        MPI_Comm_size(comm_, &world_size_);

        omp_set_num_threads(num_threads_omp_);

        if (total_base_vectors_ == 0 || vector_dim_ == 0) {
            if (rank_ == 0) {
                std::cerr << "IVFIndexMPI: Error, total_base_vectors or vector_dim is zero." << std::endl;
            }
            // All processes should be aware of this failure state.
            // Consider throwing or setting an error flag that search checks.
            num_target_clusters_ = 0; // Prevent further operations
            return;
        }
        
        if (num_target_clusters_ > total_base_vectors_) {
            if (rank_ == 0) {
                std::cerr << "IVFIndexMPI: Warning, num_clusters > total_base_vectors. Setting num_clusters = total_base_vectors." << std::endl;
            }
            num_target_clusters_ = total_base_vectors_;
        }
        if (num_target_clusters_ == 0 && total_base_vectors_ > 0) {
             if (rank_ == 0) {
                std::cerr << "IVFIndexMPI: Warning, num_clusters is 0 but base vectors exist. No index will be built effectively." << std::endl;
            }
             // No clusters to form, so methods should handle this (e.g. search returns empty)
        }


        centroids_data_.resize(num_target_clusters_ * vector_dim_);

        if (num_target_clusters_ > 0) {
            initialize_centroids_kmeans_plus_plus(full_base_data_for_rank0_init);
            run_kmeans_parallel_mpi();
            build_inverted_lists_mpi();
        }
    }

    ~IVFIndexMPI() = default;

    std::priority_queue<std::pair<float, uint32_t>> search(
        const float* query, size_t k, size_t nprobe) {
        
        std::priority_queue<std::pair<float, uint32_t>> final_top_k;
        if (k == 0 || num_target_clusters_ == 0) return final_top_k;

        size_t current_nprobe = nprobe;
        if (current_nprobe == 0) current_nprobe = 1;
        if (current_nprobe > num_target_clusters_) current_nprobe = num_target_clusters_;

        // Stage 1: Find nprobe nearest centroids
        std::vector<int> nprobe_cluster_indices;
        nprobe_cluster_indices.reserve(current_nprobe);

        if (rank_ == 0) {
            std::vector<std::pair<float, int>> all_centroid_distances;
            all_centroid_distances.reserve(num_target_clusters_);
            for (size_t c = 0; c < num_target_clusters_; ++c) {
                float dist = inner_product_distance_simd(query, centroids_data_.data() + c * vector_dim_, vector_dim_);
                all_centroid_distances.push_back({dist, static_cast<int>(c)});
            }
            std::sort(all_centroid_distances.begin(), all_centroid_distances.end());
            for (size_t i = 0; i < std::min(current_nprobe, all_centroid_distances.size()); ++i) {
                nprobe_cluster_indices.push_back(all_centroid_distances[i].second);
            }
        }
        
        // Broadcast nprobe_cluster_indices from rank 0 to all other processes
        int num_nprobe_indices = nprobe_cluster_indices.size();
        MPI_Bcast(&num_nprobe_indices, 1, MPI_INT, 0, comm_);
        if (rank_ != 0) {
            nprobe_cluster_indices.resize(num_nprobe_indices);
        }
        MPI_Bcast(nprobe_cluster_indices.data(), num_nprobe_indices, MPI_INT, 0, comm_);

        if (nprobe_cluster_indices.empty()) {
            return final_top_k;
        }

        // Stage 2: Search within selected nprobe lists
        // Each process searches a subset of the nprobe_cluster_indices
        std::priority_queue<std::pair<float, uint32_t>> local_top_k;
        
        #pragma omp parallel
        {
            std::priority_queue<std::pair<float, uint32_t>> thread_local_pq;
            #pragma omp for schedule(dynamic)
            for (size_t i = 0; i < nprobe_cluster_indices.size(); ++i) {
                 // Distribute nprobe clusters among MPI processes by index `i`
                if (i % world_size_ != static_cast<size_t>(rank_)) {
                    continue;
                }
                int cluster_idx = nprobe_cluster_indices[i];
                if (cluster_idx < 0 || static_cast<size_t>(cluster_idx) >= inverted_lists_data_.size()) continue;

                const auto& point_indices_in_cluster = inverted_lists_data_[cluster_idx];
                for (uint32_t point_orig_idx : point_indices_in_cluster) {
                    // IMPORTANT: point_orig_idx is a GLOBAL index.
                    // We need to map this global index to the actual data.
                    // For this simplified version, we assume base_data_source_ptr_ on rank 0 has all data,
                    // or we need a mechanism to fetch vectors by global_idx, which is complex with MPI.
                    // Given inverted_lists_data_ is replicated and contains global indices,
                    // and if full_base_data_for_rank0_init was used to build it by rank 0 and then lists broadcasted,
                    // then for search, each process needs access to the original base vectors.
                    // This implies full_base_data_for_rank0_init (or equivalent) must be available to all for search,
                    // or we only search local_base_data_ptr_ and results are partial.
                    // For now, assuming inverted_lists_data_ refers to indices in a globally accessible (conceptually) dataset.
                    // If base data is distributed, this search part needs a way to get the vector data.
                    // Let's assume for this example that `full_base_data_for_rank0_init` is passed to search or accessible globally.
                    // This is a simplification. A true MPI version would fetch remote data or search only local.
                    // For this exercise, we'll assume `local_base_data_ptr_` on rank 0 is the full dataset,
                    // and other ranks might not be able to resolve `point_orig_idx` if it's not in their local chunk.
                    // This part needs careful design for a production system.
                    //
                    // Let's assume `inverted_lists_data_` was built using global indices, and `full_base_data_for_rank0_init`
                    // is the pointer to the complete dataset, available to all ranks for search (memory permitting).
                    // If not, this search logic is flawed for ranks != 0.
                    //
                    // A common pattern: rank 0 has all base data. It does the search using its OpenMP threads.
                    // Or, base data is distributed, and inverted lists point to local indices + rank owner.
                    //
                    // Given the current structure where inverted_lists_data_ is replicated and contains global indices,
                    // and centroids_data_ is replicated, the most straightforward (but memory-intensive for base data)
                    // way is that all processes have access to the full base dataset for re-ranking/final distance calculation.
                    // If `full_base_data_for_rank0_init` is accessible by all ranks:
                    const float* base_vector_ptr = full_base_data_for_rank0_init_ptr_ + point_orig_idx * vector_dim_;
                    float dist = inner_product_distance_simd(query, base_vector_ptr, vector_dim_);

                    if (thread_local_pq.size() < k) {
                        thread_local_pq.push({dist, point_orig_idx});
                    } else if (dist < thread_local_pq.top().first) {
                        thread_local_pq.pop();
                        thread_local_pq.push({dist, point_orig_idx});
                    }
                }
            }
            // Merge thread-local PQs into the process-local_top_k
            #pragma omp critical
            {
                while(!thread_local_pq.empty()){
                    if(local_top_k.size() < k){
                        local_top_k.push(thread_local_pq.top());
                    } else if (thread_local_pq.top().first < local_top_k.top().first) {
                        local_top_k.pop();
                        local_top_k.push(thread_local_pq.top());
                    }
                    thread_local_pq.pop();
                }
            }
        }


        // Gather all local_top_k results at rank 0
        // Convert PQ to vector for sending
        std::vector<std::pair<float, uint32_t>> local_top_k_vec;
        while(!local_top_k.empty()){
            local_top_k_vec.push_back(local_top_k.top());
            local_top_k.pop();
        }
        std::reverse(local_top_k_vec.begin(), local_top_k_vec.end()); // Optional, if order matters for gather

        int local_size = local_top_k_vec.size();
        std::vector<int> recv_counts(world_size_);
        std::vector<int> displs(world_size_);
        
        MPI_Gather(&local_size, 1, MPI_INT, recv_counts.data(), 1, MPI_INT, 0, comm_);

        std::vector<std::pair<float, uint32_t>> gathered_results_vec;
        int total_gathered_count = 0;

        if (rank_ == 0) {
            displs[0] = 0;
            total_gathered_count = recv_counts[0];
            for (int r = 1; r < world_size_; ++r) {
                displs[r] = displs[r-1] + recv_counts[r-1];
                total_gathered_count += recv_counts[r];
            }
            gathered_results_vec.resize(total_gathered_count);
        }

        // MPI_Gatherv requires pair<float, uint32_t> to be sent.
        // Create an MPI Datatype for std::pair<float, uint32_t>
        MPI_Datatype pair_type;
        int blocklengths[2] = {1, 1};
        MPI_Aint displacements[2];
        MPI_Datatype types[2] = {MPI_FLOAT, MPI_UINT32_T};
        
        MPI_Aint float_lb, float_extent;
        MPI_Type_get_extent(MPI_FLOAT, &float_lb, &float_extent);
        displacements[0] = 0;
        displacements[1] = float_extent;

        MPI_Type_create_struct(2, blocklengths, displacements, types, &pair_type);
        MPI_Type_commit(&pair_type);

        MPI_Gatherv(local_top_k_vec.data(), local_size, pair_type,
                    gathered_results_vec.data(), recv_counts.data(), displs.data(), pair_type,
                    0, comm_);
        
        MPI_Type_free(&pair_type);

        if (rank_ == 0) {
            for(const auto& p : gathered_results_vec) {
                if (final_top_k.size() < k) {
                    final_top_k.push(p);
                } else if (p.first < final_top_k.top().first) {
                    final_top_k.pop();
                    final_top_k.push(p);
                }
            }
        }
        return final_top_k; // Rank 0 has results, others have empty
    }


private:
    MPI_Comm comm_;
    int rank_;
    int world_size_;

    const float* local_base_data_ptr_; // Points to the local chunk of base data
    size_t num_local_base_vectors_;
    std::vector<int> local_data_global_indices_; // Global indices of vectors in local_base_data_ptr_
    
    size_t total_base_vectors_;
    size_t vector_dim_;
    size_t num_target_clusters_;
    int num_threads_omp_;
    int kmeans_max_iter_;
    const float* full_base_data_for_rank0_init_ptr_; // Store for search if needed.

    std::vector<float> centroids_data_; // Replicated
    std::vector<std::vector<uint32_t>> inverted_lists_data_; // Replicated

    void initialize_centroids_kmeans_plus_plus(const float* full_base_data_for_rank0) {
        full_base_data_for_rank0_init_ptr_ = full_base_data_for_rank0; // Save for search
        if (rank_ == 0) {
            if (total_base_vectors_ == 0 || num_target_clusters_ == 0 || !full_base_data_for_rank0) {
                 // centroids_data_ will remain empty or uninitialized if num_target_clusters_ is 0
                if (num_target_clusters_ > 0) { // Only print error if we expected to initialize
                    std::cerr << "K-means++ init error on rank 0: No base data or no clusters." << std::endl;
                }
                // Ensure centroids_data_ is broadcasted even if empty or partially set, to maintain consistency.
                // Or handle this as a fatal error. For now, broadcast whatever is in centroids_data_.
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
                    if (!found_new) { // All points chosen, duplicate last
                        if (c_idx > 0) {
                             std::copy(centroids_data_.data() + (c_idx-1)*vector_dim_,
                                   centroids_data_.data() + c_idx*vector_dim_,
                                   centroids_data_.data() + c_idx*vector_dim_);
                        } else { // Should not happen if c_idx starts at 1
                             size_t fallback_idx = dist_idx(rng);
                             std::copy(full_base_data_for_rank0 + fallback_idx * vector_dim_,
                                  full_base_data_for_rank0 + (fallback_idx + 1) * vector_dim_,
                                  centroids_data_.data() + c_idx * vector_dim_);
                             chosen[fallback_idx] = true; // Mark it
                        }
                        continue; // Skip dist update if duplicated or no new point
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
                     if (chosen[next_centroid_base_idx] || current_sum < rand_val && total_weight > 0) { // Fallback
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

        std::vector<float> local_centroids_sum(num_target_clusters_ * vector_dim_);
        std::vector<int> local_centroids_count(num_target_clusters_);
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
            } // end omp parallel

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
                    } else { // Handle empty cluster - reinitialize from a random point (simplistic)
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
            // Broadcast updated centroids and convergence status
            MPI_Bcast(centroids_data_.data(), num_target_clusters_ * vector_dim_, MPI_FLOAT, 0, comm_);
            MPI_Bcast(&changed_on_rank0, 1, MPI_C_BOOL, 0, comm_);

            if (!changed_on_rank0 && iter > 0) break;
        }
    }

    void build_inverted_lists_mpi() {
        if (num_target_clusters_ == 0) return;

        // Each process determines assignments for its local data
        std::vector<std::vector<uint32_t>> local_process_lists(num_target_clusters_);
        #pragma omp parallel for schedule(static)
        for (size_t i = 0; i < num_local_base_vectors_; ++i) {
            const float* current_vector = local_base_data_ptr_ + i * vector_dim_;
            float min_dist = std::numeric_limits<float>::max();
            int best_cluster = 0;
            for (size_t c = 0; c < num_target_clusters_; ++c) {
                const float* centroid_ptr = centroids_data_.data() + c * vector_dim_;
                float dist = inner_product_distance_simd(current_vector, centroid_ptr, vector_dim_);
                if (dist < min_dist) {
                    min_dist = dist;
                    best_cluster = static_cast<int>(c);
                }
            }
            // Store global index
            uint32_t global_idx = static_cast<uint32_t>(local_data_global_indices_[i]);
            #pragma omp critical
            {
                 if (best_cluster >= 0 && static_cast<size_t>(best_cluster) < num_target_clusters_) {
                    local_process_lists[best_cluster].push_back(global_idx);
                 }
            }
        }

        // Gather all lists at rank 0
        // This is complex because list sizes vary. Serialize and Gatherv.
        std::vector<uint32_t> send_buffer;
        std::vector<int> list_sizes_per_cluster(num_target_clusters_);
        for(size_t c=0; c < num_target_clusters_; ++c) {
            list_sizes_per_cluster[c] = local_process_lists[c].size();
            send_buffer.insert(send_buffer.end(), local_process_lists[c].begin(), local_process_lists[c].end());
        }
        
        std::vector<int> all_proc_list_counts; // Stores [P0_C0_size, P0_C1_size, ..., P1_C0_size, ...]
        if (rank_ == 0) {
            all_proc_list_counts.resize(world_size_ * num_target_clusters_);
        }
        MPI_Gather(list_sizes_per_cluster.data(), num_target_clusters_, MPI_INT,
                   all_proc_list_counts.data(), num_target_clusters_, MPI_INT, 0, comm_);

        int send_count = send_buffer.size();
        std::vector<int> recv_counts_gatherv(world_size_);
        std::vector<int> displs_gatherv(world_size_);
        MPI_Gather(&send_count, 1, MPI_INT, recv_counts_gatherv.data(), 1, MPI_INT, 0, comm_);
        
        std::vector<uint32_t> gathered_indices_buffer;
        if (rank_ == 0) {
            displs_gatherv[0] = 0;
            int total_indices_to_recv = recv_counts_gatherv[0];
            for (int r = 1; r < world_size_; ++r) {
                displs_gatherv[r] = displs_gatherv[r-1] + recv_counts_gatherv[r-1];
                total_indices_to_recv += recv_counts_gatherv[r];
            }
            gathered_indices_buffer.resize(total_indices_to_recv);
        }

        MPI_Gatherv(send_buffer.data(), send_count, MPI_UINT32_T,
                    gathered_indices_buffer.data(), recv_counts_gatherv.data(), displs_gatherv.data(), MPI_UINT32_T,
                    0, comm_);

        inverted_lists_data_.assign(num_target_clusters_, std::vector<uint32_t>());
        if (rank_ == 0) {
            int current_idx_in_gathered_buffer = 0;
            for (int r = 0; r < world_size_; ++r) {
                for (size_t c = 0; c < num_target_clusters_; ++c) {
                    int list_size_for_proc_cluster = all_proc_list_counts[r * num_target_clusters_ + c];
                    if (list_size_for_proc_cluster > 0) {
                        inverted_lists_data_[c].insert(inverted_lists_data_[c].end(),
                                                       gathered_indices_buffer.begin() + current_idx_in_gathered_buffer,
                                                       gathered_indices_buffer.begin() + current_idx_in_gathered_buffer + list_size_for_proc_cluster);
                    }
                    current_idx_in_gathered_buffer += list_size_for_proc_cluster;
                }
            }
             // Optional: Sort each list if needed, though not strictly for correctness here
            // for(auto& list : inverted_lists_data_) std::sort(list.begin(), list.end());
        }

        // Broadcast the final inverted_lists_data_ (serialized)
        // Serialization: first send total size of all lists, then sizes of each list, then all data
        std::vector<uint32_t> serialized_inverted_lists_data;
        std::vector<int> list_sizes_to_bcast(num_target_clusters_);
        int total_elements_in_all_lists = 0;

        if (rank_ == 0) {
            for(size_t c=0; c < num_target_clusters_; ++c) {
                list_sizes_to_bcast[c] = inverted_lists_data_[c].size();
                total_elements_in_all_lists += inverted_lists_data_[c].size();
                serialized_inverted_lists_data.insert(serialized_inverted_lists_data.end(), 
                                                     inverted_lists_data_[c].begin(), inverted_lists_data_[c].end());
            }
        }
        MPI_Bcast(&total_elements_in_all_lists, 1, MPI_INT, 0, comm_);
        MPI_Bcast(list_sizes_to_bcast.data(), num_target_clusters_, MPI_INT, 0, comm_);
        
        if (rank_ != 0) {
            serialized_inverted_lists_data.resize(total_elements_in_all_lists);
        }
        MPI_Bcast(serialized_inverted_lists_data.data(), total_elements_in_all_lists, MPI_UINT32_T, 0, comm_);

        if (rank_ != 0) { // Reconstruct on other ranks
            inverted_lists_data_.assign(num_target_clusters_, std::vector<uint32_t>());
            int current_offset = 0;
            for(size_t c=0; c < num_target_clusters_; ++c) {
                int list_size = list_sizes_to_bcast[c];
                if (list_size > 0) {
                    inverted_lists_data_[c].assign(serialized_inverted_lists_data.begin() + current_offset,
                                                 serialized_inverted_lists_data.begin() + current_offset + list_size);
                }
                current_offset += list_size;
            }
        }
    }
};
