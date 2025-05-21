#pragma once

#include <vector>
#include <queue>
#include <cstdint>
#include <algorithm>
#include <limits>
#include <cmath>
#include <numeric> // For std::iota
#include <random>  // For K-means++ initialization
#include <chrono>  // For random seed
#include <omp.h>   // For OpenMP

#include "simd_anns.h" // For inner_product_distance_simd

class IVFIndexOpenMP {
public:
    float* base_data_source_ptr;
    size_t num_base_vectors;
    size_t vector_dim;
    size_t num_target_clusters;
    int num_threads_to_use;

    std::vector<float> centroids_data; // Stored as flat array: num_clusters * dim
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
            std::cerr << "IVFIndexOpenMP: Invalid parameters (num_base, dim, or num_clusters is zero)." << std::endl;
            return;
        }
        if (num_target_clusters > num_base_vectors) {
            std::cerr << "IVFIndexOpenMP: Warning, num_clusters > num_base. Setting num_clusters = num_base." << std::endl;
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
            if (total_weight == 0.0) { // All remaining points are identical or already chosen
                bool found_new = false;
                for (size_t i = 0; i < num_base_vectors; ++i) {
                    if (!chosen[i]) {
                        next_centroid_base_idx = i;
                        found_new = true;
                        break;
                    }
                }
                if (!found_new) { // All points chosen, duplicate previous
                     if (c_idx > 0) {
                        std::copy(centroids_data.data() + (c_idx-1)*vector_dim,
                                centroids_data.data() + c_idx*vector_dim,
                                centroids_data.data() + c_idx*vector_dim);
                     } else { // Should not happen if c_idx starts at 1
                        size_t rand_fallback_idx = dist_idx(rng);
                         std::copy(base_data_source_ptr + rand_fallback_idx * vector_dim,
                                  base_data_source_ptr + (rand_fallback_idx + 1) * vector_dim,
                                  centroids_data.data() + c_idx * vector_dim);
                        chosen[rand_fallback_idx] = true;
                     }
                     // No need to update min_dist_sq if we duplicated or no new point
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
                if (chosen[next_centroid_base_idx] || current_sum < rand_val) { // Fallback
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
            } // End parallel region

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
                } else { // Handle empty cluster
                    if (num_base_vectors > num_target_clusters) {
                        std::mt19937 rng_reinit(iter + c + omp_get_thread_num()); // Add thread_num for more seed variance if in parallel
                        std::uniform_int_distribution<size_t> dist_pt(0, num_base_vectors - 1);
                        size_t random_point_idx = dist_pt(rng_reinit);
                        
                        bool is_already_centroid_approx = false;
                        // This check can be simplified or made more robust
                        for(size_t cc=0; cc < num_target_clusters; ++cc) {
                            if (cc == c) continue;
                            float dist_to_other_centroid_sq = 0;
                            for(size_t dd=0; dd < vector_dim; ++dd) {
                                float diff = base_data_source_ptr[random_point_idx*vector_dim + dd] - centroids_data[cc*vector_dim + dd];
                                dist_to_other_centroid_sq += diff*diff;
                            }
                            if (dist_to_other_centroid_sq < 1e-12) { // Compare squared dist
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

        // Populate inverted lists sequentially from assignments
        // This part is typically not the bottleneck. Parallelizing push_back to vectors is complex.
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

        // Stage 1: Find nprobe closest centroids
        std::vector<std::pair<float, int>> all_centroid_distances;
        all_centroid_distances.reserve(num_target_clusters);

        int stage1_threads = std::max(1, (num_target_clusters == 0) ? 1 : std::min(num_threads_to_use, static_cast<int>(num_target_clusters)));

        #pragma omp parallel num_threads(stage1_threads)
        {
            std::vector<std::pair<float, int>> local_centroid_distances;
            local_centroid_distances.reserve(num_target_clusters / stage1_threads + 1); // Pre-allocate
            
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

        // Stage 2: Search within selected nprobe clusters
        // Determine effective number of threads for stage 2 based on nprobe_cluster_indices.size()
        int stage2_threads = std::max(1, (nprobe_cluster_indices.empty()) ? 1 : std::min(num_threads_to_use, static_cast<int>(nprobe_cluster_indices.size())));
        std::vector<std::priority_queue<std::pair<float, uint32_t>>> per_thread_top_k_queues(stage2_threads);


        #pragma omp parallel num_threads(stage2_threads)
        {
            int tid = omp_get_thread_num();
            // Each thread works on its own priority queue from the vector
            // std::priority_queue<std::pair<float, uint32_t>> local_pq; // This is also an option

            #pragma omp for schedule(dynamic) // Dynamic schedule as lists can vary in size
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
        } // End parallel region for stage 2

        // Merge results from per_thread_top_k_queues
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
