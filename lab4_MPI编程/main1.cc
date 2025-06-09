#include <vector>
#include <cstring>
#include <string>
#include <iostream>
#include <fstream>
#include <set>
#include <chrono>
#include <iomanip>
#include <sstream>
#include <sys/time.h>
#include <omp.h>
#include <mpi.h>
#include <numeric>
#include <random>
#include <algorithm>
#include <limits>
#include <cmath>

#include "hnswlib/hnswlib/hnswlib.h"

// --- SIMD 距离计算函数 ---
#ifdef __AVX__
#include <immintrin.h>
// L2 距离的平方
static float l2_distance_simd(const float *query, const float *vec, size_t dim) {
    __m256 diff, sum;
    sum = _mm256_setzero_ps();
    size_t aligned_dim = dim - (dim % 8);
    for (size_t i = 0; i < aligned_dim; i += 8) {
        __m256 q = _mm256_loadu_ps(query + i);
        __m256 v = _mm256_loadu_ps(vec + i);
        diff = _mm256_sub_ps(q, v);
        sum = _mm256_add_ps(sum, _mm256_mul_ps(diff, diff));
    }
    float Tmp[8];
    _mm256_storeu_ps(Tmp, sum);
    float res = Tmp[0] + Tmp[1] + Tmp[2] + Tmp[3] + Tmp[4] + Tmp[5] + Tmp[6] + Tmp[7];
    for (size_t i = aligned_dim; i < dim; ++i) {
        float d = query[i] - vec[i];
        res += d * d;
    }
    return res;
}
#else
// Fallback for non-AVX
static float l2_distance_simd(const float *query, const float *vec, size_t dim) {
    float sum = 0.0f;
    for (size_t i = 0; i < dim; ++i) {
        float d = query[i] - vec[i];
        sum += d * d;
    }
    return sum;
}
#endif


using namespace hnswlib;

// --- 全局参数 ---
const int HNSW_M = 16;
const int HNSW_EF_CONSTRUCTION = 150;
const int N_CLUSTERS = 256; 
const int N_PROBE = 32; // *** 增加 nprobe ***

// =================================================================
//                 Helper Functions and Structs
// =================================================================
// ... (SearchResult, gather_results_at_root, LoadData are unchanged) ...
// ... To save space, I will omit them here. Assume they exist from the previous answer.
struct SearchResult { float recall; double latency_us; };
template<typename T> T* LoadData(std::string data_path, size_t& n, size_t& d) {
    std::ifstream fin(data_path, std::ios::in | std::ios::binary);
    if (!fin) { std::cerr << "Cannot open " << data_path << std::endl; exit(1); }
    fin.read((char*)&n, 4); fin.read((char*)&d, 4);
    T* data = new T[n * d];
    fin.read((char*)data, (long long)n * d * sizeof(T));
    fin.close();
    if (d > 0)
        std::cerr << "Loaded " << data_path << ": " << n << " vectors, " << d << " dims" << std::endl;
    return data;
}
std::priority_queue<std::pair<float, int>> gather_results_at_root(
    const std::priority_queue<std::pair<float, int>>& local_pq, 
    size_t k, int rank, int world_size, MPI_Comm comm); // Declaration

// =================================================================
//        Strategy 1: IVF-HNSW with MPI (Corrected with L2 Distance)
// =================================================================
class IVF_HNSW_MPI {
public:
    IVF_HNSW_MPI(MPI_Comm comm,
                 const float* full_base_data_for_rank0_init, 
                 const float* local_data_chunk_for_this_rank,
                 const std::vector<int>& local_data_global_indices,
                 size_t total_num_base_vectors,
                 size_t vector_dim,
                 int n_threads_omp)
        : comm_(comm), local_base_data_ptr_(local_data_chunk_for_this_rank),
          local_data_global_indices_(local_data_global_indices),
          total_base_vectors_(total_num_base_vectors), vector_dim_(vector_dim),
          num_threads_omp_(n_threads_omp) {
        MPI_Comm_rank(comm_, &rank_);
        MPI_Comm_size(comm_, &world_size_);
        omp_set_num_threads(num_threads_omp_);
        centroids_.resize(N_CLUSTERS * vector_dim_);
        run_kmeans_parallel(full_base_data_for_rank0_init);
        partition_local_data();
        build_local_hnsw_indexes();
    }
    ~IVF_HNSW_MPI() {
        for (auto& space : spaces_) delete space;
        for (auto& hnsw_index : local_hnsw_indexes_) delete hnsw_index;
    }
    
    std::priority_queue<std::pair<float, int>> search(const float* query, size_t k, int ef) {
        std::vector<float> query_vec(vector_dim_);
        if (rank_ == 0) memcpy(query_vec.data(), query, vector_dim_ * sizeof(float));
        MPI_Bcast(query_vec.data(), vector_dim_, MPI_FLOAT, 0, comm_);

        std::vector<int> nprobe_cluster_indices = find_nprobe_centroids(query_vec.data());

        std::priority_queue<std::pair<float, int>> local_top_k;
        
        for (size_t i = 0; i < nprobe_cluster_indices.size(); ++i) {
            if (i % world_size_ != rank_) continue;
            
            int cluster_id = nprobe_cluster_indices[i];
            if (local_hnsw_indexes_[cluster_id] == nullptr || local_hnsw_indexes_[cluster_id]->getCurrentElementCount() == 0) continue;
            
            local_hnsw_indexes_[cluster_id]->setEf(ef);
            auto result_pq_hnsw = local_hnsw_indexes_[cluster_id]->searchKnn(query_vec.data(), k);
            
            std::priority_queue<std::pair<float, labeltype>> temp_pq;
            while(!result_pq_hnsw.empty()) {
                temp_pq.push(result_pq_hnsw.top());
                result_pq_hnsw.pop();
            }

            while (!temp_pq.empty()) {
                auto top_el = temp_pq.top();
                if (local_top_k.size() < k) {
                    local_top_k.push({top_el.first, (int)top_el.second});
                } else if (top_el.first < local_top_k.top().first) {
                    local_top_k.pop();
                    local_top_k.push({top_el.first, (int)top_el.second});
                }
                temp_pq.pop();
            }
        }
        return gather_results_at_root(local_top_k, k, rank_, world_size_, comm_);
    }

private:
    MPI_Comm comm_; int rank_, world_size_; const float* local_base_data_ptr_;
    std::vector<int> local_data_global_indices_; size_t total_base_vectors_, vector_dim_;
    int num_threads_omp_; std::vector<float> centroids_;
    std::vector<std::vector<size_t>> local_partitions_;
    std::vector<HierarchicalNSW<float>*> local_hnsw_indexes_;
    std::vector<L2Space*> spaces_; // *** Use L2Space ***

    void run_kmeans_parallel(const float* full_base_data) {
        // K-means++ initialization on rank 0 (using L2 distance)
        if (rank_ == 0) {
            std::vector<float> min_dist_sq(total_base_vectors_, std::numeric_limits<float>::max());
            std::mt19937 rng(42); 
            std::uniform_int_distribution<size_t> dist_idx(0, total_base_vectors_ - 1);
            size_t first_idx = dist_idx(rng);
            memcpy(centroids_.data(), full_base_data + first_idx * vector_dim_, vector_dim_ * sizeof(float));

            for (int c_idx = 1; c_idx < N_CLUSTERS; ++c_idx) {
                double total_weight = 0.0;
                #pragma omp parallel for reduction(+:total_weight)
                for (size_t i = 0; i < total_base_vectors_; ++i) {
                    float d_sq = l2_distance_simd(full_base_data + i * vector_dim_, centroids_.data() + (c_idx - 1) * vector_dim_, vector_dim_);
                    if (d_sq < min_dist_sq[i]) min_dist_sq[i] = d_sq;
                    total_weight += min_dist_sq[i];
                }
                
                std::uniform_real_distribution<double> dist_prob(0.0, total_weight);
                double rand_val = dist_prob(rng);
                double current_sum = 0.0;
                size_t next_idx = 0;
                for (size_t i = 0; i < total_base_vectors_; ++i) {
                    current_sum += min_dist_sq[i];
                    if (current_sum >= rand_val) { next_idx = i; break; }
                }
                memcpy(centroids_.data() + c_idx * vector_dim_, full_base_data + next_idx * vector_dim_, vector_dim_ * sizeof(float));
            }
        }
        MPI_Bcast(centroids_.data(), N_CLUSTERS * vector_dim_, MPI_FLOAT, 0, comm_);

        // K-means iterations (using L2 distance)
        const int kmeans_max_iter = 25; // A few more iterations
        for (int iter = 0; iter < kmeans_max_iter; ++iter) {
            std::vector<float> local_sums(N_CLUSTERS * vector_dim_, 0.0f);
            std::vector<int> local_counts(N_CLUSTERS, 0);

            #pragma omp parallel
            {
                std::vector<float> t_sums(N_CLUSTERS * vector_dim_, 0.0f);
                std::vector<int> t_counts(N_CLUSTERS, 0);
                #pragma omp for
                for (size_t i = 0; i < local_data_global_indices_.size(); ++i) {
                    const float* vec = local_base_data_ptr_ + i * vector_dim_;
                    float min_dist = std::numeric_limits<float>::max();
                    int best_c = 0;
                    for (int c = 0; c < N_CLUSTERS; ++c) {
                        float dist = l2_distance_simd(vec, centroids_.data() + c * vector_dim_, vector_dim_);
                        if (dist < min_dist) { min_dist = dist; best_c = c; }
                    }
                    t_counts[best_c]++;
                    for (size_t d = 0; d < vector_dim_; ++d) t_sums[best_c * vector_dim_ + d] += vec[d];
                }
                #pragma omp critical
                for (int c = 0; c < N_CLUSTERS; ++c) {
                    local_counts[c] += t_counts[c];
                    for (size_t d = 0; d < vector_dim_; ++d) local_sums[c * vector_dim_ + d] += t_sums[c * vector_dim_ + d];
                }
            }

            std::vector<float> global_sums(N_CLUSTERS * vector_dim_);
            std::vector<int> global_counts(N_CLUSTERS);
            MPI_Allreduce(local_sums.data(), global_sums.data(), N_CLUSTERS * vector_dim_, MPI_FLOAT, MPI_SUM, comm_);
            MPI_Allreduce(local_counts.data(), global_counts.data(), N_CLUSTERS, MPI_INT, MPI_SUM, comm_);
            
            bool changed = false;
            for (int c = 0; c < N_CLUSTERS; ++c) {
                if (global_counts[c] > 0) {
                    for (size_t d = 0; d < vector_dim_; ++d) {
                        float new_val = global_sums[c * vector_dim_ + d] / global_counts[c];
                        if (std::abs(centroids_[c * vector_dim_ + d] - new_val) > 1e-6) changed = true;
                        centroids_[c * vector_dim_ + d] = new_val;
                    }
                }
            }
            MPI_Bcast(centroids_.data(), N_CLUSTERS * vector_dim_, MPI_FLOAT, 0, comm_); // Broadcast updated centroids
            MPI_Bcast(&changed, 1, MPI_C_BOOL, 0, comm_);
            if (!changed && iter > 5) break; // Early stop if converged
        }
    }

    void partition_local_data() {
        local_partitions_.assign(N_CLUSTERS, std::vector<size_t>());
        for (size_t i = 0; i < local_data_global_indices_.size(); ++i) {
            const float* vec = local_base_data_ptr_ + i * vector_dim_;
            float min_dist = std::numeric_limits<float>::max();
            int best_c = 0;
            for (int c = 0; c < N_CLUSTERS; ++c) {
                float dist = l2_distance_simd(vec, centroids_.data() + c * vector_dim_, vector_dim_);
                if (dist < min_dist) { min_dist = dist; best_c = c; }
            }
            local_partitions_[best_c].push_back(i);
        }
    }

    void build_local_hnsw_indexes() {
        local_hnsw_indexes_.resize(N_CLUSTERS, nullptr);
        spaces_.resize(N_CLUSTERS, nullptr);
        #pragma omp parallel for
        for (int c = 0; c < N_CLUSTERS; ++c) {
            size_t num_points = local_partitions_[c].size();
            if (num_points == 0) continue;
            spaces_[c] = new L2Space(vector_dim_);
            auto* hnsw = new HierarchicalNSW<float>(spaces_[c], num_points, HNSW_M, HNSW_EF_CONSTRUCTION);
            for (size_t i = 0; i < num_points; ++i) {
                size_t local_idx = local_partitions_[c][i];
                size_t global_idx = local_data_global_indices_[local_idx];
                hnsw->addPoint(local_base_data_ptr_ + local_idx * vector_dim_, global_idx);
            }
            local_hnsw_indexes_[c] = hnsw;
        }
    }
    
    std::vector<int> find_nprobe_centroids(const float* query) {
        std::vector<std::pair<float, int>> dists;
        for (int c = 0; c < N_CLUSTERS; ++c) {
            dists.push_back({l2_distance_simd(query, centroids_.data() + c * vector_dim_, vector_dim_), c});
        }
        std::sort(dists.begin(), dists.end());
        std::vector<int> result;
        for(int i = 0; i < N_PROBE && i < N_CLUSTERS; ++i) result.push_back(dists[i].second);
        return result;
    }
};

// =================================================================
//        Strategy 2: Sharded-HNSW with MPI (Corrected with L2)
// =================================================================
class Sharded_HNSW_MPI {
public:
    Sharded_HNSW_MPI(MPI_Comm comm,
                     const float* local_data_chunk,
                     const std::vector<int>& local_global_indices,
                     size_t vector_dim,
                     int n_threads_omp)
        : comm_(comm), local_base_data_ptr_(local_data_chunk),
          local_data_global_indices_(local_global_indices),
          vector_dim_(vector_dim), num_threads_omp_(n_threads_omp) {
        MPI_Comm_rank(comm_, &rank_);
        MPI_Comm_size(comm_, &world_size_);
        omp_set_num_threads(num_threads_omp_);
        build_local_index();
    }
    ~Sharded_HNSW_MPI() { delete local_hnsw_index_; delete space_; }

    std::priority_queue<std::pair<float, int>> search(const float* query, size_t k, int ef) {
        std::vector<float> query_vec(vector_dim_);
        if (rank_ == 0) memcpy(query_vec.data(), query, vector_dim_ * sizeof(float));
        MPI_Bcast(query_vec.data(), vector_dim_, MPI_FLOAT, 0, comm_);

        local_hnsw_index_->setEf(ef);
        auto result_pq_hnsw = local_hnsw_index_->searchKnn(query_vec.data(), k);
        
        std::priority_queue<std::pair<float, int>> local_top_k;
        while (!result_pq_hnsw.empty()) {
            local_top_k.push({result_pq_hnsw.top().first, (int)result_pq_hnsw.top().second});
            result_pq_hnsw.pop();
        }
        return gather_results_at_root(local_top_k, k, rank_, world_size_, comm_);
    }

private:
    MPI_Comm comm_; int rank_, world_size_; const float* local_base_data_ptr_;
    std::vector<int> local_data_global_indices_; size_t vector_dim_; int num_threads_omp_;
    HierarchicalNSW<float>* local_hnsw_index_ = nullptr;
    L2Space* space_ = nullptr; // *** Use L2Space ***

    void build_local_index() {
        size_t num_local_points = local_data_global_indices_.size();
        if (num_local_points == 0) return;
        space_ = new L2Space(vector_dim_);
        local_hnsw_index_ = new HierarchicalNSW<float>(space_, num_local_points, HNSW_M, HNSW_EF_CONSTRUCTION);
        #pragma omp parallel for
        for (size_t i = 0; i < num_local_points; ++i) {
            local_hnsw_index_->addPoint(local_base_data_ptr_ + i * vector_dim_, local_data_global_indices_[i]);
        }
    }
};

// =================================================================
//                 Main Function with All Tests
// =================================================================
int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);
    int rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    int ppn = (argc > 1) ? std::atoi(argv[1]) : 8;
    omp_set_num_threads(ppn);

    // Data Loading on Rank 0
    size_t test_number = 0, base_number = 0, test_gt_d = 0, vecdim = 0;
    float *test_query = nullptr, *base = nullptr;
    int *test_gt = nullptr;
    if (rank == 0) {
        std::string data_path = "/anndata/";
        test_query = LoadData<float>(data_path + "DEEP100K.query.fbin", test_number, vecdim);
        test_gt = LoadData<int>(data_path + "DEEP100K.gt.query.100k.top100.bin", test_number, test_gt_d);
        base = LoadData<float>(data_path + "DEEP100K.base.100k.fbin", base_number, vecdim);
    }
    
    MPI_Bcast(&test_number, 1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);
    MPI_Bcast(&base_number, 1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);
    MPI_Bcast(&vecdim, 1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);
    test_number = 2000;
    const size_t k = 10;
    
    std::vector<int> ef_values = {10, 20, 40, 80, 160, 320};

    // --- Static objects for parallel indices to avoid rebuilding ---
    static IVF_HNSW_MPI* ivf_hnsw_index = nullptr;
    static Sharded_HNSW_MPI* sharded_hnsw_index = nullptr;
    
    // --- Data Distribution (once) ---
    std::vector<int> sendcounts(world_size), displs(world_size);
    std::vector<int> sendcounts_indices(world_size), displs_indices(world_size);
    int vectors_per_proc = base_number / world_size;
    int remainder = base_number % world_size;
    for (int i = 0; i < world_size; ++i) {
        sendcounts_indices[i] = (i < remainder) ? vectors_per_proc + 1 : vectors_per_proc;
        sendcounts[i] = sendcounts_indices[i] * vecdim;
        displs_indices[i] = (i == 0) ? 0 : displs_indices[i-1] + sendcounts_indices[i-1];
        displs[i] = displs_indices[i] * vecdim;
    }
    std::vector<float> local_base_data(sendcounts[rank]);
    MPI_Scatterv(base, sendcounts.data(), displs.data(), MPI_FLOAT, local_base_data.data(), sendcounts[rank], MPI_FLOAT, 0, MPI_COMM_WORLD);
    std::vector<int> local_global_indices(sendcounts_indices[rank]);
    for(int i = 0; i < sendcounts_indices[rank]; ++i) local_global_indices[i] = displs_indices[rank] + i;

    // --- Build parallel indices (once) ---
    double build_start, build_end;
    if (rank == 0) std::cout << "\n--- Building Parallel Indices ---" << std::endl;
    
    build_start = MPI_Wtime();
    ivf_hnsw_index = new IVF_HNSW_MPI(MPI_COMM_WORLD, base, local_base_data.data(), local_global_indices, base_number, vecdim, ppn);
    MPI_Barrier(MPI_COMM_WORLD);
    build_end = MPI_Wtime();
    if (rank == 0) std::cout << "IVF-HNSW Build Time: " << (build_end - build_start) * 1000 << " ms" << std::endl;
    
    build_start = MPI_Wtime();
    sharded_hnsw_index = new Sharded_HNSW_MPI(MPI_COMM_WORLD, local_base_data.data(), local_global_indices, vecdim, ppn);
    MPI_Barrier(MPI_COMM_WORLD);
    build_end = MPI_Wtime();
    if (rank == 0) std::cout << "Sharded-HNSW Build Time: " << (build_end - build_start) * 1000 << " ms" << std::endl;

    for (int ef : ef_values) {
        if (rank == 0) {
            std::cout << "\n========================================================" << std::endl;
            std::cout << "           PERFORMANCE TEST (ef_search = " << ef << ")" << std::endl;
            std::cout << "========================================================" << std::endl;
        }

        // Test 1: Plain HNSW (on Rank 0)
        if (rank == 0) {
            std::cout << "\n--- Test 1: Original HNSW (L2, Single Process) ---" << std::endl;
            L2Space l2space(vecdim); // *** Use L2Space ***
            HierarchicalNSW<float> appr_alg(&l2space, base_number, HNSW_M, HNSW_EF_CONSTRUCTION);
            #pragma omp parallel for
            for(int i = 0; i < base_number; ++i) appr_alg.addPoint(base + (long long)vecdim * i, i);
            appr_alg.setEf(ef);

            double total_latency = 0; float total_recall = 0;
            for(int i = 0; i < test_number; ++i) {
                double start_time = MPI_Wtime();
                auto res_pq = appr_alg.searchKnn(test_query + i * vecdim, k);
                total_latency += (MPI_Wtime() - start_time) * 1e6;
                std::set<int> gtset; for(int j=0; j<k; ++j) gtset.insert(test_gt[j + (long long)i*test_gt_d]);
                size_t acc = 0; while(!res_pq.empty()) { if(gtset.count(res_pq.top().second)) acc++; res_pq.pop(); }
                total_recall += (float)acc/k;
            }
            std::cout << "Average Recall: " << total_recall / test_number << std::endl;
            std::cout << "Average Latency (us): " << total_latency / test_number << std::endl;
        }

        // Test 2: IVF-HNSW
        if (rank == 0) std::cout << "\n--- Test 2: Strategy 1 - IVF-HNSW (MPI on Clusters) ---" << std::endl;
        double total_latency_ivf = 0; float total_recall_ivf = 0;
        for(int i = 0; i < test_number; ++i) {
            MPI_Barrier(MPI_COMM_WORLD); double start_time = MPI_Wtime();
            auto res_pq = ivf_hnsw_index->search(test_query + i * vecdim, k, ef);
            MPI_Barrier(MPI_COMM_WORLD); double end_time = MPI_Wtime();
            if (rank == 0) {
                total_latency_ivf += (end_time - start_time) * 1e6;
                std::set<int> gtset; for(int j=0; j<k; ++j) gtset.insert(test_gt[j + (long long)i*test_gt_d]);
                size_t acc = 0; while(!res_pq.empty()) { if(gtset.count(res_pq.top().second)) acc++; res_pq.pop(); }
                total_recall_ivf += (float)acc/k;
            }
        }
        if (rank == 0) {
            std::cout << "Average Recall: " << total_recall_ivf / test_number << std::endl;
            std::cout << "Average Latency (us): " << total_latency_ivf / test_number << std::endl;
        }
        
        // Test 3: Sharded HNSW
        if (rank == 0) std::cout << "\n--- Test 3: Strategy 2 - Sharded HNSW (MPI on Shards) ---" << std::endl;
        double total_latency_sharded = 0; float total_recall_sharded = 0;
        for(int i = 0; i < test_number; ++i) {
            MPI_Barrier(MPI_COMM_WORLD); double start_time = MPI_Wtime();
            auto res_pq = sharded_hnsw_index->search(test_query + i * vecdim, k, ef);
            MPI_Barrier(MPI_COMM_WORLD); double end_time = MPI_Wtime();
            if (rank == 0) {
                total_latency_sharded += (end_time - start_time) * 1e6;
                std::set<int> gtset; for(int j=0; j<k; ++j) gtset.insert(test_gt[j + (long long)i*test_gt_d]);
                size_t acc = 0; while(!res_pq.empty()) { if(gtset.count(res_pq.top().second)) acc++; res_pq.pop(); }
                total_recall_sharded += (float)acc/k;
            }
        }
        if (rank == 0) {
            std::cout << "Average Recall: " << total_recall_sharded / test_number << std::endl;
            std::cout << "Average Latency (us): " << total_latency_sharded / test_number << std::endl;
        }
    }

    // Final Cleanup
    delete ivf_hnsw_index;
    delete sharded_hnsw_index;
    if (rank == 0) {
        delete[] test_query;
        delete[] test_gt;
        delete[] base;
    }
    
    MPI_Finalize();
    return 0;
}


// Implementation of gather_results_at_root
std::priority_queue<std::pair<float, int>> gather_results_at_root(
    const std::priority_queue<std::pair<float, int>>& local_pq, 
    size_t k, int rank, int world_size, MPI_Comm comm) {

    std::vector<std::pair<float, int>> local_vec;
    auto temp_pq = local_pq;
    while (!temp_pq.empty()) {
        local_vec.push_back(temp_pq.top());
        temp_pq.pop();
    }
    
    int local_size = local_vec.size();
    std::vector<int> recv_counts(world_size, 0);
    MPI_Gather(&local_size, 1, MPI_INT, recv_counts.data(), 1, MPI_INT, 0, comm);

    std::vector<std::pair<float, int>> gathered_vec;
    std::vector<int> displs(world_size, 0);
    if (rank == 0) {
        int total_gathered_count = 0;
        displs[0] = 0;
        total_gathered_count = recv_counts[0];
        for (int r = 1; r < world_size; ++r) {
            displs[r] = displs[r-1] + recv_counts[r-1];
            total_gathered_count += recv_counts[r];
        }
        gathered_vec.resize(total_gathered_count);
    }
    
    MPI_Datatype pair_type;
    int blocklengths[2] = {1, 1};
    MPI_Aint displacements[2];
    MPI_Datatype types[2] = {MPI_FLOAT, MPI_INT};
    MPI_Aint float_lb, float_extent;
    MPI_Type_get_extent(MPI_FLOAT, &float_lb, &float_extent);
    displacements[0] = 0;
    displacements[1] = float_extent;
    MPI_Type_create_struct(2, blocklengths, displacements, types, &pair_type);
    MPI_Type_commit(&pair_type);

    MPI_Gatherv(local_vec.data(), local_size, pair_type,
                gathered_vec.data(), recv_counts.data(), displs.data(), pair_type,
                0, comm);
    MPI_Type_free(&pair_type);

    std::priority_queue<std::pair<float, int>> final_pq;
    if (rank == 0) {
        for (const auto& p : gathered_vec) {
            if (final_pq.size() < k) {
                final_pq.push(p);
            } else if (p.first < final_pq.top().first) {
                final_pq.pop();
                final_pq.push(p);
            }
        }
    }
    return final_pq;
}