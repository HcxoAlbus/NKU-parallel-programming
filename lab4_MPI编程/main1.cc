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

// =================================================================
//                 Helper Functions and Structs
// =================================================================
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
//        Strategy: Sharded-HNSW with MPI
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
    L2Space* space_ = nullptr;

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
//                 Main Function
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

    // --- Sharded HNSW index ---
    static Sharded_HNSW_MPI* sharded_hnsw_index = nullptr;
    
    // --- Data Distribution ---
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

    // --- Build Sharded HNSW index ---
    double build_start, build_end;
    if (rank == 0) std::cout << "\n--- Building Sharded HNSW Index ---" << std::endl;
    
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
            L2Space l2space(vecdim);
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
        
        // Test 2: Sharded HNSW
        if (rank == 0) std::cout << "\n--- Test 2: Sharded HNSW (MPI Distributed) ---" << std::endl;
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

    // Cleanup
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