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
#include "hnswlib/hnswlib/hnswlib.h"
#include "flat_scan2.h"
#include "hnsw_ivf_mpi.h"

using namespace hnswlib;

template<typename T>
T *LoadData(std::string data_path, size_t& n, size_t& d)
{
    std::ifstream fin;
    fin.open(data_path, std::ios::in | std::ios::binary);
    fin.read((char*)&n,4);
    fin.read((char*)&d,4);
    T* data = new T[n*d];
    int sz = sizeof(T);
    for(int i = 0; i < n; ++i){
        fin.read(((char*)data + i*d*sz), d*sz);
    }
    fin.close();

    std::cerr<<"load data "<<data_path<<"\n";
    std::cerr<<"dimension: "<<d<<"  number:"<<n<<"  size_per_element:"<<sizeof(T)<<"\n";

    return data;
}

struct SearchResult
{
    float recall;
    double latency; // Changed to double for MPI_Wtime precision
};

void build_hnsw_index(float* base, size_t base_number, size_t vecdim)
{
    const int efConstruction = 150;
    const int M = 16;

    HierarchicalNSW<float> *appr_alg;
    InnerProductSpace ipspace(vecdim);
    appr_alg = new HierarchicalNSW<float>(&ipspace, base_number, M, efConstruction);
    appr_alg->addPoint(base, 0);
    
    #pragma omp parallel for
    for(int i = 1; i < base_number; ++i) {
        appr_alg->addPoint(base + 1ll*vecdim*i, i);
    }
    
    char path_index[1024] = "files/hnsw.index";
    appr_alg->saveIndex(path_index);
    delete appr_alg;
}

// 创建全局HNSW对象，避免重复加载
class HNSWSearcher {
private:
    InnerProductSpace* ipspace_;
    HierarchicalNSW<float>* alg_;
    
public:
    HNSWSearcher(size_t vecdim) {
        char path_index[1024] = "files/hnsw.index";
        ipspace_ = new InnerProductSpace(vecdim);
        alg_ = new HierarchicalNSW<float>(ipspace_, path_index);
        // alg_->setEf(200); // 移除此处固定的ef设置，将在测试循环中动态设置
    }
    
    ~HNSWSearcher() {
        delete alg_;
        delete ipspace_;
    }
    
    void set_ef(size_t ef) {
        if (alg_) {
            alg_->setEf(ef);
        }
    }
    
    std::priority_queue<std::pair<float, uint32_t>> search(const float* query, size_t k) {
        auto hnsw_result = alg_->searchKnn(query, k);
        
        // Convert result to expected type
        std::priority_queue<std::pair<float, uint32_t>> result;
        while (!hnsw_result.empty()) {
            auto item = hnsw_result.top();
            hnsw_result.pop();
            result.push(std::make_pair(item.first, static_cast<uint32_t>(item.second)));
        }
        return result;
    }
};

int main(int argc, char *argv[])
{
    // Initialize MPI
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
    
    int rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    // Load data
    size_t test_number = 0, base_number = 0;
    size_t test_gt_d = 0, vecdim = 0;

    std::string data_path = "/anndata/"; 
    auto test_query = LoadData<float>(data_path + "DEEP100K.query.fbin", test_number, vecdim);
    auto test_gt = LoadData<int>(data_path + "DEEP100K.gt.query.100k.top100.bin", test_number, test_gt_d);
    auto base = LoadData<float>(data_path + "DEEP100K.base.100k.fbin", base_number, vecdim);
    
    test_number = 2000;
    const size_t k = 10;

    // Print MPI and OpenMP info
    if (rank == 0) {
        std::cout << "MPI World Size: " << world_size << std::endl;
        std::cout << "OpenMP Max Threads: " << omp_get_max_threads() << std::endl;
        std::cout << "Test queries: " << test_number << std::endl;
        std::cout << "Base vectors: " << base_number << std::endl;
        std::cout << "Vector dimension: " << vecdim << std::endl;
    }

    // ==================== Original HNSW Test ====================
    if (rank == 0) {
        std::cout << "\n=== Testing Original HNSW Algorithm (multiple ef_search values) ===" << std::endl;
        
        // Build HNSW index (only on rank 0)
        double hnsw_build_start = MPI_Wtime();
        build_hnsw_index(base, base_number, vecdim);
        double hnsw_build_time = MPI_Wtime() - hnsw_build_start;
        std::cout << "HNSW index build time: " << hnsw_build_time << " seconds" << std::endl;

        // 创建HNSW搜索器
        HNSWSearcher hnsw_searcher(vecdim);
        
        std::vector<size_t> ef_search_values_orig_hnsw = {10, 20, 40, 80, 160, 320};

        for (size_t current_ef_search : ef_search_values_orig_hnsw) {
            hnsw_searcher.set_ef(current_ef_search);
            std::cout << "\n--- Testing Original HNSW with ef_search = " << current_ef_search << " ---" << std::endl;

            std::vector<SearchResult> hnsw_results(test_number);
            
            for(int i = 0; i < test_number; ++i) {
                double start_time = MPI_Wtime();
                
                auto res = hnsw_searcher.search(test_query + i*vecdim, k);
                
                double search_time = MPI_Wtime() - start_time;

                // Calculate recall
                std::set<uint32_t> gtset;
                for(int j = 0; j < k; ++j){
                    int t = test_gt[j + i*test_gt_d];
                    gtset.insert(t);
                }

                size_t acc = 0;
                while (res.size()) {   
                    int x = res.top().second;
                    if(gtset.find(x) != gtset.end()){
                        ++acc;
                    }
                    res.pop();
                }
                float recall = (float)acc/k;
                hnsw_results[i] = {recall, search_time * 1000000}; // Convert to microseconds
            }

            // Calculate HNSW averages for current_ef_search
            float avg_hnsw_recall = 0;
            double avg_hnsw_latency = 0;
            for(int i = 0; i < test_number; ++i) {
                avg_hnsw_recall += hnsw_results[i].recall;
                avg_hnsw_latency += hnsw_results[i].latency;
            }

            std::cout << "HNSW average recall (ef=" << current_ef_search << "): " << avg_hnsw_recall / test_number << std::endl;
            std::cout << "HNSW average latency (us) (ef=" << current_ef_search << "): " << avg_hnsw_latency / test_number << std::endl;
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);

    // ==================== HNSW+IVF+MPI Test ====================
    if (rank == 0) {
        std::cout << "\n=== Testing HNSW+IVF+MPI Algorithm ===" << std::endl;
    }

    // Distribute data among MPI processes
    size_t local_chunk_size = base_number / world_size;
    size_t remainder = base_number % world_size;
    
    size_t local_start = rank * local_chunk_size + std::min(static_cast<size_t>(rank), remainder);
    size_t local_end = local_start + local_chunk_size + (rank < remainder ? 1 : 0);
    size_t local_size = local_end - local_start;

    // Create local data chunk
    float* local_base_chunk = new float[local_size * vecdim];
    std::vector<int> local_global_indices(local_size);
    
    for (size_t i = 0; i < local_size; ++i) {
        size_t global_idx = local_start + i;
        local_global_indices[i] = global_idx;
        std::copy(base + global_idx * vecdim, base + (global_idx + 1) * vecdim, 
                  local_base_chunk + i * vecdim);
    }

    if (rank == 0) {
        std::cout << "Data distributed. Local chunks: ";
        for (int r = 0; r < world_size; ++r) {
            size_t r_start = r * local_chunk_size + std::min(static_cast<size_t>(r), remainder);
            size_t r_end = r_start + local_chunk_size + (r < remainder ? 1 : 0);
            std::cout << (r_end - r_start) << " ";
        }
        std::cout << std::endl;
    }

    // Parameters for HNSW+IVF - 优化参数设置
    const size_t n_clusters = 32;                    // 减少簇数量，提高每个簇的质量
    const int n_threads = omp_get_max_threads();
    const size_t hnsw_M = 32;                        // 增加M参数
    const size_t hnsw_ef_construction = 400;         // 增加ef_construction
    const size_t nprobe = 8;                         // 增加nprobe
    // const size_t hnsw_ef_search = 200;               // 此行将被移除，因为ef_search将通过循环设置

    // Build HNSW+IVF index
    double build_start = MPI_Wtime();
    
    HNSWIVFIndexMPI index(MPI_COMM_WORLD,
                          base,
                          local_base_chunk,
                          local_size,
                          local_global_indices,
                          base_number,
                          vecdim,
                          n_clusters,
                          n_threads,
                          hnsw_M,
                          hnsw_ef_construction);
    
    double build_time = MPI_Wtime() - build_start;
    
    if (rank == 0) {
        std::cout << "HNSW+IVF+MPI index build time: " << build_time << " seconds" << std::endl;
        std::cout << "Index parameters:" << std::endl;
        std::cout << "  Clusters: " << n_clusters << std::endl;
        std::cout << "  HNSW M: " << hnsw_M << std::endl;
        std::cout << "  HNSW ef_construction: " << hnsw_ef_construction << std::endl;
        std::cout << "  nprobe: " << nprobe << std::endl;
        // std::cout << "  HNSW ef_search: " << hnsw_ef_search << std::endl; // 移除此行
    }

    MPI_Barrier(MPI_COMM_WORLD);

    // Test HNSW+IVF search - 优化搜索流程
    std::vector<size_t> ef_search_values = {10, 20, 40, 80, 160, 320};

    if (rank == 0) {
        std::cout << "\n=== Testing HNSW+IVF+MPI Algorithm (multiple ef_search values) ===" << std::endl;
    }

    for (size_t current_ef_search : ef_search_values) {
        if (rank == 0) {
            std::cout << "\n--- Testing with ef_search = " << current_ef_search << " ---" << std::endl;
        }
        MPI_Barrier(MPI_COMM_WORLD); // 确保所有进程在开始测试新的ef_search值之前同步

        std::vector<SearchResult> mpi_results(test_number);
        
        double total_search_start_for_ef = MPI_Wtime();
        
        // 减少同步开销：批量处理查询
        const int batch_size = 10;
        for(int batch_start = 0; batch_start < static_cast<int>(test_number); batch_start += batch_size) {
            int batch_end = std::min(batch_start + batch_size, static_cast<int>(test_number));
            
            for(int i = batch_start; i < batch_end; ++i) {
                double query_start = MPI_Wtime();
                
                // 使用 current_ef_search 替换固定的 hnsw_ef_search
                auto res = index.search(test_query + i*vecdim, k, nprobe, current_ef_search);
                
                double query_time = MPI_Wtime() - query_start;

                if (rank == 0) {
                    // Calculate recall
                    std::set<uint32_t> gtset;
                    for(int j = 0; j < k; ++j){
                        int t = test_gt[j + i*test_gt_d];
                        gtset.insert(t);
                    }

                    size_t acc = 0;
                    while (res.size()) {   
                        int x = res.top().second;
                        if(gtset.find(x) != gtset.end()){
                            ++acc;
                        }
                        res.pop();
                    }
                    float recall = (float)acc/k;
                    mpi_results[i] = {recall, query_time * 1000000};
                }
            }
            
            // 每个批次后同步一次，而不是每个查询后都同步
            MPI_Barrier(MPI_COMM_WORLD);
        }
        
        double total_search_time_for_ef = MPI_Wtime() - total_search_start_for_ef;

        // Calculate and report results (only on rank 0)
        if (rank == 0) {
            float avg_mpi_recall = 0;
            double avg_mpi_latency = 0;
            for(int i = 0; i < test_number; ++i) {
                avg_mpi_recall += mpi_results[i].recall;
                avg_mpi_latency += mpi_results[i].latency;
            }

            std::cout << "HNSW+IVF+MPI Results (ef_search = " << current_ef_search << "):" << std::endl;
            std::cout << "  Average recall: " << avg_mpi_recall / test_number << std::endl;
            std::cout << "  Average latency per query (us): " << avg_mpi_latency / test_number << std::endl;
            std::cout << "  Total search time (s): " << total_search_time_for_ef << std::endl;
            if (total_search_time_for_ef > 0) {
                std::cout << "  Throughput (queries/s): " << test_number / total_search_time_for_ef << std::endl;
            } else {
                std::cout << "  Throughput (queries/s): N/A (total search time is zero)" << std::endl;
            }
        }
        MPI_Barrier(MPI_COMM_WORLD); // 确保所有进程在移动到下一个ef_search值之前完成报告
    }

    // ==================== Brute Force Test (for comparison) ====================
    if (rank == 0) {
        std::cout << "\n=== Testing Brute Force Search ===" << std::endl;
        
        std::vector<SearchResult> bf_results(test_number);
        
        for(int i = 0; i < test_number; ++i) {
            double start_time = MPI_Wtime();
            
            auto res = flat_search(base, test_query + i*vecdim, base_number, vecdim, k);
            
            double search_time = MPI_Wtime() - start_time;

            // Calculate recall (should be 1.0 for brute force)
            std::set<uint32_t> gtset;
            for(int j = 0; j < k; ++j){
                int t = test_gt[j + i*test_gt_d];
                gtset.insert(t);
            }

            size_t acc = 0;
            while (res.size()) {   
                int x = res.top().second;
                if(gtset.find(x) != gtset.end()){
                    ++acc;
                }
                res.pop();
            }
            float recall = (float)acc/k;
            bf_results[i] = {recall, search_time * 1000000}; // Convert to microseconds
        }

        // Calculate averages
        float avg_bf_recall = 0;
        double avg_bf_latency = 0;
        for(int i = 0; i < test_number; ++i) {
            avg_bf_recall += bf_results[i].recall;
            avg_bf_latency += bf_results[i].latency;
        }

        std::cout << "Brute Force average recall: " << avg_bf_recall / test_number << std::endl;
        std::cout << "Brute Force average latency (us): " << avg_bf_latency / test_number << std::endl;
    }

    // Cleanup
    delete[] test_query;
    delete[] test_gt;
    delete[] base;
    delete[] local_base_chunk;

    MPI_Finalize();
    return 0;
}