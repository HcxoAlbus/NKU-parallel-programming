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
#include <omp.h> // For OMP_NUM_THREADS, but be careful with Pthread usage
// #include "hnswlib/hnswlib/hnswlib.h" 
#include "flat_scan.h" 
#include "simd_anns.h"
#include "pq_anns.h" 
#include "sq_anns.h"
#include "ivf_anns.h" // Include the new IVF header
#include <functional>
#include <algorithm> 
#include <queue>     
#include <stdexcept> 

#include "ivf_pq_anns.h" // Include the new IVF-PQ header
// --- 函数声明  ---
std::priority_queue<std::pair<float, uint32_t>> flat_search(float* base, const float* query, size_t base_number, size_t vecdim, size_t k);
std::priority_queue<std::pair<float, uint32_t>> simd_search(float* base, const float* query, size_t base_number, size_t vecdim, size_t k);
// ---
template<typename T>
T *LoadData(std::string data_path, size_t& n_out, size_t& d_out) // Renamed to avoid confusion
{
    std::ifstream fin;
    fin.open(data_path, std::ios::in | std::ios::binary);
    if (!fin.is_open()) {
        std::cerr << "Error opening data file: " << data_path << std::endl;
        exit(1); // Or throw std::runtime_error for better error handling
    }

    uint32_t n_file, d_file; // Use fixed-size types for reading file metadata

    fin.read(reinterpret_cast<char*>(&n_file), sizeof(uint32_t));
    if (!fin) {
        std::cerr << "Error reading 'n' from data file: " << data_path << std::endl;
        fin.close();
        exit(1);
    }
    fin.read(reinterpret_cast<char*>(&d_file), sizeof(uint32_t));
    if (!fin) {
        std::cerr << "Error reading 'd' from data file: " << data_path << std::endl;
        fin.close();
        exit(1);
    }

    n_out = static_cast<size_t>(n_file);
    d_out = static_cast<size_t>(d_file);

    if (n_out == 0 || d_out == 0) {
        std::cerr << "Warning: n or d is zero for " << data_path << ". n=" << n_out << ", d=" << d_out << std::endl;
        // Allocate a zero-size array or return nullptr, ensure downstream code handles it.
        // new T[0] is valid in C++.
    }
    
    // Check for potential overflow before multiplication for allocation size
    if (n_out > 0 && d_out > 0 && (n_out > std::numeric_limits<size_t>::max() / d_out) ) {
        std::cerr << "Error: n*d would overflow size_t for " << data_path << ". n=" << n_out << ", d=" << d_out << std::endl;
        fin.close();
        exit(1); // Or throw std::overflow_error
    }

    T* data = nullptr;
    try {
        data = new T[n_out * d_out];
    } catch (const std::bad_alloc& e) {
        std::cerr << "std::bad_alloc caught while trying to allocate for " << data_path 
                  << " with n=" << n_out << ", d=" << d_out 
                  << " (total elements: " << n_out * d_out 
                  << ", bytes: " << n_out * d_out * sizeof(T) << ")" << std::endl;
        fin.close();
        throw; // Re-throw
    }
    
    size_t vector_byte_size = d_out * sizeof(T);
    for(size_t i = 0; i < n_out; ++i){
        if (vector_byte_size > 0) { // Only read if there's something to read for the vector
            fin.read(reinterpret_cast<char*>(data) + i * vector_byte_size, vector_byte_size);
            if (fin.gcount() != static_cast<std::streamsize>(vector_byte_size)) {
                std::cerr << "Error: Failed to read full vector " << i << " from " << data_path
                          << ". Expected " << vector_byte_size << " bytes, got " << fin.gcount() << std::endl;
                delete[] data;
                fin.close();
                exit(1); // Or throw std::runtime_error
            }
        }
    }
    fin.close();

    std::cerr<<"load data "<<data_path<<"\n";
    std::cerr<<"dimension: "<<d_out<<"  number:"<<n_out<<"  size_per_element:"<<sizeof(T)<<"\n";

    return data;
}

struct SearchResult
{
    float recall; // 召回率
    int64_t latency; // 查询延迟 (us)
};

// 封装查询过程的函数模板
template<typename SearchFunc>
std::vector<SearchResult> benchmark_search(
    SearchFunc search_func, // 期望签名: func(const float* query, size_t k) -> std::priority_queue<...>
    float* test_query,      // 指向测试查询数据的指针
    int* test_gt,           // 指向 ground truth 数据的指针
    size_t base_number,     // 基向量数量 (用于信息)
    size_t vecdim,          // 向量维度 (用于计算偏移和信息)
    size_t test_number,     // 要测试的查询数量
    size_t test_gt_d,       // ground truth 的维度 (每个查询的近邻数)
    size_t k,               // 搜索时要返回的近邻数
    bool use_omp_parallel = true // Flag to control OpenMP usage
) {
    std::vector<SearchResult> results(test_number);

    // 准备 ground truth 集合 (在循环外准备)
    std::vector<std::set<uint32_t>> gt_sets(test_number);

    for(size_t i = 0; i < test_number; ++i) {
 
        for(size_t j = 0; j < k && j < test_gt_d; ++j){ // 只取 GT 的前 k 个 (并确保不越界 test_gt_d)
             int t = test_gt[j + i*test_gt_d];
             if (t >= 0) { // 确保索引非负
                gt_sets[i].insert(static_cast<uint32_t>(t));
             }
        }
    }

    // 查询测试代码，遍历查询向量
    // Conditionally enable OpenMP
    #pragma omp parallel for schedule(dynamic) if(use_omp_parallel)
    for(int i = 0; i < static_cast<int>(test_number); ++i) {
        const unsigned long Converter = 1000 * 1000;
        struct timeval val;
        gettimeofday(&val, NULL);

        const float* current_query = test_query + static_cast<size_t>(i) * vecdim;
        // 调用传入的 lambda 或函数对象
        auto res_heap = search_func(current_query, k);

        struct timeval newVal;
        gettimeofday(&newVal, NULL);
        int64_t diff = (newVal.tv_sec * Converter + newVal.tv_usec) - (val.tv_sec * Converter + val.tv_usec);

        // 计算召回率
        size_t acc = 0;
        float recall = 0.0f; // 默认召回率为 0

        if (i < static_cast<int>(gt_sets.size())) {
            const auto& gtset = gt_sets[i];
            if (!gtset.empty() && k > 0) {
                size_t results_count = 0;
                std::vector<uint32_t> result_indices;
                result_indices.reserve(k);

                while (!res_heap.empty() && result_indices.size() < k) {
                    result_indices.push_back(res_heap.top().second);
                    res_heap.pop();
                }
                
                for(uint32_t x : result_indices) {
                     if(gtset.count(x)){ 
                        ++acc;
                    }
                }
                recall = static_cast<float>(acc) / k;
            } else if (k == 0) {
                recall = 1.0f; 
            }
        }
        
        // Use #pragma omp critical if OpenMP is active for this loop
        // If not active, critical is not strictly needed but harmless
        #pragma omp critical
        {
             if (static_cast<size_t>(i) < results.size()) {
                results[i] = {recall, diff};
             }
        }
    }
    return results;
}


// 打印测试结果的辅助函数
void print_results(const std::string& method_name, const std::vector<SearchResult>& results, size_t test_number) {
    
    double total_recall = 0, total_latency = 0; 
    size_t valid_results = 0;

    for(size_t i = 0; i < results.size() && i < test_number; ++i) {
        // Only count valid results if latency is not some error marker, e.g. -1
        // For now, assume all results passed to print_results are valid to count
        total_recall += results[i].recall;
        total_latency += results[i].latency;
        valid_results++;
    }
    
    std::cout << "=== " << method_name << " ===" << std::endl;
    std::cout << std::fixed << std::setprecision(5);
    std::cout << "平均召回率: " << (valid_results > 0 ? total_recall / valid_results : 0.0) << std::endl;
    std::cout << std::fixed << std::setprecision(3); 
    std::cout << "平均延迟 (us): " << (valid_results > 0 ? total_latency / valid_results : 0.0) << std::endl;
    std::cout << std::endl;
}


int main(int argc, char *argv[])
{
    size_t test_number = 0, base_number = 0;
    size_t test_gt_d = 0, vecdim = 0;
    
    std::string data_root_path = "./"; 
    if (argc > 1) { 
        data_root_path = std::string(argv[1]);
        if (data_root_path.back() != '/') {
            data_root_path += "/";
        }
    }
    std::cout << "Using data root path: " << data_root_path << std::endl;

    std::string query_path =   data_root_path + "DEEP100K.query.fbin";
    std::string gt_path =      data_root_path + "DEEP100K.gt.query.100k.top100.bin";
    std::string base_path =    data_root_path + "DEEP100K.base.100k.fbin";


    auto test_query = LoadData<float>(query_path, test_number, vecdim); 
    
    size_t gt_n_from_file; 
    auto test_gt = LoadData<int>(gt_path, gt_n_from_file, test_gt_d); 

    if (gt_n_from_file != test_number && test_number != 0) { 
         std::cout << "Warning: Query number (" << test_number 
                   << ") and GT number (" << gt_n_from_file 
                   << ") mismatch. Using query number for test count." << std::endl;
    }
     // If test_number was updated by query loading, ensure gt_n_from_file is also considered for num_queries_to_test
    if (test_number == 0 && gt_n_from_file > 0) test_number = gt_n_from_file; // If query empty but gt exists
    

    size_t base_vecdim_check; 
    auto base = LoadData<float>(base_path, base_number, base_vecdim_check);
    if (vecdim !=0 && base_vecdim_check != 0 && vecdim != base_vecdim_check) {
        std::cout << "Critical Error: Query dimension (" << vecdim 
                  << ") and Base dimension (" << base_vecdim_check
                  << ") mismatch. Exiting." << std::endl;
        // This is usually a fatal error for ANN.
        delete[] test_query; delete[] test_gt; delete[] base;
        return 1; 
    } else if (vecdim == 0 && base_vecdim_check != 0) {
        vecdim = base_vecdim_check; 
    }
    if (base_number == 0 || vecdim == 0) {
        std::cout << "Critical Error: Base data has 0 vectors or 0 dimension. Exiting." << std::endl;
        delete[] test_query; delete[] test_gt; delete[] base;
        return 1;
    }


    size_t num_queries_to_test = 2000;
    if (test_number == 0) { 
        num_queries_to_test = 0;
        std::cout << "Warning: No queries loaded. Setting num_queries_to_test to 0." << std::endl;
    } else {
        if (num_queries_to_test > test_number) {
            num_queries_to_test = test_number;
        }
        // Also ensure it does not exceed the number of GT entries if GT is smaller
        if (num_queries_to_test > gt_n_from_file && gt_n_from_file > 0) {
             std::cout << "Warning: num_queries_to_test (" << num_queries_to_test 
                      << ") exceeds number of GT entries (" << gt_n_from_file 
                      << "). Clamping to GT entries count." << std::endl;
            num_queries_to_test = gt_n_from_file;
        }
    }
    std::cout << "将测试前 " << num_queries_to_test << " 条查询。" << std::endl;


    const size_t k = 10;
    const int num_pthreads_for_ann = 8; 

    // --- Flat Search ---
    // ... (Flat search benchmark code remains the same) ...
    auto flat_search_lambda = [&](const float* q, size_t k_param) {
        return flat_search(base, q, base_number, vecdim, k_param);
    };
    std::vector<SearchResult> results_flat = benchmark_search(
       flat_search_lambda, 
       test_query, test_gt, base_number, vecdim, num_queries_to_test, test_gt_d, k, true);


    // --- SIMD Search ---
    // ... (SIMD search benchmark code remains the same) ...
    auto simd_search_lambda = [&](const float* q, size_t k_param) {
        return simd_search(base, q, base_number, vecdim, k_param);
    };
    std::vector<SearchResult> results_simd = benchmark_search(
       simd_search_lambda,
       test_query, test_gt, base_number, vecdim, num_queries_to_test, test_gt_d, k, true);

    // --- PQ Search ---
    // ... (PQ search benchmark code remains the same) ...
    size_t pq_nsub_global = 4; // Example, should be vecdim / dsub_pq
    if (vecdim > 0 && vecdim % 4 != 0 && vecdim % 8 == 0) pq_nsub_global = 8; // Adjust if not divisible by 4 but by 8
    else if (vecdim > 0 && vecdim % 4 != 0 && vecdim % 2 == 0) pq_nsub_global = 2;
    else if (vecdim > 0 && vecdim % 4 != 0) { /* pq_nsub_global remains 4, PQ constructor will throw if not divisible */ }


    double pq_train_ratio_global = 1.0;
    size_t pq_rerank_k_global = 600;

    ProductQuantizer* pq_index_ptr = nullptr; 
    try {
        if (base_number > 0 && vecdim > 0 && pq_nsub_global > 0 && vecdim % pq_nsub_global == 0) {
             pq_index_ptr = new ProductQuantizer(base, base_number, vecdim, pq_nsub_global, pq_train_ratio_global);
        } else {
            std::cerr << "Skipping PQ index creation due to invalid parameters (base_number=" << base_number
                      << ", vecdim=" << vecdim << ", pq_nsub_global=" << pq_nsub_global 
                      << ", vecdim % pq_nsub_global = " << (vecdim > 0 && pq_nsub_global > 0 ? vecdim % pq_nsub_global : -1)
                      << ")." << std::endl;
        }
    } catch (const std::exception& e) {
        std::cerr << "Error creating PQ index: " << e.what() << std::endl;
        pq_index_ptr = nullptr;
    }
    
    std::vector<SearchResult> results_pq;
    if (pq_index_ptr) { 
        std::cout << "使用 PQ 参数: nsub=" << pq_nsub_global << ", train_ratio=" << pq_train_ratio_global << ", rerank_k=" << pq_rerank_k_global << std::endl;
        auto pq_search_lambda = [&](const float* q, size_t k_param) {
            // PQ search in pq_anns.h uses L2 distance for ADC. If your ground truth is IP, this is a mismatch.
            // For DEEP1B, IP is common. The flat_search and simd_search use IP.
            // The ProductQuantizer provided uses L2 for its internal k-means and ADC.
            // This might explain recall differences if GT is based on IP.
            // For now, we proceed with L2 as implemented in ProductQuantizer.
            return pq_index_ptr->search(q, base, k_param, pq_rerank_k_global);
        };
        results_pq = benchmark_search(
           pq_search_lambda,
           test_query, test_gt, base_number, vecdim, num_queries_to_test, test_gt_d, k, true);
    } else {
        std::cerr << "跳过 PQ 搜索测试，因为索引创建失败。" << std::endl;
        results_pq.resize(num_queries_to_test, {0.0f, -1}); 
    }


    // --- SQ Search ---
    // ... (SQ search benchmark code remains the same) ...
    ScalarQuantizer* sq_quantizer_ptr = nullptr;
    try {
        if (base_number > 0 && vecdim > 0)
            sq_quantizer_ptr = new ScalarQuantizer(base, base_number, vecdim);
    } catch (const std::exception& e) {
        std::cerr << "Error creating SQ quantizer: " << e.what() << std::endl;
        sq_quantizer_ptr = nullptr;
    }
    
    std::vector<SearchResult> results_sq;
    if (sq_quantizer_ptr) {
        auto sq_search_lambda = [&](const float* q, size_t k_param) {
            return sq_quantizer_ptr->sq_search(q, k_param);
        };
        results_sq = benchmark_search(
           sq_search_lambda,
           test_query, test_gt, base_number, vecdim, num_queries_to_test, test_gt_d, k, true);
    } else {
         std::cerr << "跳过 SQ 搜索测试，因为量化器创建失败。" << std::endl;
         results_sq.resize(num_queries_to_test, {0.0f, -1}); 
    }

    // --- IVF Search (Pthread) ---
    // ... (IVF search benchmark code remains the same) ...
    std::cout << "\n--- IVF (Pthread) 测试 ---" << std::endl;
    size_t num_ivf_clusters_ivf_only = 0;
    if (base_number > 0) {
        num_ivf_clusters_ivf_only = std::min((size_t)256, base_number / 100); 
        if (num_ivf_clusters_ivf_only == 0 && base_number > 0) num_ivf_clusters_ivf_only = std::min((size_t)1, base_number);
    }
    int ivf_kmeans_iterations_ivf_only = 20; 

    IVFIndex* ivf_index_ptr = nullptr;
    if (base_number > 0 && num_ivf_clusters_ivf_only > 0 && vecdim > 0) {
        std::cout << "构建 IVF 索引... num_clusters=" << num_ivf_clusters_ivf_only 
                  << ", pthreads=" << num_pthreads_for_ann 
                  << ", kmeans_iter=" << ivf_kmeans_iterations_ivf_only << std::endl;
        struct timeval build_start, build_end;
        gettimeofday(&build_start, NULL);
        try {
            ivf_index_ptr = new IVFIndex(base, base_number, vecdim, num_ivf_clusters_ivf_only, num_pthreads_for_ann, ivf_kmeans_iterations_ivf_only);
        } catch (const std::exception& e) {
            std::cerr << "Error creating IVF index: " << e.what() << std::endl;
            ivf_index_ptr = nullptr;
        }
        gettimeofday(&build_end, NULL);
        long long build_time_us = (build_end.tv_sec - build_start.tv_sec) * 1000000LL + (build_end.tv_usec - build_start.tv_usec);
        std::cout << "IVF 索引构建时间: " << build_time_us / 1000.0 << " ms" << std::endl;
    } else {
        std::cerr << "无法构建 IVF 索引 (IVF-only)，参数无效。" << std::endl;
    }
    
    if (ivf_index_ptr) {
        std::vector<size_t> nprobe_values = {1, 2, 4, 8, 16, 32}; 
        if (num_ivf_clusters_ivf_only < 32 && num_ivf_clusters_ivf_only > 0) { 
            nprobe_values.clear();
            for(size_t np_val = 1; np_val <= num_ivf_clusters_ivf_only; np_val *=2) nprobe_values.push_back(np_val);
            if (nprobe_values.empty() || nprobe_values.back() < num_ivf_clusters_ivf_only) {
                 bool contains_max = false;
                 for(size_t val : nprobe_values) if(val == num_ivf_clusters_ivf_only) contains_max = true;
                 if(!contains_max) nprobe_values.push_back(num_ivf_clusters_ivf_only);
            }
            if (nprobe_values.empty()) nprobe_values.push_back(1);
        } else if (num_ivf_clusters_ivf_only == 0) {
            nprobe_values.clear(); // No clusters, no nprobes
        }


        for (size_t current_nprobe : nprobe_values) {
            if (current_nprobe == 0) continue;
            size_t actual_nprobe = std::min(current_nprobe, num_ivf_clusters_ivf_only);
            if (actual_nprobe == 0 && num_ivf_clusters_ivf_only > 0) actual_nprobe = 1; // Ensure at least 1 if possible
            else if (num_ivf_clusters_ivf_only == 0) continue; // Skip if no clusters

            std::cout << "测试 IVF (Pthread) 使用 nprobe = " << actual_nprobe << std::endl;
            auto ivf_search_lambda = [&](const float* q, size_t k_param) {
                return ivf_index_ptr->search(q, k_param, actual_nprobe);
            };
            std::vector<SearchResult> results_ivf = benchmark_search(
               ivf_search_lambda,
               test_query, test_gt, base_number, vecdim, num_queries_to_test, test_gt_d, k, false); 
            
            std::string ivf_method_name = "IVF (Pthread, nprobe=" + std::to_string(actual_nprobe) + 
                                          ", clusters=" + std::to_string(num_ivf_clusters_ivf_only) + ")";
            print_results(ivf_method_name, results_ivf, num_queries_to_test);
        }
    } else {
        std::cerr << "跳过 IVF (Pthread) 搜索测试，因为索引创建失败。" << std::endl;
    }

    // --- IVF + PQ Search (Pthread) ---
    std::cout << "\n--- IVFADC (IVF+PQ, Pthread) 测试 ---" << std::endl;
    size_t num_ivf_clusters_ivfpq = 64; // Fewer clusters for IVFADC typically, as PQ handles fine-grain
    if (base_number > 0 && num_ivf_clusters_ivfpq > base_number / 10) { // Heuristic
        num_ivf_clusters_ivfpq = std::max((size_t)16, base_number / 100);
    }
    if (num_ivf_clusters_ivfpq == 0 && base_number > 0) num_ivf_clusters_ivfpq = std::min((size_t)1, base_number);

    size_t pq_nsub_ivfpq = pq_nsub_global; // Use same PQ nsub as standalone PQ for comparison
    double pq_train_ratio_ivfpq = pq_train_ratio_global;
    int ivf_kmeans_iter_ivfpq = 20;

    IVFPQIndex* ivfpq_index_ptr = nullptr;
    if (base_number > 0 && vecdim > 0 && num_ivf_clusters_ivfpq > 0 && pq_nsub_ivfpq > 0 && vecdim % pq_nsub_ivfpq == 0) {
        std::cout << "构建 IVFADC 索引... IVF_clusters=" << num_ivf_clusters_ivfpq
                  << ", PQ_nsub=" << pq_nsub_ivfpq
                  << ", pthreads=" << num_pthreads_for_ann
                  << ", ivf_kmeans_iter=" << ivf_kmeans_iter_ivfpq << std::endl;
        struct timeval build_start_ivfpq, build_end_ivfpq;
        gettimeofday(&build_start_ivfpq, NULL);
        try {
            // Corrected constructor call
            ivfpq_index_ptr = new IVFPQIndex(vecdim,
                                             num_ivf_clusters_ivfpq,
                                             pq_nsub_ivfpq,
                                             num_pthreads_for_ann,
                                             ivf_kmeans_iter_ivfpq);
            // Call build method after successful construction
            ivfpq_index_ptr->build(base, base_number, pq_train_ratio_ivfpq);

        } catch (const std::exception& e) {
            std::cerr << "Error creating or building IVFPQ index: " << e.what() << std::endl;
            if (ivfpq_index_ptr) { // If object was constructed (even if build failed)
                delete ivfpq_index_ptr;
            }
            ivfpq_index_ptr = nullptr; // Mark as unusable
        }
        gettimeofday(&build_end_ivfpq, NULL);
        if (ivfpq_index_ptr) { // Only print time if successful
            long long build_time_us_ivfpq = (build_end_ivfpq.tv_sec - build_start_ivfpq.tv_sec) * 1000000LL +
                                           (build_end_ivfpq.tv_usec - build_start_ivfpq.tv_usec);
            std::cout << "IVFADC 索引构建时间: " << build_time_us_ivfpq / 1000.0 << " ms" << std::endl;
        }
    } else {
         std::cerr << "无法构建 IVFADC 索引，参数无效 (base_number="<<base_number
                   <<", vecdim="<<vecdim<<", ivf_clusters="<<num_ivf_clusters_ivfpq
                   <<", pq_nsub="<<pq_nsub_ivfpq <<")." << std::endl;
    }

    if (ivfpq_index_ptr) {
        std::vector<size_t> nprobe_values_ivfpq = {1, 2, 4, 8, 16};
        if (num_ivf_clusters_ivfpq < 16 && num_ivf_clusters_ivfpq > 0) {
            nprobe_values_ivfpq.clear();
            for(size_t np_val = 1; np_val <= num_ivf_clusters_ivfpq; np_val *=2) nprobe_values_ivfpq.push_back(np_val);
             if (nprobe_values_ivfpq.empty() || nprobe_values_ivfpq.back() < num_ivf_clusters_ivfpq) {
                 bool contains_max = false;
                 for(size_t val : nprobe_values_ivfpq) if(val == num_ivf_clusters_ivfpq) contains_max = true;
                 if(!contains_max) nprobe_values_ivfpq.push_back(num_ivf_clusters_ivfpq);
             }
            if (nprobe_values_ivfpq.empty()) nprobe_values_ivfpq.push_back(1);
        } else if (num_ivf_clusters_ivfpq == 0) {
            nprobe_values_ivfpq.clear();
        }

        size_t ivfpq_rerank_k = 600;
        for (size_t current_nprobe : nprobe_values_ivfpq) {
            if (current_nprobe == 0) continue;
            size_t actual_nprobe = std::min(current_nprobe, num_ivf_clusters_ivfpq);
            if (actual_nprobe == 0 && num_ivf_clusters_ivfpq > 0) actual_nprobe = 1;
            else if (num_ivf_clusters_ivfpq == 0) continue;


            std::cout << "测试 IVFADC (Pthread) 使用 nprobe = " << actual_nprobe << ", rerank_k = " << ivfpq_rerank_k << std::endl;
            auto ivfpq_search_lambda = [&](const float* q, size_t k_param) {
                return ivfpq_index_ptr->search(q,          // query
                                               base,       // base_data_for_reranking
                                               k_param,    // k
                                               actual_nprobe, // nprobe
                                               ivfpq_rerank_k); // rerank_k_candidates
            };
            std::vector<SearchResult> results_ivfpq = benchmark_search(
               ivfpq_search_lambda,
               test_query, test_gt, base_number, vecdim, num_queries_to_test, test_gt_d, k, false);

            std::string ivfpq_method_name = "IVFADC (nprobe=" + std::to_string(actual_nprobe) +
                                            ", IVFclus=" + std::to_string(num_ivf_clusters_ivfpq) +
                                            ", PQnsub=" + std::to_string(pq_nsub_ivfpq) +
                                            ", rerank_k=" + std::to_string(ivfpq_rerank_k) + ")";
            print_results(ivfpq_method_name, results_ivfpq, num_queries_to_test);
        }
    } else {
        std::cerr << "跳过 IVFADC (Pthread) 搜索测试，因为索引创建失败。" << std::endl;
    }


    // --- 打印结果 ---
    // ... (Print results for Flat, SIMD, PQ, SQ, IVF-only) ...
    std::cout << "\n--- 最终结果汇总 ---" << std::endl;
    print_results("Flat Search (暴力搜索)", results_flat, num_queries_to_test);
    print_results("SIMD Search (SIMD优化)", results_simd, num_queries_to_test);
    if (pq_index_ptr) { 
       print_results("PQ Search (乘积量化, rerank=" + std::to_string(pq_rerank_k_global) + ")", results_pq, num_queries_to_test);
    } else {
       print_results("PQ Search (跳过)", results_pq, num_queries_to_test);
    }
    if (sq_quantizer_ptr) { 
        print_results("SQ Search (标量量化)", results_sq, num_queries_to_test);
    } else {
        print_results("SQ Search (跳过)", results_sq, num_queries_to_test);
    }
    // IVF-only results already printed
    // IVFPQ results also already printed


    // --- 清理 ---
    delete[] test_query;
    delete[] test_gt;
    delete[] base;
    delete pq_index_ptr; 
    delete sq_quantizer_ptr;
    delete ivf_index_ptr;
    delete ivfpq_index_ptr; // Cleanup new index

    return 0;
}