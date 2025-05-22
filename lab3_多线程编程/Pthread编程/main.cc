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
#include <omp.h> // 用于 OMP_NUM_THREADS，但要注意 Pthread 的使用
// #include "hnswlib/hnswlib/hnswlib.h" 
#include "flat_scan.h" 
#include "simd_anns.h"
#include "pq_anns.h" 
#include "sq_anns.h"
#include "ivf_anns.h" // 包含新的 IVF 头文件
#include "ivf_openmp.h" // 包含新的 IVF OpenMP 头文件

#include <functional>
#include <algorithm> 
#include <queue>     
#include <stdexcept> 

#include "ivf_pq_anns.h" // 包含新的 IVF-PQ 头文件
#include "ivf_pq_v1_anns.h" // 包含新的 IVF-PQ V1 头文件
#include "ivf_pq_openmp_anns.h" // <<< 新增: 包含 OpenMP 版本的 IVF+PQ 头文件
// --- 函数声明  ---
std::priority_queue<std::pair<float, uint32_t>> flat_search(float* base, const float* query, size_t base_number, size_t vecdim, size_t k);
std::priority_queue<std::pair<float, uint32_t>> simd_search(float* base, const float* query, size_t base_number, size_t vecdim, size_t k);
// ---
template<typename T>
T *LoadData(std::string data_path, size_t& n_out, size_t& d_out) // 重命名以避免混淆
{
    std::ifstream fin;
    fin.open(data_path, std::ios::in | std::ios::binary);
    if (!fin.is_open()) {
        std::cerr << "打开数据文件时出错: " << data_path << std::endl;
        exit(1); // 或者抛出 std::runtime_error 以进行更好的错误处理
    }

    uint32_t n_file, d_file; // 使用固定大小的类型读取文件元数据

    fin.read(reinterpret_cast<char*>(&n_file), sizeof(uint32_t));
    if (!fin) {
        std::cerr << "从数据文件读取 'n' 时出错: " << data_path << std::endl;
        fin.close();
        exit(1);
    }
    fin.read(reinterpret_cast<char*>(&d_file), sizeof(uint32_t));
    if (!fin) {
        std::cerr << "从数据文件读取 'd' 时出错: " << data_path << std::endl;
        fin.close();
        exit(1);
    }

    n_out = static_cast<size_t>(n_file);
    d_out = static_cast<size_t>(d_file);

    if (n_out == 0 || d_out == 0) {
        std::cerr << "警告: " << data_path << " 的 n 或 d 为零。n=" << n_out << ", d=" << d_out << std::endl;
        // 分配一个零大小的数组或返回 nullptr，确保下游代码处理它。
        // new T[0] 在 C++ 中是有效的。
    }
    
    // 在乘法计算分配大小之前检查潜在的溢出
    if (n_out > 0 && d_out > 0 && (n_out > std::numeric_limits<size_t>::max() / d_out) ) {
        std::cerr << "错误: " << data_path << " 的 n*d 将导致 size_t 溢出。n=" << n_out << ", d=" << d_out << std::endl;
        fin.close();
        exit(1); // 或者抛出 std::overflow_error
    }

    T* data = nullptr;
    try {
        data = new T[n_out * d_out];
    } catch (const std::bad_alloc& e) {
        std::cerr << "尝试为 " << data_path 
                  << " 分配内存时捕获到 std::bad_alloc，n=" << n_out << ", d=" << d_out 
                  << " (总元素数: " << n_out * d_out 
                  << ", 字节数: " << n_out * d_out * sizeof(T) << ")" << std::endl;
        fin.close();
        throw; // 重新抛出
    }
    
    size_t vector_byte_size = d_out * sizeof(T);
    for(size_t i = 0; i < n_out; ++i){
        if (vector_byte_size > 0) { // 仅当向量有内容可读时才读取
            fin.read(reinterpret_cast<char*>(data) + i * vector_byte_size, vector_byte_size);
            if (fin.gcount() != static_cast<std::streamsize>(vector_byte_size)) {
                std::cerr << "错误: 未能从 " << data_path << " 读取完整的向量 " << i
                          << "。期望 " << vector_byte_size << " 字节，实际读取 " << fin.gcount() << std::endl;
                delete[] data;
                fin.close();
                exit(1); // 或者抛出 std::runtime_error
            }
        }
    }
    fin.close();

    std::cerr<<"加载数据 "<<data_path<<"\n";
    std::cerr<<"维度: "<<d_out<<"  数量:"<<n_out<<"  每元素大小:"<<sizeof(T)<<"\n";

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
    bool use_omp_parallel = true // 控制 OpenMP 使用的标志
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
    // 条件性启用 OpenMP
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
        
        // 如果此循环的 OpenMP 处于活动状态，则使用 #pragma omp critical
        // 如果未激活，critical 不是严格必需的，但无害
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
        // 仅当延迟不是某个错误标记（例如 -1）时才计算有效结果
        // 目前，假设传递给 print_results 的所有结果都是有效的
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
    std::cout << "使用数据根路径: " << data_root_path << std::endl;

    std::string query_path =   data_root_path + "DEEP100K.query.fbin";
    std::string gt_path =      data_root_path + "DEEP100K.gt.query.100k.top100.bin";
    std::string base_path =    data_root_path + "DEEP100K.base.100k.fbin";


    auto test_query = LoadData<float>(query_path, test_number, vecdim); 
    
    size_t gt_n_from_file; 
    auto test_gt = LoadData<int>(gt_path, gt_n_from_file, test_gt_d); 

    if (gt_n_from_file != test_number && test_number != 0) { 
         std::cout << "警告: 查询数量 (" << test_number 
                   << ") 和 GT 数量 (" << gt_n_from_file 
                   << ") 不匹配。使用查询数量作为测试计数。" << std::endl;
    }
     // 如果 test_number 被查询加载更新，确保 gt_n_from_file 也被考虑用于 num_queries_to_test
    if (test_number == 0 && gt_n_from_file > 0) test_number = gt_n_from_file; // 如果查询为空但 gt 存在
    

    size_t base_vecdim_check; 
    auto base = LoadData<float>(base_path, base_number, base_vecdim_check);
    if (vecdim !=0 && base_vecdim_check != 0 && vecdim != base_vecdim_check) {
        std::cout << "严重错误: 查询维度 (" << vecdim 
                  << ") 和基准维度 (" << base_vecdim_check
                  << ") 不匹配。正在退出。" << std::endl;
        // 这通常是 ANN 的致命错误。
        delete[] test_query; delete[] test_gt; delete[] base;
        return 1; 
    } else if (vecdim == 0 && base_vecdim_check != 0) {
        vecdim = base_vecdim_check; 
    }
    if (base_number == 0 || vecdim == 0) {
        std::cout << "严重错误: 基准数据包含 0 个向量或 0 维度。正在退出。" << std::endl;
        delete[] test_query; delete[] test_gt; delete[] base;
        return 1;
    }


    size_t num_queries_to_test = 2000;
    if (test_number == 0) { 
        num_queries_to_test = 0;
        std::cout << "警告: 未加载查询。将 num_queries_to_test 设置为 0。" << std::endl;
    } else {
        if (num_queries_to_test > test_number) {
            num_queries_to_test = test_number;
        }
        // 同时确保它不超过 GT 条目的数量（如果 GT 较小）
        if (num_queries_to_test > gt_n_from_file && gt_n_from_file > 0) {
             std::cout << "警告: num_queries_to_test (" << num_queries_to_test 
                      << ") 超过 GT 条目数 (" << gt_n_from_file 
                      << ")。将其限制为 GT 条目数。" << std::endl;
            num_queries_to_test = gt_n_from_file;
        }
    }
    std::cout << "将测试前 " << num_queries_to_test << " 条查询。" << std::endl;


    const size_t k = 10;
    const int num_pthreads_for_ann = 8; 

    // --- Flat 搜索 ---
    // ... (Flat 搜索基准测试代码保持不变) ...
    auto flat_search_lambda = [&](const float* q, size_t k_param) {
        return flat_search(base, q, base_number, vecdim, k_param);
    };
    std::vector<SearchResult> results_flat = benchmark_search(
       flat_search_lambda, 
       test_query, test_gt, base_number, vecdim, num_queries_to_test, test_gt_d, k, true);


    // --- SIMD 搜索 ---
    // ... (SIMD 搜索基准测试代码保持不变) ...
    auto simd_search_lambda = [&](const float* q, size_t k_param) {
        return simd_search(base, q, base_number, vecdim, k_param);
    };
    std::vector<SearchResult> results_simd = benchmark_search(
       simd_search_lambda,
       test_query, test_gt, base_number, vecdim, num_queries_to_test, test_gt_d, k, true);

    // --- PQ 搜索 ---
    // ... (PQ 搜索基准测试代码保持不变) ...
    size_t pq_nsub_global = 4; // 示例，应为 vecdim / dsub_pq
    if (vecdim > 0 && vecdim % 4 != 0 && vecdim % 8 == 0) pq_nsub_global = 8; // 如果不能被 4 整除但能被 8 整除，则调整
    else if (vecdim > 0 && vecdim % 4 != 0 && vecdim % 2 == 0) pq_nsub_global = 2;
    else if (vecdim > 0 && vecdim % 4 != 0) { /* pq_nsub_global 保持为 4, 如果不能整除，PQ 构造函数将抛出异常 */ }


    double pq_train_ratio_global = 1.0;
    size_t pq_rerank_k_global = 600;

    ProductQuantizer* pq_index_ptr = nullptr; 
    try {
        if (base_number > 0 && vecdim > 0 && pq_nsub_global > 0 && vecdim % pq_nsub_global == 0) {
             pq_index_ptr = new ProductQuantizer(base, base_number, vecdim, pq_nsub_global, pq_train_ratio_global);
        } else {
            std::cerr << "由于参数无效，跳过 PQ 索引创建 (base_number=" << base_number
                      << ", vecdim=" << vecdim << ", pq_nsub_global=" << pq_nsub_global 
                      << ", vecdim % pq_nsub_global = " << (vecdim > 0 && pq_nsub_global > 0 ? vecdim % pq_nsub_global : -1)
                      << ")." << std::endl;
        }
    } catch (const std::exception& e) {
        std::cerr << "创建 PQ 索引时出错: " << e.what() << std::endl;
        pq_index_ptr = nullptr;
    }
    
    std::vector<SearchResult> results_pq;
    if (pq_index_ptr) { 
        std::cout << "使用 PQ 参数: nsub=" << pq_nsub_global << ", train_ratio=" << pq_train_ratio_global << ", rerank_k=" << pq_rerank_k_global << std::endl;
        auto pq_search_lambda = [&](const float* q, size_t k_param) {
            // pq_anns.h 中的 PQ 搜索使用 L2 距离进行 ADC。如果您的 ground truth 是 IP，则存在不匹配。
            // 对于 DEEP1B，IP 很常见。flat_search 和 simd_search 使用 IP。
            // 提供的 ProductQuantizer 对其内部 k-means 和 ADC 使用 L2。
            // 如果 GT 基于 IP，这可能解释召回率的差异。
            // 目前，我们按照 ProductQuantizer 中实现的方式使用 L2。
            return pq_index_ptr->search(q, base, k_param, pq_rerank_k_global);
        };
        results_pq = benchmark_search(
           pq_search_lambda,
           test_query, test_gt, base_number, vecdim, num_queries_to_test, test_gt_d, k, true);
    } else {
        std::cerr << "跳过 PQ 搜索测试，因为索引创建失败。" << std::endl;
        results_pq.resize(num_queries_to_test, {0.0f, -1}); 
    }


    // --- SQ 搜索 ---
    // ... (SQ 搜索基准测试代码保持不变) ...
    ScalarQuantizer* sq_quantizer_ptr = nullptr;
    try {
        if (base_number > 0 && vecdim > 0)
            sq_quantizer_ptr = new ScalarQuantizer(base, base_number, vecdim);
    } catch (const std::exception& e) {
        std::cerr << "创建 SQ 量化器时出错: " << e.what() << std::endl;
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

    // --- IVF 搜索 (Pthread) ---
    // ... (IVF 搜索基准测试代码保持不变) ...
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
            std::cerr << "创建 IVF 索引时出错: " << e.what() << std::endl;
            ivf_index_ptr = nullptr;
        }
        gettimeofday(&build_end, NULL);
        long long build_time_us = (build_end.tv_sec - build_start.tv_sec) * 1000000LL + (build_end.tv_usec - build_start.tv_usec);
        std::cout << "IVF 索引构建时间: " << build_time_us / 1000.0 << " ms" << std::endl;
    } else {
        std::cerr << "无法构建 IVF 索引 (仅 IVF)，参数无效。" << std::endl;
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
            nprobe_values.clear(); // 没有簇，就没有 nprobe
        }


        for (size_t current_nprobe : nprobe_values) {
            if (current_nprobe == 0) continue;
            size_t actual_nprobe = std::min(current_nprobe, num_ivf_clusters_ivf_only);
            if (actual_nprobe == 0 && num_ivf_clusters_ivf_only > 0) actual_nprobe = 1; // 如果可能，确保至少为 1
            else if (num_ivf_clusters_ivf_only == 0) continue; // 如果没有簇则跳过

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

    // --- IVF 搜索 (OpenMP) ---
    std::cout << "\n--- IVF (OpenMP) 测试 ---" << std::endl;
    size_t num_ivf_clusters_omp = num_ivf_clusters_ivf_only; // 为比较起见，重用 Pthread 的配置
    int ivf_kmeans_iterations_omp = ivf_kmeans_iterations_ivf_only;

    IVFIndexOpenMP* ivf_omp_index_ptr = nullptr;
    if (base_number > 0 && num_ivf_clusters_omp > 0 && vecdim > 0) {
        std::cout << "构建 IVF (OpenMP) 索引... num_clusters=" << num_ivf_clusters_omp
                  << ", threads=" << num_pthreads_for_ann // 使用相同的线程数变量
                  << ", kmeans_iter=" << ivf_kmeans_iterations_omp << std::endl;
        struct timeval build_start_omp, build_end_omp;
        gettimeofday(&build_start_omp, NULL);
        try {
            ivf_omp_index_ptr = new IVFIndexOpenMP(base, base_number, vecdim, num_ivf_clusters_omp, num_pthreads_for_ann, ivf_kmeans_iterations_omp);
        } catch (const std::exception& e) {
            std::cerr << "创建 IVF (OpenMP) 索引时出错: " << e.what() << std::endl;
            ivf_omp_index_ptr = nullptr;
        }
        gettimeofday(&build_end_omp, NULL);
        if (ivf_omp_index_ptr) {
            long long build_time_us_omp = (build_end_omp.tv_sec - build_start_omp.tv_sec) * 1000000LL + (build_end_omp.tv_usec - build_start_omp.tv_usec);
            std::cout << "IVF (OpenMP) 索引构建时间: " << build_time_us_omp / 1000.0 << " ms" << std::endl;
        }
    } else {
        std::cerr << "无法构建 IVF (OpenMP) 索引，参数无效 (base_number=" << base_number
                  << ", vecdim=" << vecdim << ", num_ivf_clusters_omp=" << num_ivf_clusters_omp << ")." << std::endl;
    }

    if (ivf_omp_index_ptr) {
        std::vector<size_t> nprobe_values_omp = {1, 2, 4, 8, 16, 32}; // 直接初始化 nprobe_values_omp
        if (num_ivf_clusters_omp < 32 && num_ivf_clusters_omp > 0) { // 如果簇数不同则调整
            nprobe_values_omp.clear();
            for(size_t np_val = 1; np_val <= num_ivf_clusters_omp; np_val *=2) nprobe_values_omp.push_back(np_val);
            if (nprobe_values_omp.empty() || nprobe_values_omp.back() < num_ivf_clusters_omp) {
                 bool contains_max = false;
                 for(size_t val : nprobe_values_omp) if(val == num_ivf_clusters_omp) contains_max = true;
                 if(!contains_max && num_ivf_clusters_omp > 0) nprobe_values_omp.push_back(num_ivf_clusters_omp);
            }
            if (nprobe_values_omp.empty() && num_ivf_clusters_omp > 0) nprobe_values_omp.push_back(1);
            else if (num_ivf_clusters_omp == 0) nprobe_values_omp.clear();
        } else if (num_ivf_clusters_omp == 0) {
            nprobe_values_omp.clear();
        }


        for (size_t current_nprobe : nprobe_values_omp) {
            if (current_nprobe == 0 && num_ivf_clusters_omp > 0) continue;
            size_t actual_nprobe = (num_ivf_clusters_omp == 0) ? 0 : std::min(current_nprobe, num_ivf_clusters_omp);
            if (actual_nprobe == 0 && num_ivf_clusters_omp > 0) actual_nprobe = 1;
            else if (num_ivf_clusters_omp == 0) continue;


            std::cout << "测试 IVF (OpenMP) 使用 nprobe = " << actual_nprobe << std::endl;
            auto ivf_omp_search_lambda = [&](const float* q, size_t k_param) {
                return ivf_omp_index_ptr->search(q, k_param, actual_nprobe);
            };
            // 设置 use_omp_parallel 为 false，因为 IVFIndexOpenMP 处理其自身的并行性。
            std::vector<SearchResult> results_ivf_omp = benchmark_search(
               ivf_omp_search_lambda,
               test_query, test_gt, base_number, vecdim, num_queries_to_test, test_gt_d, k, false); 
            
            std::string ivf_omp_method_name = "IVF (OpenMP, nprobe=" + std::to_string(actual_nprobe) + 
                                          ", clusters=" + std::to_string(num_ivf_clusters_omp) + ")";
            print_results(ivf_omp_method_name, results_ivf_omp, num_queries_to_test);
        }
    } else {
        std::cerr << "跳过 IVF (OpenMP) 搜索测试，因为索引创建失败。" << std::endl;
    }


    // --- IVF + PQ 搜索 (Pthread) ---
    std::cout << "\n--- IVFADC (IVF+PQ, Pthread) 测试 ---" << std::endl;
    size_t num_ivf_clusters_ivfpq = 64; // IVFADC 的簇数通常较少，因为 PQ 处理细粒度
    if (base_number > 0 && num_ivf_clusters_ivfpq > base_number / 10) { // 启发式
        num_ivf_clusters_ivfpq = std::max((size_t)16, base_number / 100);
    }
    if (num_ivf_clusters_ivfpq == 0 && base_number > 0) num_ivf_clusters_ivfpq = std::min((size_t)1, base_number);

    size_t pq_nsub_ivfpq = pq_nsub_global; // 为比较起见，使用与独立 PQ 相同的 PQ nsub
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
            // 校正的构造函数调用
            ivfpq_index_ptr = new IVFPQIndex(vecdim,
                                             num_ivf_clusters_ivfpq,
                                             pq_nsub_ivfpq,
                                             num_pthreads_for_ann,
                                             ivf_kmeans_iter_ivfpq);
            // 成功构造后调用 build 方法
            ivfpq_index_ptr->build(base, base_number, pq_train_ratio_ivfpq);

        } catch (const std::exception& e) {
            std::cerr << "创建或构建 IVFPQ 索引时出错: " << e.what() << std::endl;
            if (ivfpq_index_ptr) { // 如果对象已构造（即使构建失败）
                delete ivfpq_index_ptr;
            }
            ivfpq_index_ptr = nullptr; // 标记为不可用
        }
        gettimeofday(&build_end_ivfpq, NULL);
        if (ivfpq_index_ptr) { // 仅在成功时打印时间
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

        size_t ivfpq_rerank_k_global = 600;
        for (size_t current_nprobe : nprobe_values_ivfpq) {
            if (current_nprobe == 0) continue;
            size_t actual_nprobe = std::min(current_nprobe, num_ivf_clusters_ivfpq);
            if (actual_nprobe == 0 && num_ivf_clusters_ivfpq > 0) actual_nprobe = 1;
            else if (num_ivf_clusters_ivfpq == 0) continue;


            std::cout << "测试 IVFADC (，第二种，Pthread) 使用 nprobe = " << actual_nprobe << ", rerank_k = " << ivfpq_rerank_k_global << std::endl;
            auto ivfpq_search_lambda = [&](const float* q, size_t k_param) {
                return ivfpq_index_ptr->search(q,          // 查询
                                               base,       // 用于重排序的基准数据
                                               k_param,    // k
                                               actual_nprobe, // nprobe
                                               ivfpq_rerank_k_global); // rerank_k_candidates
            };
            std::vector<SearchResult> results_ivfpq = benchmark_search(
               ivfpq_search_lambda,
               test_query, test_gt, base_number, vecdim, num_queries_to_test, test_gt_d, k, false);

            std::string ivfpq_method_name = "IVFADC (nprobe=" + std::to_string(actual_nprobe) +
                                            ", IVFclus=" + std::to_string(num_ivf_clusters_ivfpq) +
                                            ", PQnsub=" + std::to_string(pq_nsub_ivfpq) +
                                            ", rerank_k=" + std::to_string(ivfpq_rerank_k_global) + ")";
            print_results(ivfpq_method_name, results_ivfpq, num_queries_to_test);
        }
    } else {
        std::cerr << "跳过 IVFADC (Pthread) 搜索测试，因为索引创建失败。" << std::endl;
    }


    // --- IVF + PQ 搜索 (Pthread, 方法 1: 先 PQ，然后在重构数据上进行 IVF) ---
    std::cout << "\n--- IVFADC (IVF+PQ, Pthread, 方法 1: PQ 然后 IVF) 测试 ---" << std::endl;
    size_t num_ivf_clusters_ivfpq_v1 = 64; // 可以调整，例如与其他 IVFPQ 相同
    if (base_number > 0 && num_ivf_clusters_ivfpq_v1 > base_number / 10) {
        num_ivf_clusters_ivfpq_v1 = std::max((size_t)16, base_number / 100);
    }
    if (num_ivf_clusters_ivfpq_v1 == 0 && base_number > 0) num_ivf_clusters_ivfpq_v1 = std::min((size_t)1, base_number);
    
    size_t pq_nsub_ivfpq_v1 = pq_nsub_global; 
    double pq_train_ratio_ivfpq_v1 = pq_train_ratio_global;
    int ivf_kmeans_iter_ivfpq_v1 = 20;

    IVFPQIndexV1* ivfpq_v1_index_ptr = nullptr;
    if (base_number > 0 && vecdim > 0 && /*num_ivf_clusters_ivfpq_v1 > 0 &&*/ pq_nsub_ivfpq_v1 > 0 && vecdim % pq_nsub_ivfpq_v1 == 0) {
        std::cout << "构建 IVFADC (方法 1) 索引... IVF_clusters=" << num_ivf_clusters_ivfpq_v1
                  << ", PQ_nsub=" << pq_nsub_ivfpq_v1
                  << ", pthreads=" << num_pthreads_for_ann
                  << ", ivf_kmeans_iter=" << ivf_kmeans_iter_ivfpq_v1 << std::endl;
        struct timeval build_start_ivfpq_v1, build_end_ivfpq_v1;
        gettimeofday(&build_start_ivfpq_v1, NULL);
        try {
            ivfpq_v1_index_ptr = new IVFPQIndexV1(vecdim,
                                                 num_ivf_clusters_ivfpq_v1,
                                                 pq_nsub_ivfpq_v1,
                                                 num_pthreads_for_ann,
                                                 ivf_kmeans_iter_ivfpq_v1);
            ivfpq_v1_index_ptr->build(base, base_number, pq_train_ratio_ivfpq_v1);

        } catch (const std::exception& e) {
            std::cerr << "创建或构建 IVFPQ (方法 1) 索引时出错: " << e.what() << std::endl;
            delete ivfpq_v1_index_ptr; // 如果构建在构造后部分失败，确保清理
            ivfpq_v1_index_ptr = nullptr;
        }
        gettimeofday(&build_end_ivfpq_v1, NULL);
        if (ivfpq_v1_index_ptr) { 
            long long build_time_us_ivfpq_v1 = (build_end_ivfpq_v1.tv_sec - build_start_ivfpq_v1.tv_sec) * 1000000LL +
                                           (build_end_ivfpq_v1.tv_usec - build_start_ivfpq_v1.tv_usec);
            std::cout << "IVFADC (方法 1) 索引构建时间: " << build_time_us_ivfpq_v1 / 1000.0 << " ms" << std::endl;
        }
    } else {
         std::cerr << "无法构建 IVFADC (方法 1) 索引，参数无效 (base_number="<<base_number
                   <<", vecdim="<<vecdim<<", ivf_clusters="<<num_ivf_clusters_ivfpq_v1
                   <<", pq_nsub="<<pq_nsub_ivfpq_v1 <<")." << std::endl;
    }

    if (ivfpq_v1_index_ptr) {
        std::vector<size_t> nprobe_values_ivfpq_v1 = {1, 2, 4, 8, 16};
        if (num_ivf_clusters_ivfpq_v1 > 0 && num_ivf_clusters_ivfpq_v1 < 16 ) {
            nprobe_values_ivfpq_v1.clear();
            for(size_t np_val = 1; np_val <= num_ivf_clusters_ivfpq_v1; np_val *=2) {
                nprobe_values_ivfpq_v1.push_back(np_val);
            }
            if (nprobe_values_ivfpq_v1.empty() && num_ivf_clusters_ivfpq_v1 > 0) nprobe_values_ivfpq_v1.push_back(1); // 如果簇数 > 0，确保至少有一个 nprobe
            else if (num_ivf_clusters_ivfpq_v1 == 0) nprobe_values_ivfpq_v1.clear(); // 如果没有簇，则没有 nprobe
        } else if (num_ivf_clusters_ivfpq_v1 == 0) {
             nprobe_values_ivfpq_v1.clear(); // 如果没有簇，则没有 nprobe
        }


        size_t ivfpq_v1_rerank_k = pq_rerank_k_global; // 为比较起见，使用相同的 rerank_k
        for (size_t current_nprobe : nprobe_values_ivfpq_v1) {
            if (current_nprobe == 0 && num_ivf_clusters_ivfpq_v1 > 0) continue; // 如果 IVF 处于活动状态，则跳过 nprobe=0
            size_t actual_nprobe = (num_ivf_clusters_ivfpq_v1 == 0) ? 0 : std::min(current_nprobe, num_ivf_clusters_ivfpq_v1);
             if (actual_nprobe == 0 && num_ivf_clusters_ivfpq_v1 > 0) actual_nprobe = 1; // 如果 IVF 处于活动状态且 current_nprobe 为 0，则默认为 1
             else if (num_ivf_clusters_ivfpq_v1 == 0) actual_nprobe = 0; // 如果没有簇，确保 nprobe 为 0


            std::cout << "测试 IVFADC (方法 1, Pthread) 使用 nprobe = " << actual_nprobe << ", rerank_k = " << ivfpq_v1_rerank_k << std::endl;
            auto ivfpq_v1_search_lambda = [&](const float* q, size_t k_param) {
                return ivfpq_v1_index_ptr->search(q, base, k_param, actual_nprobe, ivfpq_v1_rerank_k);
            };
            std::vector<SearchResult> results_ivfpq_v1 = benchmark_search(
               ivfpq_v1_search_lambda,
               test_query, test_gt, base_number, vecdim, num_queries_to_test, test_gt_d, k, false); // pthreads 为 false
            
            std::string ivfpq_v1_method_name = "IVFADC (M1, Pthread, nprobe=" + std::to_string(actual_nprobe) + 
                                          ", clusters=" + std::to_string(num_ivf_clusters_ivfpq_v1) + 
                                          ", rerank_k=" + std::to_string(ivfpq_v1_rerank_k) + ")";
            print_results(ivfpq_v1_method_name, results_ivfpq_v1, num_queries_to_test);
        }
         // 特殊情况：如果 IVF 部分被跳过 (num_ivf_clusters_ivfpq_v1 == 0)，则测试 nprobe=0
        if (num_ivf_clusters_ivfpq_v1 == 0) {
            std::cout << "测试 IVFADC (方法 1, Pthread) 使用 nprobe = 0 (由于 IVF 未激活，进行完整 PQ 扫描)" 
                      << ", rerank_k = " << ivfpq_v1_rerank_k << std::endl;
            auto ivfpq_v1_search_lambda_full_scan = [&](const float* q, size_t k_param) {
                return ivfpq_v1_index_ptr->search(q, base, k_param, 0, ivfpq_v1_rerank_k);
            };
            std::vector<SearchResult> results_ivfpq_v1_full_scan = benchmark_search(
               ivfpq_v1_search_lambda_full_scan,
               test_query, test_gt, base_number, vecdim, num_queries_to_test, test_gt_d, k, false);
            
            std::string ivfpq_v1_method_name_fs = std::string("IVFADC (M1, Pthread, nprobe=0, 完整 PQ 扫描") + 
                                                  ", rerank_k=" + std::to_string(ivfpq_v1_rerank_k) + ")";
            print_results(ivfpq_v1_method_name_fs, results_ivfpq_v1_full_scan, num_queries_to_test);
        }

    } else {
        std::cerr << "跳过 IVFADC (方法 1, Pthread) 搜索测试，因为索引创建失败。" << std::endl;
    }


    // --- IVF + PQ 搜索 (OpenMP) ---
    std::cout << "\n--- IVFADC (IVF+PQ, OpenMP) 测试 ---" << std::endl;
    size_t num_ivf_clusters_ivfpq_omp = num_ivf_clusters_ivfpq; // 使用与 Pthread 版本相同的配置进行比较
    size_t pq_nsub_ivfpq_omp = pq_nsub_ivfpq;
    double pq_train_ratio_ivfpq_omp = pq_train_ratio_ivfpq;
    int ivf_kmeans_iter_ivfpq_omp = ivf_kmeans_iter_ivfpq;
    int num_omp_threads_for_ivfpq = num_pthreads_for_ann; // 使用相同的线程数变量

    IVFPQIndexOpenMP* ivfpq_omp_index_ptr = nullptr;
    if (base_number > 0 && vecdim > 0 && num_ivf_clusters_ivfpq_omp > 0 && pq_nsub_ivfpq_omp > 0 && vecdim % pq_nsub_ivfpq_omp == 0) {
        std::cout << "构建 IVFADC (OpenMP) 索引... IVF_clusters=" << num_ivf_clusters_ivfpq_omp
                  << ", PQ_nsub=" << pq_nsub_ivfpq_omp
                  << ", OpenMP_threads=" << num_omp_threads_for_ivfpq
                  << ", ivf_kmeans_iter=" << ivf_kmeans_iter_ivfpq_omp << std::endl;
        struct timeval build_start_ivfpq_omp, build_end_ivfpq_omp;
        gettimeofday(&build_start_ivfpq_omp, NULL);
        try {
            ivfpq_omp_index_ptr = new IVFPQIndexOpenMP(vecdim,
                                                       num_ivf_clusters_ivfpq_omp,
                                                       pq_nsub_ivfpq_omp,
                                                       num_omp_threads_for_ivfpq,
                                                       ivf_kmeans_iter_ivfpq_omp);
            ivfpq_omp_index_ptr->build(base, base_number, pq_train_ratio_ivfpq_omp);

        } catch (const std::exception& e) {
            std::cerr << "创建或构建 IVFPQ (OpenMP) 索引时出错: " << e.what() << std::endl;
            delete ivfpq_omp_index_ptr; // 确保清理
            ivfpq_omp_index_ptr = nullptr;
        }
        gettimeofday(&build_end_ivfpq_omp, NULL);
        if (ivfpq_omp_index_ptr) { 
            long long build_time_us_ivfpq_omp = (build_end_ivfpq_omp.tv_sec - build_start_ivfpq_omp.tv_sec) * 1000000LL +
                                           (build_end_ivfpq_omp.tv_usec - build_start_ivfpq_omp.tv_usec);
            std::cout << "IVFADC (OpenMP) 索引构建时间: " << build_time_us_ivfpq_omp / 1000.0 << " ms" << std::endl;
        }
    } else {
         std::cerr << "无法构建 IVFADC (OpenMP) 索引，参数无效 (base_number="<<base_number
                   <<", vecdim="<<vecdim<<", ivf_clusters="<<num_ivf_clusters_ivfpq_omp
                   <<", pq_nsub="<<pq_nsub_ivfpq_omp <<")." << std::endl;
    }

    if (ivfpq_omp_index_ptr) {
        std::vector<size_t> nprobe_values_ivfpq_omp = {1, 2, 4, 8, 16}; // 直接初始化
        // 应用与 Pthread 版本相似的逻辑来调整 nprobe 值
        if (num_ivf_clusters_ivfpq_omp < 16 && num_ivf_clusters_ivfpq_omp > 0) {
            nprobe_values_ivfpq_omp.clear();
            for(size_t np_val = 1; np_val <= num_ivf_clusters_ivfpq_omp; np_val *=2) nprobe_values_ivfpq_omp.push_back(np_val);
             if (nprobe_values_ivfpq_omp.empty() || nprobe_values_ivfpq_omp.back() < num_ivf_clusters_ivfpq_omp) {
                 bool contains_max = false;
                 for(size_t val : nprobe_values_ivfpq_omp) if(val == num_ivf_clusters_ivfpq_omp) contains_max = true;
                 if(!contains_max && num_ivf_clusters_ivfpq_omp > 0) nprobe_values_ivfpq_omp.push_back(num_ivf_clusters_ivfpq_omp);
             }
            if (nprobe_values_ivfpq_omp.empty() && num_ivf_clusters_ivfpq_omp > 0) nprobe_values_ivfpq_omp.push_back(1);
        } else if (num_ivf_clusters_ivfpq_omp == 0) {
            nprobe_values_ivfpq_omp.clear();
        }

        size_t ivfpq_omp_rerank_k_global = pq_rerank_k_global; // 使用全局定义的 pq_rerank_k_global

        for (size_t current_nprobe : nprobe_values_ivfpq_omp) {
            if (current_nprobe == 0) continue;
            size_t actual_nprobe = std::min(current_nprobe, num_ivf_clusters_ivfpq_omp);
            if (actual_nprobe == 0 && num_ivf_clusters_ivfpq_omp > 0) actual_nprobe = 1;
            else if (num_ivf_clusters_ivfpq_omp == 0) continue;

            std::cout << "测试 IVFADC (OpenMP) 使用 nprobe = " << actual_nprobe << ", rerank_k = " << ivfpq_omp_rerank_k_global << std::endl;
            auto ivfpq_omp_search_lambda = [&](const float* q, size_t k_param) {
                return ivfpq_omp_index_ptr->search(q, base, k_param, actual_nprobe, ivfpq_omp_rerank_k_global);
            };
            // 对于 OpenMP 版本的 IVFADC，其内部已处理并行，benchmark_search 的 use_omp_parallel 应为 false
            std::vector<SearchResult> results_ivfpq_omp = benchmark_search(
               ivfpq_omp_search_lambda,
               test_query, test_gt, base_number, vecdim, num_queries_to_test, test_gt_d, k, false); 

            std::string ivfpq_omp_method_name = "IVFADC (OpenMP, nprobe=" + std::to_string(actual_nprobe) +
                                            ", IVFclus=" + std::to_string(num_ivf_clusters_ivfpq_omp) +
                                            ", PQnsub=" + std::to_string(pq_nsub_ivfpq_omp) +
                                            ", rerank_k=" + std::to_string(ivfpq_omp_rerank_k_global) + ")";
            print_results(ivfpq_omp_method_name, results_ivfpq_omp, num_queries_to_test);
        }
    } else {
        std::cerr << "跳过 IVFADC (OpenMP) 搜索测试，因为索引创建失败。" << std::endl;
    }


    // --- 打印结果 ---
    // ... (打印 Flat, SIMD, PQ, SQ, 仅 IVF 的结果) ...
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
    // 仅 IVF 的结果已打印
    // IVFPQ 的结果也已打印


    // --- 清理 ---
    delete[] test_query;
    delete[] test_gt;
    delete[] base;
    delete pq_index_ptr; 
    delete sq_quantizer_ptr;
    delete ivf_index_ptr;
    delete ivf_omp_index_ptr; // 清理新的 OpenMP IVF 索引
    delete ivfpq_index_ptr; // 清理新索引
    delete ivfpq_v1_index_ptr; // 清理新索引
    delete ivfpq_omp_index_ptr; // <<< 新增: 清理 OpenMP 版本的 IVF+PQ 索引
    return 0;
}