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
#include "flat_scan.h" 
#include "simd_anns.h"
#include "pq_anns.h" 
#include "sq_anns.h"
#include "ivf_anns.h" 
#include "ivf_openmp.h" 

#include <functional>
#include <algorithm> 
#include <queue>     
#include <stdexcept> 

#include "ivf_pq_anns.h" 
#include "ivf_pq_v1_anns.h" 
#include "ivf_pq_openmp_anns.h" 
#include <mpi.h> 
#include "ivf_mpi_anns.h" 
#include "ivf_pq_mpi_anns.h"

// =================================================================
// <<< 新增: 包含 HNSWLIB 头文件 >>>
#include "hnswlib/hnswlib/hnswlib.h"
// =================================================================


// --- 函数声明  ---
std::priority_queue<std::pair<float, uint32_t>> flat_search(float* base, float* query, size_t base_number, size_t vecdim, size_t k);
std::priority_queue<std::pair<float, uint32_t>> simd_search(float* base, const float* query, size_t base_number, size_t vecdim, size_t k);
// ---
template<typename T>
T *LoadData(std::string data_path, size_t& n_out, size_t& d_out) 
{
    std::ifstream fin;
    fin.open(data_path, std::ios::in | std::ios::binary);
    if (!fin.is_open()) {
        std::cerr << "错误: 无法打开文件 " << data_path << std::endl;
        // 在 MPI 环境中，一个进程失败可能需要通知其他进程
        // MPI_Abort(MPI_COMM_WORLD, 1); // 或者抛出异常，让调用者处理
        throw std::runtime_error("无法打开文件: " + data_path);
    }
    uint32_t n_file, d_file; 

    fin.read(reinterpret_cast<char*>(&n_file), sizeof(uint32_t));
    if (fin.gcount() != sizeof(uint32_t)) {
        std::cerr << "错误: 从 " << data_path << " 读取 n_file 失败。" << std::endl;
        fin.close();
        throw std::runtime_error("从 " + data_path + " 读取 n_file 失败。");
    }
    fin.read(reinterpret_cast<char*>(&d_file), sizeof(uint32_t));
    if (fin.gcount() != sizeof(uint32_t)) {
        std::cerr << "错误: 从 " << data_path << " 读取 d_file 失败。" << std::endl;
        fin.close();
        throw std::runtime_error("从 " + data_path + " 读取 d_file 失败。");
    }

    n_out = static_cast<size_t>(n_file);
    d_out = static_cast<size_t>(d_file);

    if (n_out == 0 || d_out == 0) {
        std::cerr << "警告: 从 " << data_path << " 加载的数据维度或数量为零 (n=" << n_out << ", d=" << d_out << ")" << std::endl;
        // 即使为空，也返回 nullptr 或空数据，让调用者处理
        fin.close();
        return nullptr; 
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
        throw; 
    }
    
    size_t vector_byte_size = d_out * sizeof(T);
    for(size_t i = 0; i < n_out; ++i){
        if (vector_byte_size > 0) { 
            fin.read(reinterpret_cast<char*>(data) + i * vector_byte_size, vector_byte_size);
            if (fin.gcount() != static_cast<std::streamsize>(vector_byte_size)) {
                 std::cerr << "错误: 从 " << data_path << " 读取向量 " << i << " 时数据不足。"
                           << "预期 " << vector_byte_size << " 字节，得到 " << fin.gcount() << " 字节。" << std::endl;
                delete[] data;
                fin.close();
                throw std::runtime_error("从 " + data_path + " 读取向量时数据不足。");
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
    float recall; 
    int64_t latency; 
};

template<typename SearchFunc>
std::vector<SearchResult> benchmark_search(
    SearchFunc search_func, 
    float* test_query,      
    int* test_gt,           
    size_t base_number,     
    size_t vecdim,          
    size_t test_number,     
    size_t test_gt_d,       
    size_t k,               
    bool use_omp_parallel = true 
) {
    std::vector<SearchResult> results(test_number);
    std::vector<std::set<uint32_t>> gt_sets(test_number);

    for(size_t i = 0; i < test_number; ++i) {
        if (test_gt == nullptr || test_gt_d == 0) continue;
        for(size_t j = 0; j < k && j < test_gt_d; ++j){ 
             int t = test_gt[j + i*test_gt_d];
             if (t >= 0) {
                gt_sets[i].insert(static_cast<uint32_t>(t));
             }
        }
    }

    #pragma omp parallel for schedule(dynamic) if(use_omp_parallel)
    for(int i = 0; i < static_cast<int>(test_number); ++i) {
        const unsigned long Converter = 1000 * 1000;
        struct timeval val;
        gettimeofday(&val, NULL);

        float* current_query = test_query + static_cast<size_t>(i) * vecdim;
        auto res_heap = search_func(current_query, k);

        struct timeval newVal;
        gettimeofday(&newVal, NULL);
        int64_t diff = (newVal.tv_sec * Converter + newVal.tv_usec) - (val.tv_sec * Converter + val.tv_usec);

        size_t acc = 0;
        float recall = 0.0f; 

        if (i < static_cast<int>(gt_sets.size())) {
            const auto& gtset = gt_sets[i];
            if (!gtset.empty() && k > 0) {
                size_t found_count = 0;
                // 注意：hnswlib 返回的队列是小顶堆（距离/负内积），而我们通常用大顶堆（ID）
                // 为了统一处理，我们将结果转换为标准的大顶堆
                std::priority_queue<std::pair<float, uint32_t>> temp_heap;
                while(!res_heap.empty()){
                    // HNSW 返回 <distance, label>
                    // flat_search 返回 <distance, label>
                    // 统一为 <metric, id> 格式
                    temp_heap.push(res_heap.top());
                    res_heap.pop();
                }

                while(!temp_heap.empty()){
                    if(gtset.count(temp_heap.top().second)) {
                        found_count++;
                    }
                    temp_heap.pop();
                }
                // 召回率定义为找到的真实近邻数 / min(k, GT中的近邻数)
                recall = static_cast<float>(found_count) / std::min(k, gtset.size());
            } else if (k==0 || gtset.empty()) { // 如果 k=0 或 GT 为空，则召回率为 1 (或未定义，这里设为1)
                recall = 1.0f;
            }
        }
        
        #pragma omp critical
        {
             if (static_cast<size_t>(i) < results.size()) {
                results[i] = {recall, diff};
             }
        }
    }
    return results;
}

void print_results(const std::string& method_name, const std::vector<SearchResult>& results, size_t test_number_limit) {
    
    double total_recall = 0, total_latency = 0; 
    size_t valid_results = 0;

    for(size_t i = 0; i < results.size() && i < test_number_limit; ++i) {
        if (results[i].latency >= 0) { // 假设 -1 是无效延迟的标记
            total_recall += results[i].recall;
            total_latency += results[i].latency;
            valid_results++;
        }
    }
    
    std::cout << "=== " << method_name << " ===" << std::endl;
    std::cout << std::fixed << std::setprecision(5);
    std::cout << "平均召回率: " << (valid_results > 0 ? total_recall / valid_results : 0.0) << std::endl;
    std::cout << std::fixed << std::setprecision(3); 
    std::cout << "平均延迟 (us): " << (valid_results > 0 ? total_latency / valid_results : 0.0) << std::endl;
    std::cout << "测试查询数: " << valid_results << std::endl;
    std::cout << std::endl;
}


int main(int argc, char *argv[])
{
    MPI_Init(&argc, &argv); 
    int rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    size_t test_number = 0, base_number = 0;
    size_t test_gt_d = 0, vecdim = 0;
    
    std::string data_root_path = "/anndata/"; 
    if (rank == 0) { 
        if (argc > 1) { 
            data_root_path = std::string(argv[1]);
            if (data_root_path.back() != '/') {
                data_root_path += "/";
            }
        }
        std::cout << "使用数据根路径: " << data_root_path << std::endl;
    }
    int path_len;
    if (rank == 0) {
        path_len = data_root_path.length();
    }
    MPI_Bcast(&path_len, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (rank != 0) {
        data_root_path.resize(path_len);
    }
    MPI_Bcast(&data_root_path[0], path_len, MPI_CHAR, 0, MPI_COMM_WORLD);


    std::string query_path =   data_root_path + "DEEP100K.query.fbin";
    std::string gt_path =      data_root_path + "DEEP100K.gt.query.100k.top100.bin";
    std::string base_path =    data_root_path + "DEEP100K.base.100k.fbin";

    float* test_query = nullptr;
    int* test_gt = nullptr;
    float* base = nullptr;
    size_t gt_n_from_file = 0; 
    size_t base_vecdim_check = 0; 

    if (rank == 0) {
        try {
            test_query = LoadData<float>(query_path, test_number, vecdim); 
            test_gt = LoadData<int>(gt_path, gt_n_from_file, test_gt_d); 
            if (test_number == 0 && gt_n_from_file > 0 && test_query == nullptr) test_number = gt_n_from_file; // 如果查询为空但 GT 存在
            base = LoadData<float>(base_path, base_number, base_vecdim_check);
        } catch (const std::exception& e) {
            std::cerr << "Rank 0 加载数据时发生错误: " << e.what() << std::endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        if (vecdim !=0 && base_vecdim_check != 0 && vecdim != base_vecdim_check) {
            std::cout << "严重错误: 查询维度 (" << vecdim 
                      << ") 和基准维度 (" << base_vecdim_check
                      << ") 不匹配。正在退出。" << std::endl;
            MPI_Abort(MPI_COMM_WORLD, 1); 
        } else if (vecdim == 0 && base_vecdim_check != 0) {
            vecdim = base_vecdim_check; 
        } else if (vecdim == 0 && base_vecdim_check == 0 && test_query != nullptr) {
            // vecdim 应该从 test_query 中获取，如果 base 为空
        }


        if (base == nullptr && base_number > 0) { // LoadData 可能返回 nullptr
             std::cout << "严重错误: 基准数据指针为空但 base_number > 0。正在退出。" << std::endl;
             MPI_Abort(MPI_COMM_WORLD, 1);
        }
         if (test_query == nullptr && test_number > 0) {
             std::cout << "严重错误: 查询数据指针为空但 test_number > 0。正在退出。" << std::endl;
             MPI_Abort(MPI_COMM_WORLD, 1);
        }


        if (base_number == 0 || vecdim == 0) { // 如果基准为空，则 vecdim 可能来自查询
            if (base_number == 0 && rank == 0) std::cout << "警告: 基准数据集为空。" << std::endl;
            if (vecdim == 0 && rank == 0) std::cout << "警告: 向量维度为零。" << std::endl;
            if (base_number == 0 && vecdim == 0 && test_number == 0) { // 完全没有数据
                 std::cout << "严重错误: 所有数据集均为空或维度为零。正在退出。" << std::endl;
                 MPI_Abort(MPI_COMM_WORLD, 1);
            }
        }
    }

    MPI_Bcast(&test_number, 1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);
    MPI_Bcast(&vecdim, 1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);
    MPI_Bcast(&gt_n_from_file, 1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);
    MPI_Bcast(&test_gt_d, 1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);
    MPI_Bcast(&base_number, 1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);
    // base_vecdim_check 不需要广播，因为 vecdim 已经是最终确定的维度

    if (rank != 0) { 
        if (test_number > 0 && vecdim > 0) test_query = new float[test_number * vecdim]; else test_query = nullptr;
        if (gt_n_from_file > 0 && test_gt_d > 0) test_gt = new int[gt_n_from_file * test_gt_d]; else test_gt = nullptr;
        if (base_number > 0 && vecdim > 0) base = new float[base_number * vecdim]; else base = nullptr;
    }

    if (test_number > 0 && vecdim > 0 && test_query != nullptr) MPI_Bcast(test_query, test_number * vecdim, MPI_FLOAT, 0, MPI_COMM_WORLD);
    if (gt_n_from_file > 0 && test_gt_d > 0 && test_gt != nullptr) MPI_Bcast(test_gt, gt_n_from_file * test_gt_d, MPI_INT, 0, MPI_COMM_WORLD);
    if (base_number > 0 && vecdim > 0 && base != nullptr) MPI_Bcast(base, base_number * vecdim, MPI_FLOAT, 0, MPI_COMM_WORLD);


    size_t num_queries_to_test = 2000;
    if (test_number == 0) { 
        num_queries_to_test = 0;
        if (rank == 0) std::cout << "警告: 未加载查询。将 num_queries_to_test 设置为 0。" << std::endl;
    } else {
        if (num_queries_to_test > test_number) {
            num_queries_to_test = test_number;
        }
        if (test_gt != nullptr && num_queries_to_test > gt_n_from_file && gt_n_from_file > 0) {
            if (rank == 0) std::cout << "警告: num_queries_to_test (" << num_queries_to_test 
                      << ") 超过 GT 条目数 (" << gt_n_from_file 
                      << ")。将其限制为 GT 条目数。" << std::endl;
            num_queries_to_test = gt_n_from_file;
        }
    }
    if (rank == 0) std::cout << "将测试前 " << num_queries_to_test << " 条查询。" << std::endl;


    const size_t k = 10;
    const int num_pthreads_for_ann = 8; 

    int ppn_script = 8; 
    int num_nodes_script = 2; 
    int num_omp_threads_per_mpi_process = 1; 
    if (world_size > 0 && num_nodes_script > 0) {
        int mpi_procs_per_node = world_size / num_nodes_script;
        if (mpi_procs_per_node == 0 && world_size > 0) mpi_procs_per_node = 1; // 如果节点多于进程

        if (mpi_procs_per_node > 0) {
            num_omp_threads_per_mpi_process = ppn_script / mpi_procs_per_node;
        } else { 
            num_omp_threads_per_mpi_process = ppn_script;
        }
    }
    if (num_omp_threads_per_mpi_process <= 0) num_omp_threads_per_mpi_process = 1; 
    if (rank == 0) {
        std::cout << "MPI world_size: " << world_size << ", Nodes in script: " << num_nodes_script << ", PPN in script: " << ppn_script << std::endl;
        std::cout << "计算得到的每个 MPI 进程的 OpenMP 线程数: " << num_omp_threads_per_mpi_process << std::endl;
    }
    
    // --- 仅由 Rank 0 执行的基准测试 ---
    if (rank == 0) {
        // --- Flat 搜索 ---
        if (base != nullptr && test_query != nullptr) {
            auto flat_search_lambda = [&](const float* q, size_t k_param) {
                // flat_search 期望非常量指针，进行类型转换
                return flat_search(const_cast<float*>(base), const_cast<float*>(q), base_number, vecdim, k_param);
            };
            std::vector<SearchResult> results_flat = benchmark_search(
            flat_search_lambda, 
            test_query, test_gt, base_number, vecdim, num_queries_to_test, test_gt_d, k, true);
            print_results("Flat Search (暴力搜索)", results_flat, num_queries_to_test);
        } else {
            std::cout << "跳过 Flat 搜索，因为基准或查询数据为空。" << std::endl;
        }

        // =================================================================
        // <<< 新增: HNSW 性能测试 >>>
        std::cout << "\n--- HNSW (hnswlib) 测试 ---" << std::endl;
        const int M = 16;
        const int efConstruction = 200;
        hnswlib::HierarchicalNSW<float>* hnsw_index_ptr = nullptr;

        if (base != nullptr && test_query != nullptr && base_number > 0 && vecdim > 0) {
            std::cout << "构建 HNSW 索引... M=" << M << ", efConstruction=" << efConstruction << std::endl;
            struct timeval build_start, build_end;
            gettimeofday(&build_start, NULL);

            // 使用内积空间，与示例保持一致
            hnswlib::InnerProductSpace space(vecdim);
            hnsw_index_ptr = new hnswlib::HierarchicalNSW<float>(&space, base_number, M, efConstruction);

            // 使用 OpenMP 并行添加数据点
            #pragma omp parallel for
            for(int i = 0; i < base_number; ++i) {
                hnsw_index_ptr->addPoint(base + i * vecdim, i);
            }

            gettimeofday(&build_end, NULL);
            long long build_time_us = (build_end.tv_sec - build_start.tv_sec) * 1000000LL + (build_end.tv_usec - build_start.tv_usec);
            std::cout << "HNSW 索引构建时间: " << build_time_us / 1000.0 << " ms" << std::endl;
        } else {
            std::cerr << "无法构建 HNSW 索引，参数无效或数据为空。" << std::endl;
        }

        if (hnsw_index_ptr) {
            // 测试不同的 efSearch 参数
            std::vector<size_t> ef_values = {k, 20, 40, 80, 160, 320};
            for (size_t current_ef : ef_values) {
                if (current_ef < k) continue; // ef 必须大于等于 k

                std::cout << "测试 HNSW 使用 efSearch = " << current_ef << std::endl;
                hnsw_index_ptr->setEf(current_ef);

                auto hnsw_search_lambda = [&](const float* q, size_t k_param) {
                    return hnsw_index_ptr->searchKnn(q, k_param);
                };

                // 使用 benchmark_search 的 OpenMP 并行化来加速多个查询
                std::vector<SearchResult> results_hnsw = benchmark_search(
                    hnsw_search_lambda,
                    test_query, test_gt, base_number, vecdim, num_queries_to_test, test_gt_d, k, true);
                
                std::string hnsw_method_name = "HNSW (M=" + std::to_string(M) + 
                                               ", efC=" + std::to_string(efConstruction) + 
                                               ", efS=" + std::to_string(current_ef) + ")";
                print_results(hnsw_method_name, results_hnsw, num_queries_to_test);
            }
        } else {
            std::cerr << "跳过 HNSW 搜索测试，因为索引创建失败。" << std::endl;
        }
        // <<< HNSW 测试结束 >>>
        // =================================================================


        // --- IVF 搜索 (Pthread) ---
        std::cout << "\n--- IVF (Pthread) 测试 ---" << std::endl;
        size_t num_ivf_clusters_ivf_only = 0;
        if (base_number > 0) {
            num_ivf_clusters_ivf_only = std::min((size_t)256, base_number / 39); // 确保每个簇至少有约 39 个点用于 k-means
            if (num_ivf_clusters_ivf_only == 0 && base_number > 0) num_ivf_clusters_ivf_only = std::min((size_t)1, base_number);
        }
        int ivf_kmeans_iterations_ivf_only = 20; 

        IVFIndex* ivf_index_ptr = nullptr;
        if (base != nullptr && test_query != nullptr && base_number > 0 && num_ivf_clusters_ivf_only > 0 && vecdim > 0) {
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
            if (ivf_index_ptr) {
                long long build_time_us = (build_end.tv_sec - build_start.tv_sec) * 1000000LL + (build_end.tv_usec - build_start.tv_usec);
                std::cout << "IVF 索引构建时间: " << build_time_us / 1000.0 << " ms" << std::endl;
            }
        } else {
            std::cerr << "无法构建 IVF 索引 (仅 IVF)，参数无效或数据为空。" << std::endl;
        }
        
        if (ivf_index_ptr) {
            std::vector<size_t> nprobe_values = {1, 2, 4, 8, 16, 32}; 
            if (num_ivf_clusters_ivf_only < 32 && num_ivf_clusters_ivf_only > 0) { 
                nprobe_values.clear();
                for(size_t np_val = 1; np_val <= num_ivf_clusters_ivf_only; np_val *=2) nprobe_values.push_back(np_val);
                if (nprobe_values.empty() || (nprobe_values.back() < num_ivf_clusters_ivf_only && std::find(nprobe_values.begin(), nprobe_values.end(), num_ivf_clusters_ivf_only) == nprobe_values.end() ) ) {
                     if (num_ivf_clusters_ivf_only > 0 && (nprobe_values.empty() || nprobe_values.back() < num_ivf_clusters_ivf_only)) nprobe_values.push_back(num_ivf_clusters_ivf_only);
                }
                if (nprobe_values.empty()) nprobe_values.push_back(1);
            } else if (num_ivf_clusters_ivf_only == 0) { nprobe_values.clear(); }

            for (size_t current_nprobe : nprobe_values) {
                if (current_nprobe == 0 && num_ivf_clusters_ivf_only > 0) continue;
                size_t actual_nprobe = (num_ivf_clusters_ivf_only == 0) ? 0 : std::min(current_nprobe, num_ivf_clusters_ivf_only);
                if (actual_nprobe == 0 && num_ivf_clusters_ivf_only > 0) actual_nprobe = 1; 
                else if (num_ivf_clusters_ivf_only == 0) actual_nprobe = 0;

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
        size_t num_ivf_clusters_omp = num_ivf_clusters_ivf_only; 
        int ivf_kmeans_iterations_omp = ivf_kmeans_iterations_ivf_only;

        IVFIndexOpenMP* ivf_omp_index_ptr = nullptr;
        if (base != nullptr && test_query != nullptr && base_number > 0 && num_ivf_clusters_omp > 0 && vecdim > 0) {
            std::cout << "构建 IVF (OpenMP) 索引... num_clusters=" << num_ivf_clusters_omp
                      << ", threads=" << num_pthreads_for_ann 
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
            std::vector<size_t> nprobe_values_omp = {1, 2, 4, 8, 16, 32}; 
            if (num_ivf_clusters_omp < 32 && num_ivf_clusters_omp > 0) { 
                nprobe_values_omp.clear();
                for(size_t np_val = 1; np_val <= num_ivf_clusters_omp; np_val *=2) nprobe_values_omp.push_back(np_val);
                 if (nprobe_values_omp.empty() || (nprobe_values_omp.back() < num_ivf_clusters_omp && std::find(nprobe_values_omp.begin(), nprobe_values_omp.end(), num_ivf_clusters_omp) == nprobe_values_omp.end() ) ) {
                     if (num_ivf_clusters_omp > 0 && (nprobe_values_omp.empty() || nprobe_values_omp.back() < num_ivf_clusters_omp)) nprobe_values_omp.push_back(num_ivf_clusters_omp);
                }
                if (nprobe_values_omp.empty()) nprobe_values_omp.push_back(1);
            } else if (num_ivf_clusters_omp == 0) { nprobe_values_omp.clear(); }

            for (size_t current_nprobe : nprobe_values_omp) {
                if (current_nprobe == 0 && num_ivf_clusters_omp > 0) continue;
                size_t actual_nprobe = (num_ivf_clusters_omp == 0) ? 0 : std::min(current_nprobe, num_ivf_clusters_omp);
                if (actual_nprobe == 0 && num_ivf_clusters_omp > 0) actual_nprobe = 1;
                else if (num_ivf_clusters_omp == 0) actual_nprobe = 0;

                std::cout << "测试 IVF (OpenMP) 使用 nprobe = " << actual_nprobe << std::endl;
                auto ivf_omp_search_lambda = [&](const float* q, size_t k_param) {
                    return ivf_omp_index_ptr->search(q, k_param, actual_nprobe);
                };
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
        size_t num_ivf_clusters_ivfpq = 64; 
        if (base_number > 0 && num_ivf_clusters_ivfpq > base_number / 39) { 
            num_ivf_clusters_ivfpq = std::max((size_t)16, base_number / 100);
             if (num_ivf_clusters_ivfpq == 0 && base_number > 0) num_ivf_clusters_ivfpq = std::min((size_t)1, base_number);
        }
        if (num_ivf_clusters_ivfpq == 0 && base_number > 0) num_ivf_clusters_ivfpq = std::min((size_t)1, base_number);

        size_t pq_nsub_global = 4; 
        if (vecdim > 0 && vecdim % 4 != 0 && vecdim % 8 == 0) pq_nsub_global = 8; 
        else if (vecdim > 0 && vecdim % 4 != 0 && vecdim % 2 == 0) pq_nsub_global = 2;
        else if (vecdim > 0 && vecdim % 4 != 0) {
            std::cerr << "PQ: vecdim " << vecdim << " 不能被 4, 8, 或 2 整除。将使用 pq_nsub=1。" << std::endl;
            pq_nsub_global = 1;
        } else if (vecdim == 0) {
            pq_nsub_global = 0; // 无效
        }
        if (pq_nsub_global > 0 && vecdim > 0 && vecdim % pq_nsub_global != 0) pq_nsub_global = 1; // 最后的回退

        double pq_train_ratio_global = 1.0;
        size_t pq_rerank_k_global = 600;

        size_t pq_nsub_ivfpq = pq_nsub_global; 
        double pq_train_ratio_ivfpq = pq_train_ratio_global;
        int ivf_kmeans_iter_ivfpq = 20;

        IVFPQIndex* ivfpq_index_ptr = nullptr;
        if (base != nullptr && test_query != nullptr && base_number > 0 && vecdim > 0 && num_ivf_clusters_ivfpq > 0 && pq_nsub_ivfpq > 0 && vecdim % pq_nsub_ivfpq == 0) {
            std::cout << "构建 IVFADC 索引... IVF_clusters=" << num_ivf_clusters_ivfpq
                      << ", PQ_nsub=" << pq_nsub_ivfpq
                      << ", pthreads=" << num_pthreads_for_ann
                      << ", ivf_kmeans_iter=" << ivf_kmeans_iter_ivfpq << std::endl;
            struct timeval build_start_ivfpq, build_end_ivfpq;
            gettimeofday(&build_start_ivfpq, NULL);
            try {
                ivfpq_index_ptr = new IVFPQIndex(vecdim, num_ivf_clusters_ivfpq, pq_nsub_ivfpq, num_pthreads_for_ann, ivf_kmeans_iter_ivfpq);
                ivfpq_index_ptr->build(base, base_number, pq_train_ratio_ivfpq);
            } catch (const std::exception& e) {
                std::cerr << "创建或构建 IVFPQ 索引时出错: " << e.what() << std::endl;
                delete ivfpq_index_ptr; 
                ivfpq_index_ptr = nullptr;            
            }
            gettimeofday(&build_end_ivfpq, NULL);
            if (ivfpq_index_ptr) { 
                long long build_time_us_ivfpq = (build_end_ivfpq.tv_sec - build_start_ivfpq.tv_sec) * 1000000LL +
                                               (build_end_ivfpq.tv_usec - build_start_ivfpq.tv_usec);
                std::cout << "IVFADC 索引构建时间: " << build_time_us_ivfpq / 1000.0 << " ms" << std::endl;
            }
        } else {
             std::cerr << "无法构建 IVFADC 索引，参数无效 (base_number="<<base_number
                       <<", vecdim="<<vecdim<<", ivf_clusters="<<num_ivf_clusters_ivfpq
                       <<", pq_nsub="<<pq_nsub_ivfpq <<", div=" << (pq_nsub_ivfpq > 0 ? vecdim % pq_nsub_ivfpq : -1) <<")." << std::endl;
        }

        if (ivfpq_index_ptr) {
            std::vector<size_t> nprobe_values = {1, 2, 4, 8, 16}; 
            if (num_ivf_clusters_ivfpq < 32 && num_ivf_clusters_ivfpq > 0) { 
                nprobe_values.clear();
                for(size_t np_val = 1; np_val <= num_ivf_clusters_ivfpq; np_val *=2) nprobe_values.push_back(np_val);
                if (nprobe_values.empty() || (nprobe_values.back() < num_ivf_clusters_ivfpq && std::find(nprobe_values.begin(), nprobe_values.end(), num_ivf_clusters_ivfpq) == nprobe_values.end() ) ) {
                     if (num_ivf_clusters_ivfpq > 0 && (nprobe_values.empty() || nprobe_values.back() < num_ivf_clusters_ivfpq)) nprobe_values.push_back(num_ivf_clusters_ivfpq);
                }
                if (nprobe_values.empty()) nprobe_values.push_back(1);
            } else if (num_ivf_clusters_ivfpq == 0) { nprobe_values.clear(); }

            for (size_t current_nprobe : nprobe_values) {
                if (current_nprobe == 0 && num_ivf_clusters_ivfpq > 0) continue;
                size_t actual_nprobe = (num_ivf_clusters_ivfpq == 0) ? 0 : std::min(current_nprobe, num_ivf_clusters_ivfpq);
                if (actual_nprobe == 0 && num_ivf_clusters_ivfpq > 0) actual_nprobe = 1; 
                else if (num_ivf_clusters_ivfpq == 0) actual_nprobe = 0;

                std::cout << "测试 IVFADC (Pthread) 使用 nprobe = " << actual_nprobe << std::endl;
                auto ivfpq_search_lambda = [&](const float* q, size_t k_param) {
                    return ivfpq_index_ptr->search(q, base, k_param, actual_nprobe, pq_rerank_k_global);
                };
                std::vector<SearchResult> results_ivfpq = benchmark_search(
                   ivfpq_search_lambda,
                   test_query, test_gt, base_number, vecdim, num_queries_to_test, test_gt_d, k, false); 
                
                std::string ivfpq_method_name = "IVFADC (Pth, nprobe=" + std::to_string(actual_nprobe) +
                                                ", IVFclus=" + std::to_string(num_ivf_clusters_ivfpq) +
                                                ", PQnsub=" + std::to_string(pq_nsub_ivfpq) +
                                                ", rerank_k=" + std::to_string(pq_rerank_k_global) + ")";
                print_results(ivfpq_method_name, results_ivfpq, num_queries_to_test);
            }
        } else {
            std::cerr << "跳过 IVFADC 搜索测试，因为索引创建失败。" << std::endl;
        }


        // --- IVF + PQ 搜索 (OpenMP) ---
        std::cout << "\n--- IVFADC (IVF+PQ, OpenMP) 测试 ---" << std::endl;
        size_t num_ivf_clusters_ivfpq_omp = num_ivf_clusters_ivfpq; 
        size_t pq_nsub_ivfpq_omp = pq_nsub_ivfpq;
        double pq_train_ratio_ivfpq_omp = pq_train_ratio_ivfpq;
        int ivf_kmeans_iter_ivfpq_omp = ivf_kmeans_iter_ivfpq;
        int num_omp_threads_for_ivfpq = num_pthreads_for_ann; 

        IVFPQIndexOpenMP* ivfpq_omp_index_ptr = nullptr;
        if (base != nullptr && test_query != nullptr && base_number > 0 && vecdim > 0 && num_ivf_clusters_ivfpq_omp > 0 && pq_nsub_ivfpq_omp > 0 && vecdim % pq_nsub_ivfpq_omp == 0) {
            std::cout << "构建 IVFADC (OpenMP) 索引... IVF_clusters=" << num_ivf_clusters_ivfpq_omp
                      << ", PQ_nsub=" << pq_nsub_ivfpq_omp
                      << ", OpenMP_threads=" << num_omp_threads_for_ivfpq
                      << ", ivf_kmeans_iter=" << ivf_kmeans_iter_ivfpq_omp << std::endl;
            struct timeval build_start_ivfpq_omp, build_end_ivfpq_omp;
            gettimeofday(&build_start_ivfpq_omp, NULL);
            try {
                ivfpq_omp_index_ptr = new IVFPQIndexOpenMP(vecdim, num_ivf_clusters_ivfpq_omp, pq_nsub_ivfpq_omp, num_pthreads_for_ann, ivf_kmeans_iter_ivfpq_omp);
                ivfpq_omp_index_ptr->build(base, base_number, pq_train_ratio_ivfpq_omp);
            } catch (const std::exception& e) {
                std::cerr << "创建 IVFADC (OpenMP) 索引时出错: " << e.what() << std::endl;
                ivfpq_omp_index_ptr = nullptr;
            }
            gettimeofday(&build_end_ivfpq_omp, NULL);
            if (ivfpq_omp_index_ptr) {
                 long long build_time_us_ivfpq_omp = (build_end_ivfpq_omp.tv_sec - build_start_ivfpq_omp.tv_sec) * 1000000LL +
                                               (build_end_ivfpq_omp.tv_usec - build_start_ivfpq_omp.tv_usec);
                std::cout << "IVFADC (OpenMP) 索引构建时间: " << build_time_us_ivfpq_omp / 1000.0 << " ms" << std::endl;
            }
        } else {
            std::cerr << "无法构建 IVFADC (OpenMP) 索引，参数无效 (base_number=" << base_number
                      << ", vecdim=" << vecdim << ", num_ivf_clusters_ivfpq_omp=" << num_ivf_clusters_ivfpq_omp << ")." << std::endl;
        }

        if (ivfpq_omp_index_ptr) {
            std::vector<size_t> nprobe_values_ivfpq_omp = {1, 2, 4, 8, 16, 32}; 
            if (num_ivf_clusters_ivfpq_omp < 32 && num_ivf_clusters_ivfpq_omp > 0) { 
                nprobe_values_ivfpq_omp.clear();
                for(size_t np_val = 1; np_val <= num_ivf_clusters_ivfpq_omp; np_val *=2) nprobe_values_ivfpq_omp.push_back(np_val);
                 if (nprobe_values_ivfpq_omp.empty() || (nprobe_values_ivfpq_omp.back() < num_ivf_clusters_ivfpq_omp && std::find(nprobe_values_ivfpq_omp.begin(), nprobe_values_ivfpq_omp.end(), num_ivf_clusters_ivfpq_omp) == nprobe_values_ivfpq_omp.end() ) ) {
                     if (num_ivf_clusters_ivfpq_omp > 0 && (nprobe_values_ivfpq_omp.empty() || nprobe_values_ivfpq_omp.back() < num_ivf_clusters_ivfpq_omp)) nprobe_values_ivfpq_omp.push_back(num_ivf_clusters_ivfpq_omp);
                }
                if (nprobe_values_ivfpq_omp.empty()) nprobe_values_ivfpq_omp.push_back(1);
            } else if (num_ivf_clusters_ivfpq_omp == 0) { nprobe_values_ivfpq_omp.clear(); }

            size_t ivfpq_omp_rerank_k_global = pq_rerank_k_global;
            for (size_t current_nprobe : nprobe_values_ivfpq_omp) {
                if (current_nprobe == 0 && num_ivf_clusters_ivfpq_omp > 0) continue;
                size_t actual_nprobe = (num_ivf_clusters_ivfpq_omp == 0) ? 0 : std::min(current_nprobe, num_ivf_clusters_ivfpq_omp);
                if (actual_nprobe == 0 && num_ivf_clusters_ivfpq_omp > 0) actual_nprobe = 1;
                else if (num_ivf_clusters_ivfpq_omp == 0) actual_nprobe = 0;

                std::cout << "测试 IVFADC (OpenMP) 使用 nprobe = " << actual_nprobe << ", rerank_k = " << ivfpq_omp_rerank_k_global << std::endl;
                auto ivfpq_omp_search_lambda = [&](const float* q, size_t k_param) {
                    return ivfpq_omp_index_ptr->search(q, base, k_param, actual_nprobe, ivfpq_omp_rerank_k_global);
                };
                std::vector<SearchResult> results_ivfpq_omp = benchmark_search(
                   ivfpq_omp_search_lambda,
                   test_query, test_gt, base_number, vecdim, num_queries_to_test, test_gt_d, k, false); 
                
                std::string ivfpq_omp_method_name = "IVFADC (OMP, nprobe=" + std::to_string(actual_nprobe) +
                                                    ", IVFclus=" + std::to_string(num_ivf_clusters_ivfpq_omp) +
                                                    ", PQnsub=" + std::to_string(pq_nsub_ivfpq_omp) +
                                                    ", rerank_k=" + std::to_string(ivfpq_omp_rerank_k_global) + ")";
                print_results(ivfpq_omp_method_name, results_ivfpq_omp, num_queries_to_test);
            }
        } else {
            std::cerr << "跳过 IVFADC (OpenMP) 搜索测试，因为索引创建失败。" << std::endl;
        }

        // =================================================================
        // <<< 新增: HNSW 资源清理 >>>
        delete hnsw_index_ptr;
        // =================================================================

        delete ivf_index_ptr;
        delete ivf_omp_index_ptr; 
        delete ivfpq_index_ptr; 
        delete ivfpq_omp_index_ptr;
    }

    // --- IVF (MPI+OpenMP) 测试 ---
    if (rank == 0) std::cout << "\n--- IVF (MPI+OpenMP) 测试 ---" << std::endl;
    
    float* local_base_data_ptr_for_rank = nullptr;
    size_t num_local_base_vectors_for_rank = 0;
    std::vector<int> local_data_global_indices_for_rank;

    if (base != nullptr && base_number > 0 && vecdim > 0 && world_size > 0) {
        size_t vectors_per_rank = base_number / world_size;
        size_t remainder_vectors = base_number % world_size;
        size_t current_offset_global = 0;

        for (int r = 0; r < world_size; ++r) {
            size_t count_for_this_rank = vectors_per_rank + (r < static_cast<int>(remainder_vectors) ? 1 : 0);
            if (r == rank) {
                num_local_base_vectors_for_rank = count_for_this_rank;
                local_base_data_ptr_for_rank = base + current_offset_global * vecdim;
                local_data_global_indices_for_rank.resize(count_for_this_rank);
                std::iota(local_data_global_indices_for_rank.begin(), local_data_global_indices_for_rank.end(), current_offset_global);
            }
            current_offset_global += count_for_this_rank;
        }
    }

    IVFIndexMPI* ivf_mpi_index_ptr = nullptr;
    size_t num_ivf_clusters_mpi = 0;
    if (rank == 0 && base_number > 0) { 
        num_ivf_clusters_mpi = std::min((size_t)256, base_number / 39);
        if (num_ivf_clusters_mpi == 0 && base_number > 0) num_ivf_clusters_mpi = std::min((size_t)1, base_number);
    }
    MPI_Bcast(&num_ivf_clusters_mpi, 1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);

    int ivf_kmeans_iterations_mpi = 20; 

    if (base != nullptr && test_query != nullptr && base_number > 0 && num_ivf_clusters_mpi > 0 && vecdim > 0) {
        if (rank == 0) {
            std::cout << "构建 IVF (MPI+OpenMP) 索引... num_clusters=" << num_ivf_clusters_mpi
                      << ", OMP_threads_per_proc=" << num_omp_threads_per_mpi_process
                      << ", kmeans_iter=" << ivf_kmeans_iterations_mpi << std::endl;
        }
        struct timeval build_start_mpi, build_end_mpi;
        MPI_Barrier(MPI_COMM_WORLD); 
        if (rank == 0) gettimeofday(&build_start_mpi, NULL);

        try {
            ivf_mpi_index_ptr = new IVFIndexMPI(MPI_COMM_WORLD,
                                                base,                                               
                                                local_base_data_ptr_for_rank,
                                                num_local_base_vectors_for_rank,
                                                local_data_global_indices_for_rank,
                                                base_number,
                                                vecdim,
                                                num_ivf_clusters_mpi,
                                                num_omp_threads_per_mpi_process,
                                                ivf_kmeans_iterations_mpi);
        } catch (const std::exception& e) {
            if (rank == 0) std::cerr << "创建 IVF MPI 索引时出错: " << e.what() << std::endl;
            delete ivf_mpi_index_ptr;
            ivf_mpi_index_ptr = nullptr;
            MPI_Abort(MPI_COMM_WORLD, 1);       
        }

        MPI_Barrier(MPI_COMM_WORLD); 
        if (rank == 0 && ivf_mpi_index_ptr) {
            gettimeofday(&build_end_mpi, NULL);
            long long build_time_us_mpi = (build_end_mpi.tv_sec - build_start_mpi.tv_sec) * 1000000LL + (build_end_mpi.tv_usec - build_start_mpi.tv_usec);
            std::cout << "IVF (MPI+OpenMP) 索引构建时间: " << build_time_us_mpi / 1000.0 << " ms" << std::endl;
        }
    } else {
        if (rank == 0) std::cerr << "无法构建 IVF (MPI+OpenMP) 索引，参数无效或数据为空。" << std::endl;
    }

    if (ivf_mpi_index_ptr) {
        std::vector<size_t> nprobe_values_mpi = {1, 2, 4, 8, 16, 32};
        if (num_ivf_clusters_mpi < 32 && num_ivf_clusters_mpi > 0) {
            nprobe_values_mpi.clear();
            for(size_t np_val = 1; np_val <= num_ivf_clusters_mpi; np_val *=2) nprobe_values_mpi.push_back(np_val);
            if (nprobe_values_mpi.empty() || (nprobe_values_mpi.back() < num_ivf_clusters_mpi && std::find(nprobe_values_mpi.begin(), nprobe_values_mpi.end(), num_ivf_clusters_mpi) == nprobe_values_mpi.end() ) ) {
                 if (num_ivf_clusters_mpi > 0 && (nprobe_values_mpi.empty() || nprobe_values_mpi.back() < num_ivf_clusters_mpi)) nprobe_values_mpi.push_back(num_ivf_clusters_mpi);
            }
            if (nprobe_values_mpi.empty()) nprobe_values_mpi.push_back(1);
        } else if (num_ivf_clusters_mpi == 0) { 
            nprobe_values_mpi.clear(); 
        }

        float* current_query_broadcast = nullptr;
        if (vecdim > 0) current_query_broadcast = new float[vecdim];

        for (size_t current_nprobe : nprobe_values_mpi) {
            if (current_nprobe == 0 && num_ivf_clusters_mpi > 0) continue;
            size_t actual_nprobe = (num_ivf_clusters_mpi == 0) ? 0 : std::min(current_nprobe, num_ivf_clusters_mpi);
            if (actual_nprobe == 0 && num_ivf_clusters_mpi > 0) actual_nprobe = 1;
            else if (num_ivf_clusters_mpi == 0) actual_nprobe = 0;

            if (rank == 0) {
                std::cout << "测试 IVF (MPI+OpenMP) 使用 nprobe = " << actual_nprobe << std::endl;
            }
            
            std::vector<SearchResult> results_ivf_mpi_run;
            if (rank == 0 && num_queries_to_test > 0) {
                 results_ivf_mpi_run.resize(num_queries_to_test);
            }

            for (size_t query_idx = 0; query_idx < num_queries_to_test; ++query_idx) {
                if (rank == 0 && test_query != nullptr && query_idx < test_number) {
                     memcpy(current_query_broadcast, test_query + query_idx * vecdim, vecdim * sizeof(float));
                }
                if (vecdim > 0 && current_query_broadcast != nullptr) MPI_Bcast(current_query_broadcast, vecdim, MPI_FLOAT, 0, MPI_COMM_WORLD);

                struct timeval val_mpi_s, newVal_mpi_s;
                MPI_Barrier(MPI_COMM_WORLD);
                if (rank == 0) gettimeofday(&val_mpi_s, NULL);

                std::priority_queue<std::pair<float, uint32_t>> res_heap_mpi_s = 
                    ivf_mpi_index_ptr->search(current_query_broadcast, k, actual_nprobe);
                
                MPI_Barrier(MPI_COMM_WORLD);
                if (rank == 0) {
                    gettimeofday(&newVal_mpi_s, NULL);
                    long long diff_mpi_s = (newVal_mpi_s.tv_sec * 1000000LL + newVal_mpi_s.tv_usec) -
                                       (val_mpi_s.tv_sec * 1000000LL + val_mpi_s.tv_usec);
                    
                    size_t acc_mpi_s = 0;
                    std::set<uint32_t> gtset_mpi_s;
                     if (test_gt != nullptr && query_idx < gt_n_from_file) {
                        for (size_t j = 0; j < k && j < test_gt_d; ++j) {
                            int t_mpi_s = test_gt[j + query_idx * test_gt_d];
                            if (t_mpi_s >=0) gtset_mpi_s.insert(static_cast<uint32_t>(t_mpi_s));
                        }
                    }
                    std::priority_queue<std::pair<float, uint32_t>> temp_heap_s = res_heap_mpi_s;
                    while (!temp_heap_s.empty()) {
                        if (gtset_mpi_s.count(temp_heap_s.top().second)) {
                            acc_mpi_s++;
                        }
                        temp_heap_s.pop();
                    }
                    float recall_mpi_s = (gtset_mpi_s.empty() || k == 0) ? 1.0f : static_cast<float>(acc_mpi_s) / std::min(k, gtset_mpi_s.size());
                     if (query_idx < results_ivf_mpi_run.size()) {
                        results_ivf_mpi_run[query_idx] = {recall_mpi_s, diff_mpi_s};
                     }
                    
                    // 添加进度输出，帮助调试
                    if ((query_idx + 1) % 100 == 0 || query_idx == num_queries_to_test - 1) {
                        std::cout << "IVF MPI: 完成查询 " << (query_idx + 1) << "/" << num_queries_to_test << std::endl;
                    }
                }
            }

            if (rank == 0) {
                std::string ivf_mpi_method_name = "IVF (MPI, nprobe=" + std::to_string(actual_nprobe) + 
                                              ", clusters=" + std::to_string(num_ivf_clusters_mpi) + 
                                              ", OMPth=" + std::to_string(num_omp_threads_per_mpi_process) + ")";
                print_results(ivf_mpi_method_name, results_ivf_mpi_run, num_queries_to_test);
            }
        }
        if (current_query_broadcast) delete[] current_query_broadcast;
    } else {
        if (rank == 0) std::cerr << "跳过 IVF (MPI+OpenMP) 搜索测试，因为索引创建失败。" << std::endl;
    }
    delete ivf_mpi_index_ptr; 

    // --- IVF+PQ (MPI+OpenMP) 测试 ---
    if (rank == 0) std::cout << "\n--- IVFADC (MPI+OpenMP) 测试 ---" << std::endl;
    
    float* local_base_data_ptr_for_ivfpq_mpi = nullptr;
    size_t num_local_base_vectors_for_ivfpq_mpi = 0;
    std::vector<int> local_data_global_indices_for_ivfpq_mpi;

    if (base != nullptr && base_number > 0 && vecdim > 0 && world_size > 0) {
        size_t vectors_per_rank = base_number / world_size;
        size_t remainder_vectors = base_number % world_size;
        size_t current_offset_global = 0;

        for (int r = 0; r < world_size; ++r) {
            size_t count_for_this_rank = vectors_per_rank + (r < static_cast<int>(remainder_vectors) ? 1 : 0);
            if (r == rank) {
                num_local_base_vectors_for_ivfpq_mpi = count_for_this_rank;
                local_base_data_ptr_for_ivfpq_mpi = base + current_offset_global * vecdim;
                local_data_global_indices_for_ivfpq_mpi.resize(count_for_this_rank);
                std::iota(local_data_global_indices_for_ivfpq_mpi.begin(), local_data_global_indices_for_ivfpq_mpi.end(), current_offset_global);
            }
            current_offset_global += count_for_this_rank;
        }
    }

    IVFPQMPIIndex* ivfpq_mpi_index_ptr = nullptr;
    size_t num_ivf_clusters_ivfpq_mpi = 0;
    size_t pq_nsub_ivfpq_mpi = 4; 
    if (rank == 0) {         
        num_ivf_clusters_ivfpq_mpi = 64;
        if (base_number > 0 && num_ivf_clusters_ivfpq_mpi > base_number / 39) {
            num_ivf_clusters_ivfpq_mpi = std::max((size_t)16, base_number / 100);
        }
        if (num_ivf_clusters_ivfpq_mpi == 0 && base_number > 0) num_ivf_clusters_ivfpq_mpi = std::min((size_t)1, base_number);

        if (vecdim > 0 && vecdim % 4 != 0 && vecdim % 8 == 0) pq_nsub_ivfpq_mpi = 8;
        else if (vecdim > 0 && vecdim % 4 != 0 && vecdim % 2 == 0) pq_nsub_ivfpq_mpi = 2;
        else if (vecdim > 0 && vecdim % 4 != 0) {
             std::cerr << "IVFPQ MPI: vecdim " << vecdim << " 不能被 4, 8, 或 2 整除。将使用 pq_nsub=1。" << std::endl;
             pq_nsub_ivfpq_mpi = 1;
        }
        if (pq_nsub_ivfpq_mpi > 0 && vecdim > 0 && vecdim % pq_nsub_ivfpq_mpi != 0) {
             pq_nsub_ivfpq_mpi = 1; // 最后的回退
             std::cerr << "IVFPQ MPI: vecdim " << vecdim << " 不能被选定的 pq_nsub 整除。将 pq_nsub 调整为 " << pq_nsub_ivfpq_mpi << std::endl;
        }
        if (vecdim == 0 || pq_nsub_ivfpq_mpi == 0) { 
             std::cerr << "IVFPQ MPI: vecdim 或 pq_nsub 无效，无法继续。" << std::endl;
             num_ivf_clusters_ivfpq_mpi = 0; 
        }
    }
    MPI_Bcast(&num_ivf_clusters_ivfpq_mpi, 1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);
    MPI_Bcast(&pq_nsub_ivfpq_mpi, 1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);

    int ivf_kmeans_iterations_ivfpq_mpi = 20;
    double pq_train_ratio_ivfpq_mpi = 0.1; 

    bool can_build_ivfpq_mpi = base != nullptr && test_query != nullptr && base_number > 0 && vecdim > 0 && num_ivf_clusters_ivfpq_mpi > 0 && pq_nsub_ivfpq_mpi > 0 && (vecdim % pq_nsub_ivfpq_mpi == 0);

    if (can_build_ivfpq_mpi) {
        if (rank == 0) {
            std::cout << "构建 IVFADC (MPI+OpenMP) 索引... IVF_clusters=" << num_ivf_clusters_ivfpq_mpi
                      << ", PQ_nsub=" << pq_nsub_ivfpq_mpi
                      << ", OMP_threads_per_proc=" << num_omp_threads_per_mpi_process
                      << ", ivf_kmeans_iter=" << ivf_kmeans_iterations_ivfpq_mpi
                      << ", pq_train_ratio=" << pq_train_ratio_ivfpq_mpi << std::endl;
        }
        struct timeval build_start_ivfpq_mpi, build_end_ivfpq_mpi;
        MPI_Barrier(MPI_COMM_WORLD);
        if (rank == 0) gettimeofday(&build_start_ivfpq_mpi, NULL);

        try {
            ivfpq_mpi_index_ptr = new IVFPQMPIIndex(MPI_COMM_WORLD,
                                                    base, 
                                                    local_base_data_ptr_for_ivfpq_mpi,
                                                    num_local_base_vectors_for_ivfpq_mpi,
                                                    local_data_global_indices_for_ivfpq_mpi,
                                                    base_number, 
                                                    vecdim,
                                                    num_ivf_clusters_ivfpq_mpi,
                                                    pq_nsub_ivfpq_mpi,
                                                    20, 
                                                    ivf_kmeans_iterations_ivfpq_mpi,
                                                    num_omp_threads_per_mpi_process,
                                                    (rank==0)
                                                    );
            ivfpq_mpi_index_ptr->build(pq_train_ratio_ivfpq_mpi);
        } catch (const std::exception& e) {
            if (rank == 0) {
                std::cerr << "创建或构建 IVFPQ (MPI) 索引时出错: " << e.what() << std::endl;
            }
            delete ivfpq_mpi_index_ptr; 
            ivfpq_mpi_index_ptr = nullptr;
            MPI_Abort(MPI_COMM_WORLD, 1); 
        }

        MPI_Barrier(MPI_COMM_WORLD);
        if (rank == 0 && ivfpq_mpi_index_ptr) {
            gettimeofday(&build_end_ivfpq_mpi, NULL);
            long long build_time_us_ivfpq_mpi = (build_end_ivfpq_mpi.tv_sec - build_start_ivfpq_mpi.tv_sec) * 1000000LL + (build_end_ivfpq_mpi.tv_usec - build_start_ivfpq_mpi.tv_usec);
            std::cout << "IVFADC (MPI+OpenMP) 索引构建时间: " << build_time_us_ivfpq_mpi / 1000.0 << " ms" << std::endl;
        }
    } else {
        if (rank == 0) std::cerr << "无法构建 IVFADC (MPI+OpenMP) 索引，参数无效或数据为空。" << std::endl;
    }

    if (ivfpq_mpi_index_ptr) {
        std::vector<size_t> nprobe_values_ivfpq_mpi = {1, 2, 4, 8, 16};
         if (num_ivf_clusters_ivfpq_mpi < 16 && num_ivf_clusters_ivfpq_mpi > 0) {
            nprobe_values_ivfpq_mpi.clear();
            for(size_t np_val = 1; np_val <= num_ivf_clusters_ivfpq_mpi; np_val *=2) {
                 nprobe_values_ivfpq_mpi.push_back(np_val);
            }
            if (nprobe_values_ivfpq_mpi.empty() || (nprobe_values_ivfpq_mpi.back() < num_ivf_clusters_ivfpq_mpi && std::find(nprobe_values_ivfpq_mpi.begin(), nprobe_values_ivfpq_mpi.end(), num_ivf_clusters_ivfpq_mpi) == nprobe_values_ivfpq_mpi.end() ) ) {
                 if (num_ivf_clusters_ivfpq_mpi > 0 && (nprobe_values_ivfpq_mpi.empty() || nprobe_values_ivfpq_mpi.back() < num_ivf_clusters_ivfpq_mpi)) nprobe_values_ivfpq_mpi.push_back(num_ivf_clusters_ivfpq_mpi);
            }
            if (nprobe_values_ivfpq_mpi.empty()) nprobe_values_ivfpq_mpi.push_back(1); 
        } else if (num_ivf_clusters_ivfpq_mpi == 0) {
            nprobe_values_ivfpq_mpi.clear(); 
        }

        std::vector<size_t> rerank_k_adc_values = {k, 100, 600}; 

        float* current_query_broadcast_ivfpq = nullptr;
        if (vecdim > 0) current_query_broadcast_ivfpq = new float[vecdim];

        for (size_t current_nprobe : nprobe_values_ivfpq_mpi) {
            if (current_nprobe == 0 && num_ivf_clusters_ivfpq_mpi > 0) continue; 
            size_t actual_nprobe = (num_ivf_clusters_ivfpq_mpi == 0) ? 0 : std::min(current_nprobe, num_ivf_clusters_ivfpq_mpi);
             if (actual_nprobe == 0 && num_ivf_clusters_ivfpq_mpi > 0) actual_nprobe = 1; 
             else if (num_ivf_clusters_ivfpq_mpi == 0) actual_nprobe = 0;

            for (size_t current_rerank_k_adc : rerank_k_adc_values) {
                if (current_rerank_k_adc < k && k > 0) continue; 
                if (k == 0 && current_rerank_k_adc > 0) continue; // 如果 k=0, rerank_k_adc 也应为0或不适用

                if (rank == 0) {
                    std::cout << "测试 IVFADC (MPI+OpenMP) 使用 nprobe = " << actual_nprobe
                              << ", rerank_k_adc = " << current_rerank_k_adc << std::endl;
                }

                std::vector<SearchResult> results_ivfpq_mpi_run;
                if (rank == 0 && num_queries_to_test > 0) {
                    results_ivfpq_mpi_run.resize(num_queries_to_test);
                }

                // 修复: 确保查询循环使用正确的循环变量类型和边界检查
                for (size_t query_idx = 0; query_idx < num_queries_to_test; ++query_idx) {
                    if (rank == 0 && test_query != nullptr && query_idx < test_number) {
                        memcpy(current_query_broadcast_ivfpq, test_query + query_idx * vecdim, vecdim * sizeof(float));
                    }
                    if (vecdim > 0 && current_query_broadcast_ivfpq != nullptr) {
                        MPI_Bcast(current_query_broadcast_ivfpq, vecdim, MPI_FLOAT, 0, MPI_COMM_WORLD);
                    }

                    struct timeval val_mpi, newVal_mpi;
                    MPI_Barrier(MPI_COMM_WORLD); 
                    if (rank == 0) gettimeofday(&val_mpi, NULL);

                    std::priority_queue<std::pair<float, uint32_t>> res_heap_mpi =
                        ivfpq_mpi_index_ptr->search(current_query_broadcast_ivfpq, k, actual_nprobe, current_rerank_k_adc);

                    MPI_Barrier(MPI_COMM_WORLD); 
                    if (rank == 0) {
                        gettimeofday(&newVal_mpi, NULL);
                        int64_t diff_mpi = (newVal_mpi.tv_sec * 1000000LL + newVal_mpi.tv_usec) -
                                           (val_mpi.tv_sec * 1000000LL + val_mpi.tv_usec);

                        size_t acc_mpi = 0;
                        std::set<uint32_t> gtset_mpi;
                        if (test_gt != nullptr && query_idx < gt_n_from_file) {
                            for (size_t j = 0; j < k && j < test_gt_d; ++j) {
                                int t_mpi = test_gt[j + query_idx * test_gt_d];
                                if (t_mpi >=0) gtset_mpi.insert(static_cast<uint32_t>(t_mpi));
                            }
                        }
                        std::priority_queue<std::pair<float, uint32_t>> temp_heap_mpi = res_heap_mpi;
                        while (!temp_heap_mpi.empty()) {
                            if (gtset_mpi.count(temp_heap_mpi.top().second)) {
                                acc_mpi++;
                            }
                            temp_heap_mpi.pop();
                        }
                        float recall_mpi = (gtset_mpi.empty() || k == 0) ? 1.0f : static_cast<float>(acc_mpi) / std::min(k, gtset_mpi.size());
                        if (query_idx < results_ivfpq_mpi_run.size()) {
                           results_ivfpq_mpi_run[query_idx] = {recall_mpi, diff_mpi};
                        }
                        
                        // 添加进度输出，帮助调试
                        if ((query_idx + 1) % 100 == 0 || query_idx == num_queries_to_test - 1) {
                            std::cout << "IVFPQ MPI: 完成查询 " << (query_idx + 1) << "/" << num_queries_to_test << std::endl;
                        }
                    }
                } 

                if (rank == 0) {
                    std::string ivfpq_mpi_method_name = "IVFADC (MPI, nprobe=" + std::to_string(actual_nprobe) +
                                                        ", IVFclus=" + std::to_string(num_ivf_clusters_ivfpq_mpi) +
                                                        ", PQnsub=" + std::to_string(pq_nsub_ivfpq_mpi) +
                                                        ", rerank_k_adc=" + std::to_string(current_rerank_k_adc) +
                                                        ", OMPth=" + std::to_string(num_omp_threads_per_mpi_process) + ")";
                    print_results(ivfpq_mpi_method_name, results_ivfpq_mpi_run, num_queries_to_test);
                }
            } 
        } 
        if (current_query_broadcast_ivfpq) delete[] current_query_broadcast_ivfpq;
    } else {
        if (rank == 0) std::cerr << "跳过 IVFADC (MPI+OpenMP) 搜索测试，因为索引创建失败。" << std::endl;
    }
    delete ivfpq_mpi_index_ptr; 

    delete[] test_query;
    delete[] test_gt;
    delete[] base;
    
    MPI_Finalize(); 
    return 0;
}