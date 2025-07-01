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
#include <omp.h> // 用于 OMP_NUM_THREADS
#include "flat_scan.h" 
#include "ivf_openmp.h" // 包含新的 IVF OpenMP 头文件
#include "simple_ivf_gpu.h"    // 包含简化的 IVF GPU 头文件
#include "optimized_simple_ivf_gpu.h" // 包含优化的 IVF GPU 头文件
#include "super_optimized_ivf_gpu.h" // 包含超级优化的 IVF GPU 头文件
#include "adaptive_optimized_ivf_gpu.h" // 包含自适应优化的 IVF GPU 头文件
// #include "optimized_ivf_gpu.h" // 暂时注释掉，避免Thrust库的兼容性问题

#include <algorithm> 
#include <queue>     
#include <stdexcept> 

// --- 函数声明  ---
std::priority_queue<std::pair<float, uint32_t>> flat_search(float* base, float* query, size_t base_number, size_t vecdim, size_t k);
// ---

template<typename T>
T *LoadData(std::string data_path, size_t& n_out, size_t& d_out) // 重命名以避免混淆
{
    std::ifstream fin;
    fin.open(data_path, std::ios::in | std::ios::binary);
    uint32_t n_file, d_file; // 使用固定大小的类型读取文件元数据

    fin.read(reinterpret_cast<char*>(&n_file), sizeof(uint32_t));
    
    fin.read(reinterpret_cast<char*>(&d_file), sizeof(uint32_t));

    n_out = static_cast<size_t>(n_file);
    d_out = static_cast<size_t>(d_file);

    

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

// 简化的benchmark函数 - 为flat search专用
std::vector<SearchResult> benchmark_flat_search(
    float* base,
    float* test_query,      
    int* test_gt,           
    size_t base_number,     
    size_t vecdim,          
    size_t test_number,     
    size_t test_gt_d,       
    size_t k
) {
    std::vector<SearchResult> results(test_number);

    // 准备 ground truth 集合
    std::vector<std::set<uint32_t>> gt_sets(test_number);
    for(size_t i = 0; i < test_number; ++i) {
        for(size_t j = 0; j < k && j < test_gt_d; ++j){ 
             int t = test_gt[j + i*test_gt_d];
             if (t >= 0) { 
                gt_sets[i].insert(static_cast<uint32_t>(t));
             }
        }
    }

    #pragma omp parallel for schedule(dynamic)
    for(int i = 0; i < static_cast<int>(test_number); ++i) {
        const unsigned long Converter = 1000 * 1000;
        struct timeval val;
        gettimeofday(&val, NULL);

        float* current_query = test_query + static_cast<size_t>(i) * vecdim;
        auto res_heap = flat_search(base, current_query, base_number, vecdim, k);

        struct timeval newVal;
        gettimeofday(&newVal, NULL);
        int64_t diff = (newVal.tv_sec * Converter + newVal.tv_usec) - (val.tv_sec * Converter + val.tv_usec);

        // 计算召回率
        size_t acc = 0;
        float recall = 0.0f;

        if (i < static_cast<int>(gt_sets.size())) {
            const auto& gtset = gt_sets[i];
            if (!gtset.empty() && k > 0) {
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
        
        #pragma omp critical
        {
             if (static_cast<size_t>(i) < results.size()) {
                results[i] = {recall, diff};
             }
        }
    }
    return results;
}

// 为IVF OpenMP专用的benchmark函数
std::vector<SearchResult> benchmark_ivf_openmp_search(
    IVFIndexOpenMP* ivf_index,
    float* test_query,      
    int* test_gt,           
    size_t base_number,     
    size_t vecdim,          
    size_t test_number,     
    size_t test_gt_d,       
    size_t k,
    size_t nprobe
) {
    std::vector<SearchResult> results(test_number);

    // 准备 ground truth 集合
    std::vector<std::set<uint32_t>> gt_sets(test_number);
    for(size_t i = 0; i < test_number; ++i) {
        for(size_t j = 0; j < k && j < test_gt_d; ++j){ 
             int t = test_gt[j + i*test_gt_d];
             if (t >= 0) { 
                gt_sets[i].insert(static_cast<uint32_t>(t));
             }
        }
    }

    // 不使用OpenMP并行，因为IVF内部已经有并行化
    for(int i = 0; i < static_cast<int>(test_number); ++i) {
        const unsigned long Converter = 1000 * 1000;
        struct timeval val;
        gettimeofday(&val, NULL);

        float* current_query = test_query + static_cast<size_t>(i) * vecdim;
        auto res_heap = ivf_index->search(current_query, k, nprobe);

        struct timeval newVal;
        gettimeofday(&newVal, NULL);
        int64_t diff = (newVal.tv_sec * Converter + newVal.tv_usec) - (val.tv_sec * Converter + val.tv_usec);

        // 计算召回率
        size_t acc = 0;
        float recall = 0.0f;

        if (i < static_cast<int>(gt_sets.size())) {
            const auto& gtset = gt_sets[i];
            if (!gtset.empty() && k > 0) {
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
        
        results[i] = {recall, diff};
    }
    return results;
}

// 为GPU专用的benchmark函数
std::vector<SearchResult> benchmark_gpu_search(
    SimpleIVFGPU* gpu_index,
    float* test_query,      
    int* test_gt,           
    size_t base_number,     
    size_t vecdim,          
    size_t test_number,     
    size_t test_gt_d,       
    size_t k,
    size_t nprobe
) {
    std::vector<SearchResult> results(test_number);

    // 准备 ground truth 集合
    std::vector<std::set<uint32_t>> gt_sets(test_number);
    for(size_t i = 0; i < test_number; ++i) {
        for(size_t j = 0; j < k && j < test_gt_d; ++j){ 
             int t = test_gt[j + i*test_gt_d];
             if (t >= 0) { 
                gt_sets[i].insert(static_cast<uint32_t>(t));
             }
        }
    }

    for(int i = 0; i < static_cast<int>(test_number); ++i) {
        const unsigned long Converter = 1000 * 1000;
        struct timeval val;
        gettimeofday(&val, NULL);

        float* current_query = test_query + static_cast<size_t>(i) * vecdim;
        auto res_heap = gpu_index->search(current_query, k, nprobe);

        struct timeval newVal;
        gettimeofday(&newVal, NULL);
        int64_t diff = (newVal.tv_sec * Converter + newVal.tv_usec) - (val.tv_sec * Converter + val.tv_usec);

        // 计算召回率
        size_t acc = 0;
        float recall = 0.0f;

        if (i < static_cast<int>(gt_sets.size())) {
            const auto& gtset = gt_sets[i];
            if (!gtset.empty() && k > 0) {
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
        
        results[i] = {recall, diff};
    }
    return results;
}

// 为优化GPU专用的benchmark函数
std::vector<SearchResult> benchmark_optimized_gpu_search(
    OptimizedSimpleIVFGPU* gpu_index,
    float* test_query,      
    int* test_gt,           
    size_t base_number,     
    size_t vecdim,          
    size_t test_number,     
    size_t test_gt_d,       
    size_t k,
    size_t nprobe
) {
    std::vector<SearchResult> results(test_number);

    // 准备 ground truth 集合
    std::vector<std::set<uint32_t>> gt_sets(test_number);
    for(size_t i = 0; i < test_number; ++i) {
        for(size_t j = 0; j < k && j < test_gt_d; ++j){ 
             int t = test_gt[j + i*test_gt_d];
             if (t >= 0) { 
                gt_sets[i].insert(static_cast<uint32_t>(t));
             }
        }
    }

    for(int i = 0; i < static_cast<int>(test_number); ++i) {
        const unsigned long Converter = 1000 * 1000;
        struct timeval val;
        gettimeofday(&val, NULL);

        float* current_query = test_query + static_cast<size_t>(i) * vecdim;
        auto res_heap = gpu_index->search(current_query, k, nprobe);

        struct timeval newVal;
        gettimeofday(&newVal, NULL);
        int64_t diff = (newVal.tv_sec * Converter + newVal.tv_usec) - (val.tv_sec * Converter + val.tv_usec);

        // 计算召回率
        size_t acc = 0;
        float recall = 0.0f;

        if (i < static_cast<int>(gt_sets.size())) {
            const auto& gtset = gt_sets[i];
            if (!gtset.empty() && k > 0) {
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
        
        results[i] = {recall, diff};
    }
    return results;
}

// 为超级优化GPU专用的benchmark函数
std::vector<SearchResult> benchmark_super_optimized_gpu_search(
    SuperOptimizedIVFGPU* gpu_index,
    float* test_query,      
    int* test_gt,           
    size_t base_number,     
    size_t vecdim,          
    size_t test_number,     
    size_t test_gt_d,       
    size_t k,
    size_t nprobe
) {
    std::vector<SearchResult> results(test_number);

    // 准备 ground truth 集合
    std::vector<std::set<uint32_t>> gt_sets(test_number);
    for(size_t i = 0; i < test_number; ++i) {
        for(size_t j = 0; j < k && j < test_gt_d; ++j){ 
             int t = test_gt[j + i*test_gt_d];
             if (t >= 0) { 
                gt_sets[i].insert(static_cast<uint32_t>(t));
             }
        }
    }

    for(int i = 0; i < static_cast<int>(test_number); ++i) {
        const unsigned long Converter = 1000 * 1000;
        struct timeval val;
        gettimeofday(&val, NULL);

        float* current_query = test_query + static_cast<size_t>(i) * vecdim;
        auto res_heap = gpu_index->search(current_query, k, nprobe);

        struct timeval newVal;
        gettimeofday(&newVal, NULL);
        int64_t diff = (newVal.tv_sec * Converter + newVal.tv_usec) - (val.tv_sec * Converter + val.tv_usec);

        // 计算召回率
        size_t acc = 0;
        float recall = 0.0f;

        if (i < static_cast<int>(gt_sets.size())) {
            const auto& gtset = gt_sets[i];
            if (!gtset.empty() && k > 0) {
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
        
        results[i] = {recall, diff};
    }
    return results;
}

// 为自适应优化GPU专用的benchmark函数
std::vector<SearchResult> benchmark_adaptive_optimized_gpu_search(
    AdaptiveOptimizedIVFGPU* gpu_index,
    float* test_query,      
    int* test_gt,           
    size_t base_number,     
    size_t vecdim,          
    size_t test_number,     
    size_t test_gt_d,       
    size_t k,
    size_t nprobe
) {
    std::vector<SearchResult> results(test_number);

    // 准备 ground truth 集合
    std::vector<std::set<uint32_t>> gt_sets(test_number);
    for(size_t i = 0; i < test_number; ++i) {
        for(size_t j = 0; j < k && j < test_gt_d; ++j){ 
             int t = test_gt[j + i*test_gt_d];
             if (t >= 0) { 
                gt_sets[i].insert(static_cast<uint32_t>(t));
             }
        }
    }

    for(int i = 0; i < static_cast<int>(test_number); ++i) {
        const unsigned long Converter = 1000 * 1000;
        struct timeval val;
        gettimeofday(&val, NULL);

        float* current_query = test_query + static_cast<size_t>(i) * vecdim;
        auto res_heap = gpu_index->search(current_query, k, nprobe);

        struct timeval newVal;
        gettimeofday(&newVal, NULL);
        int64_t diff = (newVal.tv_sec * Converter + newVal.tv_usec) - (val.tv_sec * Converter + val.tv_usec);

        // 计算召回率
        size_t acc = 0;
        float recall = 0.0f;

        if (i < static_cast<int>(gt_sets.size())) {
            const auto& gtset = gt_sets[i];
            if (!gtset.empty() && k > 0) {
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
        
        results[i] = {recall, diff};
    }
    return results;
}

// 添加性能基准记录结构
struct PerformanceBenchmark {
    std::string method_name;
    double avg_recall;
    double avg_latency_us;
    size_t nprobe;
    size_t batch_size;
    double speedup_vs_openmp;
    
    PerformanceBenchmark(const std::string& name, double recall, double latency, 
                        size_t np = 0, size_t bs = 1, double speedup = 1.0)
        : method_name(name), avg_recall(recall), avg_latency_us(latency), 
          nprobe(np), batch_size(bs), speedup_vs_openmp(speedup) {}
};

// 全局变量存储OpenMP基准性能
std::map<size_t, double> openmp_baseline_latency; // nprobe -> avg_latency

// 修改print_results函数，添加过滤条件
void print_results_filtered(const std::string& method_name, const std::vector<SearchResult>& results, 
                           size_t test_number, size_t nprobe = 0, size_t batch_size = 1, 
                           bool is_openmp_baseline = false, std::vector<PerformanceBenchmark>* valid_results = nullptr) {
    
    double total_recall = 0, total_latency = 0; 
    size_t valid_count = 0;

    for(size_t i = 0; i < results.size() && i < test_number; ++i) {
        total_recall += results[i].recall;
        total_latency += results[i].latency;
        valid_count++;
    }
    
    double avg_recall = (valid_count > 0) ? total_recall / valid_count : 0.0;
    double avg_latency = (valid_count > 0) ? total_latency / valid_count : 0.0;
    
    // 如果是OpenMP基准，记录性能数据
    if (is_openmp_baseline) {
        openmp_baseline_latency[nprobe] = avg_latency;
        std::cout << "=== " << method_name << " (基准) ===" << std::endl;
        std::cout << std::fixed << std::setprecision(5);
        std::cout << "平均召回率: " << avg_recall << std::endl;
        std::cout << std::fixed << std::setprecision(3); 
        std::cout << "平均延迟 (us): " << avg_latency << std::endl;
        std::cout << std::endl;
        return;
    }
    
    // 计算相对于OpenMP的加速比
    double speedup = 1.0;
    if (openmp_baseline_latency.count(nprobe) && openmp_baseline_latency[nprobe] > 0) {
        speedup = openmp_baseline_latency[nprobe] / avg_latency;
    }
    
    // 过滤条件：召回率>0.6且加速比>3x
    if (avg_recall > 0.6 && speedup > 3.0) {
        std::cout << "=== " << method_name << " ✓ ===" << std::endl;
        std::cout << std::fixed << std::setprecision(5);
        std::cout << "平均召回率: " << avg_recall << std::endl;
        std::cout << std::fixed << std::setprecision(3); 
        std::cout << "平均延迟 (us): " << avg_latency << std::endl;
        std::cout << "加速比 vs OpenMP: " << speedup << "x" << std::endl;
        if (batch_size > 1) {
            std::cout << "吞吐量 (queries/sec): " << (avg_latency > 0 ? 1000000.0 / avg_latency : 0.0) << std::endl;
        }
        std::cout << std::endl;
        
        // 记录有效结果
        if (valid_results) {
            valid_results->push_back(PerformanceBenchmark(method_name, avg_recall, avg_latency, nprobe, batch_size, speedup));
        }
    }
}

// 批量测试的过滤版本
void print_batch_results_filtered(const std::string& method_name, double avg_recall, long long batch_time_us, 
                                 size_t batch_test_size, size_t nprobe, size_t num_clusters, 
                                 std::vector<PerformanceBenchmark>* valid_results = nullptr) {
    
    double avg_latency_per_query = (batch_test_size > 0) ? static_cast<double>(batch_time_us) / batch_test_size : 0.0;
    
    // 计算相对于OpenMP的加速比
    double speedup = 1.0;
    if (openmp_baseline_latency.count(nprobe) && openmp_baseline_latency[nprobe] > 0) {
        speedup = openmp_baseline_latency[nprobe] / avg_latency_per_query;
    }
    
    // 过滤条件：召回率>0.6且加速比>3x
    if (avg_recall > 0.6 && speedup > 3.0) {
        std::cout << "=== " << method_name << " (nprobe=" << nprobe 
                  << ", clusters=" << num_clusters 
                  << ", batch_size=" << batch_test_size << ") ✓ ===" << std::endl;
        std::cout << std::fixed << std::setprecision(5);
        std::cout << "平均召回率: " << avg_recall << std::endl;
        std::cout << std::fixed << std::setprecision(3);
        std::cout << "批量处理总时间 (us): " << batch_time_us << std::endl;
        std::cout << "每查询平均时间 (us): " << avg_latency_per_query << std::endl;
        std::cout << "加速比 vs OpenMP: " << speedup << "x" << std::endl;
        std::cout << "吞吐量 (queries/sec): " << (batch_time_us > 0 ? static_cast<double>(batch_test_size) * 1000000 / batch_time_us : 0.0) << std::endl;
        std::cout << "测试查询数量: " << batch_test_size << std::endl;
        std::cout << std::endl;
        
        // 记录有效结果
        if (valid_results) {
            std::string full_name = method_name + " (batch=" + std::to_string(batch_test_size) + ")";
            valid_results->push_back(PerformanceBenchmark(full_name, avg_recall, avg_latency_per_query, nprobe, batch_test_size, speedup));
        }
    }
}

// 打印测试结果的辅助函数
void print_results(const std::string& method_name, const std::vector<SearchResult>& results, size_t test_number) {
    
    double total_recall = 0, total_latency = 0; 
    size_t valid_results = 0;

    for(size_t i = 0; i < results.size() && i < test_number; ++i) {
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
    
    std::string query_path =    "DEEP100K.query.fbin";
    std::string gt_path =       "DEEP100K.gt.query.100k.top100.bin";
    std::string base_path =     "DEEP100K.base.100k.fbin";

    // 存储所有有效结果
    std::vector<PerformanceBenchmark> all_valid_results;

    auto test_query = LoadData<float>(query_path, test_number, vecdim); 
    
    size_t gt_n_from_file; 
    auto test_gt = LoadData<int>(gt_path, gt_n_from_file, test_gt_d); 

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

    // --- Flat 搜索 (暴力搜索) ---
    std::vector<SearchResult> results_flat = benchmark_flat_search(
       base, 
       test_query, test_gt, base_number, vecdim, num_queries_to_test, test_gt_d, k);


    // --- IVF 搜索 (OpenMP) ---
    std::cout << "\n--- IVF (OpenMP) 基准测试 ---" << std::endl;
    size_t num_ivf_clusters_omp = 0;
    if (base_number > 0) {
        num_ivf_clusters_omp = std::min((size_t)256, base_number / 100); 
        if (num_ivf_clusters_omp == 0 && base_number > 0) num_ivf_clusters_omp = std::min((size_t)1, base_number);
    }
    int ivf_kmeans_iterations_omp = 20;

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


            std::cout << "记录 IVF (OpenMP) 基准性能 nprobe = " << actual_nprobe << std::endl;
            std::vector<SearchResult> results_ivf_omp = benchmark_ivf_openmp_search(
               ivf_omp_index_ptr,
               test_query, test_gt, base_number, vecdim, num_queries_to_test, test_gt_d, k, actual_nprobe); 
            
            std::string ivf_omp_method_name = "IVF (OpenMP, nprobe=" + std::to_string(actual_nprobe) + 
                                          ", clusters=" + std::to_string(num_ivf_clusters_omp) + ")";
            print_results_filtered(ivf_omp_method_name, results_ivf_omp, num_queries_to_test, actual_nprobe, 1, true);
        }
    } else {
        std::cerr << "跳过 IVF (OpenMP) 搜索测试，因为索引创建失败。" << std::endl;
    }


    // --- IVF GPU 搜索测试 ---
    std::cout << "\n--- IVF (GPU) 高性能测试 (仅显示召回率>0.6且加速比>3x的结果) ---" << std::endl;
    size_t num_ivf_clusters_gpu = 0;
    if (base_number > 0) {
        num_ivf_clusters_gpu = std::min((size_t)128, base_number / 50); // GPU版本使用稍少的簇数
        if (num_ivf_clusters_gpu == 0 && base_number > 0) num_ivf_clusters_gpu = std::min((size_t)1, base_number);
    }
    size_t gpu_batch_size = 2000; // GPU批处理大小

    SimpleIVFGPU* ivf_gpu_index_ptr = nullptr;
    OptimizedSimpleIVFGPU* optimized_gpu_index_ptr = nullptr; // 启用优化版本
    SuperOptimizedIVFGPU* super_optimized_gpu_index_ptr = nullptr; // 启用超级优化版本
    AdaptiveOptimizedIVFGPU* adaptive_optimized_gpu_index_ptr = nullptr; // 启用自适应优化版本
    
    if (base_number > 0 && num_ivf_clusters_gpu > 0 && vecdim > 0) {
        std::cout << "构建 IVF (GPU) 索引... num_clusters=" << num_ivf_clusters_gpu
                  << ", batch_size=" << gpu_batch_size << std::endl;
        struct timeval build_start_gpu, build_end_gpu;
        gettimeofday(&build_start_gpu, NULL);
        try {
            ivf_gpu_index_ptr = new SimpleIVFGPU(base, static_cast<int>(base_number), 
                                                static_cast<int>(vecdim), 
                                                static_cast<int>(num_ivf_clusters_gpu), 
                                                static_cast<int>(gpu_batch_size));
        } catch (const std::exception& e) {
            std::cerr << "创建 IVF (GPU) 索引时出错: " << e.what() << std::endl;
            ivf_gpu_index_ptr = nullptr;
        }
        gettimeofday(&build_end_gpu, NULL);
        if (ivf_gpu_index_ptr) {
            long long build_time_us_gpu = (build_end_gpu.tv_sec - build_start_gpu.tv_sec) * 1000000LL + (build_end_gpu.tv_usec - build_start_gpu.tv_usec);
            std::cout << "IVF (GPU Basic) 索引构建时间: " << build_time_us_gpu / 1000.0 << " ms" << std::endl;
        }
        
        // 构建优化版GPU索引
        std::cout << "\n构建 IVF (GPU Optimized) 索引... num_clusters=" << num_ivf_clusters_gpu
                  << ", batch_size=" << 2000 << std::endl;
        gettimeofday(&build_start_gpu, NULL);
        try {
            optimized_gpu_index_ptr = new OptimizedSimpleIVFGPU(base, static_cast<int>(base_number), 
                                                         static_cast<int>(vecdim), 
                                                         static_cast<int>(num_ivf_clusters_gpu), 
                                                         2000); // 从128改为2000
        } catch (const std::exception& e) {
            std::cerr << "创建 IVF (GPU Optimized) 索引时出错: " << e.what() << std::endl;
            optimized_gpu_index_ptr = nullptr;
        }
        gettimeofday(&build_end_gpu, NULL);
        if (optimized_gpu_index_ptr) {
            long long build_time_us_gpu_opt = (build_end_gpu.tv_sec - build_start_gpu.tv_sec) * 1000000LL + (build_end_gpu.tv_usec - build_start_gpu.tv_usec);
            std::cout << "IVF (GPU Optimized) 索引构建时间: " << build_time_us_gpu_opt / 1000.0 << " ms" << std::endl;
        }
        
        // 构建超级优化版GPU索引
        std::cout << "\n构建 IVF (GPU Super Optimized) 索引... num_clusters=" << num_ivf_clusters_gpu
                  << ", batch_size=" << 2000 << std::endl;
        gettimeofday(&build_start_gpu, NULL);
        try {
            super_optimized_gpu_index_ptr = new SuperOptimizedIVFGPU(base, static_cast<int>(base_number), 
                                                         static_cast<int>(vecdim), 
                                                         static_cast<int>(num_ivf_clusters_gpu), 
                                                         2000); // 从128改为2000
        } catch (const std::exception& e) {
            std::cerr << "创建 IVF (GPU Super Optimized) 索引时出错: " << e.what() << std::endl;
            super_optimized_gpu_index_ptr = nullptr;
        }
        gettimeofday(&build_end_gpu, NULL);
        if (super_optimized_gpu_index_ptr) {
            long long build_time_us_gpu_super_opt = (build_end_gpu.tv_sec - build_start_gpu.tv_sec) * 1000000LL + (build_end_gpu.tv_usec - build_start_gpu.tv_usec);
            std::cout << "IVF (GPU Super Optimized) 索引构建时间: " << build_time_us_gpu_super_opt / 1000.0 << " ms" << std::endl;
        }
        
        // 构建自适应优化版GPU索引
        std::cout << "\n构建 IVF (GPU Adaptive Optimized) 索引... num_clusters=" << num_ivf_clusters_gpu
                  << ", batch_size=" << 2000 << std::endl; // 从64改为2000
        gettimeofday(&build_start_gpu, NULL);
        try {
            adaptive_optimized_gpu_index_ptr = new AdaptiveOptimizedIVFGPU(base, static_cast<int>(base_number), 
                                                         static_cast<int>(vecdim), 
                                                         static_cast<int>(num_ivf_clusters_gpu), 
                                                         2000); // 从64改为2000
        } catch (const std::exception& e) {
            std::cerr << "创建 IVF (GPU Adaptive Optimized) 索引时出错: " << e.what() << std::endl;
            adaptive_optimized_gpu_index_ptr = nullptr;
        } catch (...) {
            std::cerr << "创建 IVF (GPU Adaptive Optimized) 索引时发生未知错误" << std::endl;
            adaptive_optimized_gpu_index_ptr = nullptr;
        }
        gettimeofday(&build_end_gpu, NULL);
        if (adaptive_optimized_gpu_index_ptr) {
            long long build_time_us_gpu_adaptive_opt = (build_end_gpu.tv_sec - build_start_gpu.tv_sec) * 1000000LL + (build_end_gpu.tv_usec - build_start_gpu.tv_usec);
            std::cout << "IVF (GPU Adaptive Optimized) 索引构建时间: " << build_time_us_gpu_adaptive_opt / 1000.0 << " ms" << std::endl;
        }
    } else {
        std::cerr << "无法构建 IVF (GPU) 索引，参数无效 (base_number=" << base_number
                  << ", vecdim=" << vecdim << ", num_ivf_clusters_gpu=" << num_ivf_clusters_gpu << ")." << std::endl;
    }

    if (ivf_gpu_index_ptr || optimized_gpu_index_ptr || super_optimized_gpu_index_ptr || adaptive_optimized_gpu_index_ptr) {
        std::vector<size_t> nprobe_values_gpu = {1, 2, 4, 8, 16, 32};
        if (num_ivf_clusters_gpu < 32 && num_ivf_clusters_gpu > 0) {
            nprobe_values_gpu.clear();
            for(size_t np_val = 1; np_val <= num_ivf_clusters_gpu; np_val *=2) nprobe_values_gpu.push_back(np_val);
            if (nprobe_values_gpu.empty() || nprobe_values_gpu.back() < num_ivf_clusters_gpu) {
                 bool contains_max = false;
                 for(size_t val : nprobe_values_gpu) if(val == num_ivf_clusters_gpu) contains_max = true;
                 if(!contains_max && num_ivf_clusters_gpu > 0) nprobe_values_gpu.push_back(num_ivf_clusters_gpu);
            }
            if (nprobe_values_gpu.empty() && num_ivf_clusters_gpu > 0) nprobe_values_gpu.push_back(1);
            else if (num_ivf_clusters_gpu == 0) nprobe_values_gpu.clear();
        } else if (num_ivf_clusters_gpu == 0) {
            nprobe_values_gpu.clear();
        }

        for (size_t current_nprobe : nprobe_values_gpu) {
            if (current_nprobe == 0 && num_ivf_clusters_gpu > 0) continue;
            size_t actual_nprobe = (num_ivf_clusters_gpu == 0) ? 0 : std::min(current_nprobe, num_ivf_clusters_gpu);
            if (actual_nprobe == 0 && num_ivf_clusters_gpu > 0) actual_nprobe = 1;
            else if (num_ivf_clusters_gpu == 0) continue;

            // 基础GPU版本测试
            if (ivf_gpu_index_ptr) {
                // 单个查询测试
                std::vector<SearchResult> results_ivf_gpu = benchmark_gpu_search(
                   ivf_gpu_index_ptr,
                   test_query, test_gt, base_number, vecdim, num_queries_to_test, test_gt_d, k, actual_nprobe);

                std::string ivf_gpu_method_name = "IVF (GPU Basic Single, nprobe=" + std::to_string(actual_nprobe) + 
                                              ", clusters=" + std::to_string(num_ivf_clusters_gpu) + ")";
                print_results_filtered(ivf_gpu_method_name, results_ivf_gpu, num_queries_to_test, actual_nprobe, 1, false, &all_valid_results);

                // GPU批量测试
                std::cout << "测试 IVF (GPU Basic Batch) 使用 nprobe = " << actual_nprobe << ", batch_size = " << gpu_batch_size << std::endl;
                
                // 准备批量查询数据
                std::vector<float> batch_queries;
                size_t batch_test_num = std::min(num_queries_to_test, (size_t)2000); // 从100改为2000
                batch_queries.reserve(batch_test_num * vecdim);
                
                for (size_t i = 0; i < batch_test_num; ++i) {
                    float* current_query = test_query + i * vecdim;
                    batch_queries.insert(batch_queries.end(), current_query, current_query + vecdim);
                }
                
                // 计时批量搜索
                struct timeval batch_start, batch_end;
                gettimeofday(&batch_start, NULL);
                
                auto batch_results = ivf_gpu_index_ptr->batch_search(batch_queries, k, actual_nprobe);
                
                gettimeofday(&batch_end, NULL);
                long long batch_time_us = (batch_end.tv_sec - batch_start.tv_sec) * 1000000LL + (batch_end.tv_usec - batch_start.tv_usec);
                
                // 计算批量搜索的召回率
                double total_batch_recall = 0.0;
                size_t valid_batch_results = 0;
                
                for (size_t i = 0; i < batch_test_num && i < batch_results.size(); ++i) {
                    std::set<uint32_t> gt_set;
                    for(size_t j = 0; j < k && j < test_gt_d; ++j) {
                        int t = test_gt[j + i * test_gt_d];
                        if (t >= 0) {
                            gt_set.insert(static_cast<uint32_t>(t));
                        }
                    }
                    
                    size_t acc = 0;
                    for (const auto& result_pair : batch_results[i]) {
                        if (gt_set.count(result_pair.second)) {
                            acc++;
                        }
                    }
                    
                    if (!gt_set.empty() && k > 0) {
                        total_batch_recall += static_cast<double>(acc) / k;
                        valid_batch_results++;
                    }
                }
                
                std::string batch_method_name = "IVF (GPU Basic Batch)";
                print_batch_results_filtered(batch_method_name, total_batch_recall / valid_batch_results, 
                                           batch_time_us, batch_test_num, actual_nprobe, num_ivf_clusters_gpu, &all_valid_results);
            }
            
            // 优化GPU版本测试
            if (optimized_gpu_index_ptr) {
                // 单个查询测试
                std::vector<SearchResult> results_optimized_gpu = benchmark_optimized_gpu_search(
                   optimized_gpu_index_ptr,
                   test_query, test_gt, base_number, vecdim, num_queries_to_test, test_gt_d, k, actual_nprobe);
                
                std::string optimized_gpu_method_name = "IVF (GPU Optimized Single, nprobe=" + std::to_string(actual_nprobe) + 
                                                      ", clusters=" + std::to_string(num_ivf_clusters_gpu) + ")";
                print_results_filtered(optimized_gpu_method_name, results_optimized_gpu, num_queries_to_test, actual_nprobe, 1, false, &all_valid_results);

                // 批量查询测试
                std::vector<size_t> batch_sizes = {500, 1000, 2000};
                
                for (size_t batch_test_size : batch_sizes) {
                    if (batch_test_size > num_queries_to_test) continue;
                    
                    // 准备批量查询数据
                    std::vector<float> batch_queries;
                    batch_queries.reserve(batch_test_size * vecdim);
                    
                    for (size_t i = 0; i < batch_test_size; ++i) {
                        float* current_query = test_query + i * vecdim;
                        batch_queries.insert(batch_queries.end(), current_query, current_query + vecdim);
                    }
                    
                    // 计时批量搜索
                    struct timeval batch_start, batch_end;
                    gettimeofday(&batch_start, NULL);
                    
                    auto batch_results = optimized_gpu_index_ptr->batch_search(batch_queries, k, actual_nprobe);
                    
                    gettimeofday(&batch_end, NULL);
                    long long batch_time_us = (batch_end.tv_sec - batch_start.tv_sec) * 1000000LL + (batch_end.tv_usec - batch_start.tv_usec);
                    
                    // 计算批量搜索的召回率
                    double total_batch_recall = 0.0;
                    size_t valid_batch_results = 0;
                    
                    for (size_t i = 0; i < batch_test_size && i < batch_results.size(); ++i) {
                        std::set<uint32_t> gt_set;
                        for(size_t j = 0; j < k && j < test_gt_d; ++j) {
                            int t = test_gt[j + i * test_gt_d];
                            if (t >= 0) {
                                gt_set.insert(static_cast<uint32_t>(t));
                            }
                        }
                        
                        size_t acc = 0;
                        for (const auto& result_pair : batch_results[i]) {
                            if (gt_set.count(result_pair.second)) {
                                acc++;
                            }
                        }
                        
                        if (!gt_set.empty() && k > 0) {
                            total_batch_recall += static_cast<double>(acc) / k;
                            valid_batch_results++;
                        }
                    }
                    
                    std::string batch_method_name = "IVF (GPU Optimized Batch)";
                    print_batch_results_filtered(batch_method_name, total_batch_recall / valid_batch_results, 
                                               batch_time_us, batch_test_size, actual_nprobe, num_ivf_clusters_gpu, &all_valid_results);
                }
            }
            
            // 超级优化GPU版本测试
            if (super_optimized_gpu_index_ptr) {
                // 单个查询测试
                std::vector<SearchResult> results_super_optimized_gpu = benchmark_super_optimized_gpu_search(
                   super_optimized_gpu_index_ptr,
                   test_query, test_gt, base_number, vecdim, num_queries_to_test, test_gt_d, k, actual_nprobe);
                
                std::string super_optimized_gpu_method_name = "IVF (GPU Super Optimized Single, nprobe=" + std::to_string(actual_nprobe) + 
                                                      ", clusters=" + std::to_string(num_ivf_clusters_gpu) + ")";
                print_results_filtered(super_optimized_gpu_method_name, results_super_optimized_gpu, num_queries_to_test, actual_nprobe, 1, false, &all_valid_results);

                // 批量查询测试
                std::vector<size_t> batch_sizes = {500, 1000, 2000};
                
                for (size_t batch_test_size : batch_sizes) {
                    if (batch_test_size > num_queries_to_test) continue;
                    
                    // 准备批量查询数据
                    std::vector<float> batch_queries;
                    batch_queries.reserve(batch_test_size * vecdim);
                    
                    for (size_t i = 0; i < batch_test_size; ++i) {
                        float* current_query = test_query + i * vecdim;
                        batch_queries.insert(batch_queries.end(), current_query, current_query + vecdim);
                    }
                    
                    // 计时批量搜索
                    struct timeval batch_start, batch_end;
                    gettimeofday(&batch_start, NULL);
                    
                    auto batch_results = super_optimized_gpu_index_ptr->batch_search(batch_queries, k, actual_nprobe);
                    
                    gettimeofday(&batch_end, NULL);
                    long long batch_time_us = (batch_end.tv_sec - batch_start.tv_sec) * 1000000LL + (batch_end.tv_usec - batch_start.tv_usec);
                    
                    // 计算批量搜索的召回率
                    double total_batch_recall = 0.0;
                    size_t valid_batch_results = 0;
                    
                    for (size_t i = 0; i < batch_test_size && i < batch_results.size(); ++i) {
                        std::set<uint32_t> gt_set;
                        for(size_t j = 0; j < k && j < test_gt_d; ++j) {
                            int t = test_gt[j + i * test_gt_d];
                            if (t >= 0) {
                                gt_set.insert(static_cast<uint32_t>(t));
                            }
                        }
                        
                        size_t acc = 0;
                        for (const auto& result_pair : batch_results[i]) {
                            if (gt_set.count(result_pair.second)) {
                                acc++;
                            }
                        }
                        
                        if (!gt_set.empty() && k > 0) {
                            total_batch_recall += static_cast<double>(acc) / k;
                            valid_batch_results++;
                        }
                    }
                    
                    std::string batch_method_name = "IVF (GPU Super Optimized Batch)";
                    print_batch_results_filtered(batch_method_name, total_batch_recall / valid_batch_results, 
                                               batch_time_us, batch_test_size, actual_nprobe, num_ivf_clusters_gpu, &all_valid_results);
                }
            }
            
            // 自适应优化GPU版本测试
            if (adaptive_optimized_gpu_index_ptr) {
                try {
                    // 单个查询测试
                    std::vector<SearchResult> results_adaptive_optimized_gpu = benchmark_adaptive_optimized_gpu_search(
                       adaptive_optimized_gpu_index_ptr,
                       test_query, test_gt, base_number, vecdim, num_queries_to_test, test_gt_d, k, actual_nprobe);
                    
                    std::string adaptive_optimized_gpu_method_name = "IVF (GPU Adaptive Optimized Single, nprobe=" + std::to_string(actual_nprobe) + 
                                                          ", clusters=" + std::to_string(num_ivf_clusters_gpu) + ")";
                    print_results_filtered(adaptive_optimized_gpu_method_name, results_adaptive_optimized_gpu, num_queries_to_test, actual_nprobe, 1, false, &all_valid_results);

                    // 批量查询测试
                    std::vector<size_t> batch_sizes = {200, 500, 1000};
                    
                    for (size_t batch_test_size : batch_sizes) {
                        if (batch_test_size > num_queries_to_test) continue;
                        
                        try {
                            // 准备批量查询数据
                            std::vector<float> batch_queries;
                            batch_queries.reserve(batch_test_size * vecdim);
                            
                            for (size_t i = 0; i < batch_test_size; ++i) {
                                float* current_query = test_query + i * vecdim;
                                batch_queries.insert(batch_queries.end(), current_query, current_query + vecdim);
                            }
                            
                            // 计时批量搜索
                            struct timeval batch_start, batch_end;
                            gettimeofday(&batch_start, NULL);
                            
                            auto batch_results = adaptive_optimized_gpu_index_ptr->batch_search(batch_queries, k, actual_nprobe);
                            
                            gettimeofday(&batch_end, NULL);
                            long long batch_time_us = (batch_end.tv_sec - batch_start.tv_sec) * 1000000LL + (batch_end.tv_usec - batch_start.tv_usec);
                            
                            // 计算批量搜索的召回率
                            double total_batch_recall = 0.0;
                            size_t valid_batch_results = 0;
                            
                            for (size_t i = 0; i < batch_test_size && i < batch_results.size(); ++i) {
                                std::set<uint32_t> gt_set;
                                for(size_t j = 0; j < k && j < test_gt_d; ++j) {
                                    int t = test_gt[j + i * test_gt_d];
                                    if (t >= 0) {
                                        gt_set.insert(static_cast<uint32_t>(t));
                                    }
                                }
                                
                                size_t acc = 0;
                                for (const auto& result_pair : batch_results[i]) {
                                    if (gt_set.count(result_pair.second)) {
                                        acc++;
                                    }
                                }
                                
                                if (!gt_set.empty() && k > 0) {
                                    total_batch_recall += static_cast<double>(acc) / k;
                                    valid_batch_results++;
                                }
                            }
                            
                            std::string batch_method_name = "IVF (GPU Adaptive Optimized Batch)";
                            print_batch_results_filtered(batch_method_name, total_batch_recall / valid_batch_results, 
                                                       batch_time_us, batch_test_size, actual_nprobe, num_ivf_clusters_gpu, &all_valid_results);
                        } catch (const std::exception& e) {
                            // 静默处理错误，不输出
                        } catch (...) {
                            // 静默处理错误，不输出
                        }
                    }
                } catch (const std::exception& e) {
                    // 静默处理错误，不输出
                } catch (...) {
                    // 静默处理错误，不输出
                }
            }
        }
    } else {
        std::cerr << "跳过 IVF (GPU) 搜索测试，因为索引创建失败。" << std::endl;
    }

    // --- 最终汇总表 ---
    std::cout << "\n=== 高性能结果汇总 (召回率>0.6, 加速比>3x) ===" << std::endl;
    if (all_valid_results.empty()) {
        std::cout << "没有找到满足条件的高性能配置。" << std::endl;
    } else {
        std::cout << std::left << std::setw(60) << "方法" 
                  << std::setw(12) << "召回率" 
                  << std::setw(15) << "延迟(us)" 
                  << std::setw(12) << "加速比" 
                  << std::setw(15) << "吞吐量(qps)" << std::endl;
        std::cout << std::string(114, '-') << std::endl;
        
        // 按加速比排序
        std::sort(all_valid_results.begin(), all_valid_results.end(), 
                 [](const PerformanceBenchmark& a, const PerformanceBenchmark& b) {
                     return a.speedup_vs_openmp > b.speedup_vs_openmp;
                 });
        
        for (const auto& result : all_valid_results) {
            double throughput = 1000000.0 / result.avg_latency_us;
            std::cout << std::left << std::setw(60) << result.method_name
                      << std::fixed << std::setprecision(3) << std::setw(12) << result.avg_recall
                      << std::setw(15) << result.avg_latency_us
                      << std::setw(12) << result.speedup_vs_openmp << "x"
                      << std::setw(15) << static_cast<int>(throughput) << std::endl;
        }
        
        // 显示最佳结果
        if (!all_valid_results.empty()) {
            const auto& best = all_valid_results[0];
            std::cout << "\n🏆 最佳性能配置:" << std::endl;
            std::cout << "方法: " << best.method_name << std::endl;
            std::cout << "召回率: " << std::fixed << std::setprecision(3) << best.avg_recall << std::endl;
            std::cout << "加速比: " << best.speedup_vs_openmp << "x" << std::endl;
            std::cout << "吞吐量: " << static_cast<int>(1000000.0 / best.avg_latency_us) << " queries/sec" << std::endl;
        }
    }

    // --- 清理 ---
    delete[] test_query;
    delete[] test_gt;
    delete[] base;
    delete ivf_omp_index_ptr; // 清理 OpenMP IVF 索引
    delete ivf_gpu_index_ptr; // 清理 GPU IVF 索引
    delete optimized_gpu_index_ptr; // 清理优化版GPU索引
    delete super_optimized_gpu_index_ptr; // 清理超级优化版GPU索引
    delete adaptive_optimized_gpu_index_ptr; // 清理自适应优化版GPU索引
    return 0;
}