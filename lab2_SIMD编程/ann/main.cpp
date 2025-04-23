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
// #include "hnswlib/hnswlib/hnswlib.h"
#include "flat_scan.h"
#include "simd_anns.h"
#include "pq_anns.h"
#include "sq_anns.h"
#include <functional>
#include <algorithm>
#include <queue>
#include <stdexcept>
#include <cstdlib> // for std::atoi, std::stoul, std::stod

// --- 函数声明 ---
std::priority_queue<std::pair<float, uint32_t>> flat_search(float* base, const float* query, size_t base_number, size_t vecdim, size_t k);
std::priority_queue<std::pair<float, uint32_t>> simd_search(float* base, const float* query, size_t base_number, size_t vecdim, size_t k);
// ---

// ... LoadData, SearchResult, benchmark_search, print_results 函数保持不变 ...
// (确保 benchmark_search 使用的是 Recall@k vs GT@k 的版本)
template<typename T>
T *LoadData(std::string data_path, size_t& n, size_t& d)
{
    std::ifstream fin;
    // 以读取+二进制的模式打开文件
    fin.open(data_path, std::ios::in | std::ios::binary);
    // 将n的地址强制转换为char*，因为read函数需要指向字节的指针
    // 读取文件的前8个字节，分别存储数量n和维度d
    fin.read((char*)&n,4);
    fin.read((char*)&d,4);
    T* data = new T[n*d];
    int sz = sizeof(T);// 单个元素的字节大小
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
    size_t k                // 搜索时要返回的近邻数
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
    #pragma omp parallel for schedule(dynamic)
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
             // *** gtset 现在只包含 GT 的前 k 个 ***
            const auto& gtset = gt_sets[i];
            if (!gtset.empty() && k > 0) {
                size_t results_count = 0;
                std::vector<uint32_t> result_indices;
                result_indices.reserve(k);

                // 从堆中取出最多 k 个元素
                while (!res_heap.empty() && result_indices.size() < k) {
                    result_indices.push_back(res_heap.top().second);
                    res_heap.pop();
                }

                // 计算与 ground truth (Top-k) 的交集大小
                for(uint32_t x : result_indices) {
                     if(gtset.count(x)){ // 与 GT 的 Top-k 比较
                        ++acc;
                    }
                }
                // 按照 k 计算召回率
                recall = static_cast<float>(acc) / k;
            } else if (k == 0) {
                recall = 1.0f; // k=0 时召回率为 1
            }
            // 如果 gtset 为空 (因为 k > test_gt_d 或 GT 无效) 或 k=0 (已处理), recall 保持 0.0f
        }
        // 如果 GT 无效，recall 保持 0.0f

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
    
    double total_recall = 0, total_latency = 0; // Use double for accumulation
    size_t valid_results = std::min(test_number, results.size());

    for(size_t i = 0; i < valid_results; ++i) {
        total_recall += results[i].recall;
        total_latency += results[i].latency;
    }

    std::cout << "=== " << method_name << " ===" << std::endl;
    std::cout << std::fixed << std::setprecision(5);
    std::cout << "平均召回率: " << (valid_results > 0 ? total_recall / valid_results : 0.0) << std::endl;
    std::cout << std::fixed << std::setprecision(3); // 延迟保留3位小数
    std::cout << "平均延迟 (us): " << (valid_results > 0 ? total_latency / valid_results : 0.0) << std::endl;
    std::cout << std::endl;
}


// --- 用于解析命令行参数的辅助函数 ---
long get_arg_long(char** begin, char** end, const std::string& option, long default_val) {
    char** itr = std::find(begin, end, option);
    if (itr != end && ++itr != end) {
        try {
            return std::stol(*itr);
        } catch (const std::exception& e) {
            std::cerr << "警告: 无法解析参数 " << option << " 的值 '" << *itr << "'. 使用默认值: " << default_val << std::endl;
        }
    }
    return default_val;
}

int main(int argc, char *argv[])
{
    // --- 解析命令行参数 ---
    // 默认值
    size_t target_nsub = 16;
    size_t target_rerank_k = 50;
    size_t num_queries_to_test = 2000;
    size_t k = 10;

    // 从 argv 解析 (简单的示例)
    target_nsub = get_arg_long(argv, argv + argc, "--nsub", target_nsub);
    target_rerank_k = get_arg_long(argv, argv + argc, "--rerank", target_rerank_k);
    num_queries_to_test = get_arg_long(argv, argv + argc, "--num_queries", num_queries_to_test);
    k = get_arg_long(argv, argv + argc, "--k", k);

    // --- 加载数据 ---
    size_t test_number = 0, base_number = 0;
    size_t test_gt_d = 0, vecdim = 0;

    std::string query_path = "DEEP100K.query.fbin";
    std::string gt_path = "DEEP100K.gt.query.100k.top100.bin";
    std::string base_path = "DEEP100K.base.100k.fbin";

    auto test_query = LoadData<float>(query_path, test_number, vecdim);
    auto test_gt = LoadData<int>(gt_path, test_number, test_gt_d);
    auto base = LoadData<float>(base_path, base_number, vecdim);

    if (!test_query || !base || vecdim == 0 || base_number == 0 || test_number == 0) {
        std::cerr << "错误：加载查询或基向量数据失败，或数据维度/数量无效，程序终止。" << std::endl;
        delete[] test_query; delete[] test_gt; delete[] base; return 1;
    }
    if (!test_gt || test_gt_d == 0) {
         std::cerr << "警告: 加载 ground truth 数据失败或维度为 0。召回率将为 0。" << std::endl;
    }
    if (num_queries_to_test > test_number) {
        num_queries_to_test = test_number;
    }
     // 打印将要使用的参数
    // std::cout << "测试查询数量: " << num_queries_to_test << ", k=" << k << std::endl;
    // std::cout << "PQ 参数: nsub=" << target_nsub << ", rerank_k=" << target_rerank_k << std::endl;


    if (k > test_gt_d && test_gt) {
        std::cerr << "警告: 请求的 k=" << k << " 大于 ground truth 的维度 test_gt_d=" << test_gt_d << std::endl;
    }

    // --- 运行 PQ 测试 ---
    ProductQuantizer* pq_index_ptr = nullptr;
    std::vector<SearchResult> results_pq;
    double avg_recall_pq = 0.0;
    double avg_latency_pq = 0.0;
    bool pq_success = false;

    if (vecdim == 0) {
         std::cerr << "错误: 向量维度为 0，无法创建 PQ 索引。" << std::endl;
    } else if (vecdim % target_nsub != 0) {
        std::cerr << "错误: 向量维度 (" << vecdim << ") 不能被 nsub (" << target_nsub << ") 整除。跳过此 PQ 配置。" << std::endl;
    } else {
         try {
              // 注意：PQ 训练和编码只应在必要时进行，如果索引已存在可考虑加载
              // 为简化脚本，这里每次都重新训练
              std::cerr << "训练 PQ 索引 (nsub=" << target_nsub << ")..." << std::endl;
              pq_index_ptr = new ProductQuantizer(base, base_number, vecdim, target_nsub, 1.0); // train_ratio=1.0
              std::cerr << "PQ 索引创建完毕。" << std::endl;

              if (pq_index_ptr) {
                  auto pq_search_lambda = [&](const float* q, size_t k_param) {
                      return pq_index_ptr->search(q, base, k_param, target_rerank_k); // 使用命令行参数
                  };
                  std::cerr << "开始 PQ 基准测试 (rerank=" << target_rerank_k << ")..." << std::endl;
                  results_pq = benchmark_search(
                     pq_search_lambda,
                     test_query, test_gt, base_number, vecdim, num_queries_to_test, test_gt_d, k);
                  std::cerr << "PQ 基准测试完成。" << std::endl;

                  // 计算平均值
                  if (!results_pq.empty()) {
                      for(const auto& res : results_pq) {
                          avg_recall_pq += res.recall;
                          avg_latency_pq += res.latency;
                      }
                      avg_recall_pq /= results_pq.size();
                      avg_latency_pq /= results_pq.size();
                      pq_success = true;
                  }
              }
         } catch (const std::exception& e) {
              std::cerr << "运行 PQ 测试时发生错误 (nsub=" << target_nsub << ", rerank=" << target_rerank_k << "): " << e.what() << std::endl;
         }
    }

    // --- 输出结果到标准输出 (CSV 格式) ---
    // 脚本将捕获此输出
    if (pq_success) {
        std::cout << target_nsub << ","
                  << target_rerank_k << ","
                  << std::fixed << std::setprecision(5) << avg_recall_pq << ","
                  << std::fixed << std::setprecision(3) << avg_latency_pq
                  << std::endl;
    } else {
         // 输出一个标记失败的行或保持沉默，取决于脚本如何处理
         // 这里输出 0 值表示失败
         std::cout << target_nsub << ","
                   << target_rerank_k << ","
                   << std::fixed << std::setprecision(5) << 0.0 << ","
                   << std::fixed << std::setprecision(3) << 0.0
                   << std::endl;
    }

    // --- 清理 ---
    delete[] test_query;
    delete[] test_gt;
    delete[] base;
    delete pq_index_ptr;

    return 0;
}