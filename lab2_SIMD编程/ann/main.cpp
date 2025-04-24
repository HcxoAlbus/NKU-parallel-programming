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

// --- benchmark_search 函数定义 (确保是 9 参数版本) ---
template<typename SearchFunc>
std::vector<SearchResult> benchmark_search(
    SearchFunc search_func, // 1. 搜索函数 (flat_search, simd_search, 或 lambda)
    float* base,           // 2. 基向量数据 (传递给需要它的 search_func)
    float* test_query,     // 3. 查询向量数据
    int* test_gt,          // 4. Ground truth 数据
    size_t base_number,    // 5. 基向量数量 (传递给需要它的 search_func)
    size_t vecdim,         // 6. 向量维度 (传递给需要它的 search_func)
    size_t test_number,    // 7. 要测试的查询数量
    size_t test_gt_d,      // 8. Ground truth 维度
    size_t k               // 9. 搜索的近邻数 (传递给 search_func)
) {
    std::vector<SearchResult> results(test_number);

    // 准备 ground truth 集合 (使用 GT 的前 k 个)
    std::vector<std::set<uint32_t>> gt_sets(test_number);
    bool gt_valid = (test_gt != nullptr);
    if (gt_valid) {
        for(size_t i = 0; i < test_number; ++i) {
            for(size_t j = 0; j < k && j < test_gt_d; ++j){ // 只取 GT 的前 k 个
                 int t = test_gt[j + i*test_gt_d];
                 if (t >= 0) { gt_sets[i].insert(static_cast<uint32_t>(t)); }
            }
        }
    } else {
         std::cerr << "警告: Ground truth 数据为空，无法计算召回率。" << std::endl;
    }

    // 并行或串行执行搜索
    // #pragma omp parallel for schedule(dynamic) // 可以选择是否并行化 benchmark
    for(int i = 0; i < static_cast<int>(test_number); ++i) {
        const unsigned long Converter = 1000 * 1000;
        struct timeval val;
        gettimeofday(&val, NULL);

        const float* current_query = test_query + static_cast<size_t>(i) * vecdim;

        // *** 调用 search_func ***
        // 这里需要根据 search_func 的实际签名来调用
        // 假设 flat_search 和 simd_search 接受 (base, query, base_number, vecdim, k)
        // 而 SQ 的 lambda 只接受 (query, k)
        // 为了通用性，最好修改 search_func 的接口或使用 std::function/std::bind
        // 临时的解决方法：假设 search_func 总是接受这 5 个参数，SQ 的 lambda 需要捕获 base, base_number, vecdim
        auto res_heap = search_func(base, current_query, base_number, vecdim, k);
        // 如果 SearchFunc 是 SQ 的 lambda (只接受 query, k)，上面的调用会失败
        // 需要更复杂的处理，例如类型检查或不同的 benchmark 函数

        struct timeval newVal;
        gettimeofday(&newVal, NULL);
        int64_t diff = (newVal.tv_sec * Converter + newVal.tv_usec) - (val.tv_sec * Converter + val.tv_usec);

        // 计算召回率
        size_t acc = 0;
        float recall = 0.0f;
        if (gt_valid && i < static_cast<int>(gt_sets.size())) {
            const auto& gtset = gt_sets[i];
            if (!gtset.empty() && k > 0) {
                size_t count = 0;
                while (!res_heap.empty() && count < k) { // 最多检查 k 个结果
                    if(gtset.count(res_heap.top().second)){
                        ++acc;
                    }
                    res_heap.pop();
                    count++;
                }
                recall = static_cast<float>(acc) / k;
            } else if (k == 0) {
                recall = 1.0f;
            }
        }

        // 使用 #pragma omp critical 保护对 results 的写入（如果使用 omp parallel for）
        // #pragma omp critical
        {
             if (static_cast<size_t>(i) < results.size()) {
                results[i] = {recall, diff};
             }
        }
    }
    return results;
}


// --- 用于解析命令行参数的辅助函数 (修正版) ---
long get_arg_long(char** begin, char** end, const std::string& option, long default_val) {
    char** itr = begin;
    while (itr != end) {
        // 比较 C 风格字符串
        if (strcmp(*itr, option.c_str()) == 0) {
            // 找到了选项，下一个应该是值
            if (++itr != end) {
                try {
                    return std::stol(*itr);
                } catch (const std::exception& e) {
                    // std::cerr << "警告: 无法解析参数 " << option << " 的值 '" << *itr << "'. 使用默认值: " << default_val << std::endl;
                }
            }
            // 选项后面没有值，跳出循环返回默认值
            break;
        }
        ++itr;
    }
    return default_val;
}

const char* get_arg_string(char** begin, char** end, const std::string& option, const char* default_val) {
    char** itr = begin;
    while (itr != end) {
        if (strcmp(*itr, option.c_str()) == 0) {
            if (++itr != end) {
                return *itr; // 返回值的 C 字符串指针
            }
            break;
        }
        ++itr;
    }
    return default_val;
}


int main(int argc, char *argv[])
{
    // --- 解析命令行参数 ---
    const char* target_algo = get_arg_string(argv + 1, argv + argc, "--algo", "all"); // 从 argv+1 开始查找
    size_t num_queries_to_test = get_arg_long(argv + 1, argv + argc, "--num_queries", 2000);
    size_t k = get_arg_long(argv + 1, argv + argc, "--k", 10);

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
    if (!test_gt || test_gt_d == 0) { /* 警告 */ }
    if (num_queries_to_test > test_number) { num_queries_to_test = test_number; }
    if (k > test_gt_d && test_gt) { /* 警告 */ }
    std::cerr << "请求算法: " << target_algo << ", 测试查询数: " << num_queries_to_test << ", k=" << k << std::endl;


    // --- 根据参数选择并运行算法 ---
    std::string algo_name = "";
    std::vector<SearchResult> results;
    bool success = false;
    ScalarQuantizer* sq_quantizer_ptr = nullptr; // SQ 需要先构建

    try {
        if (strcmp(target_algo, "flat") == 0 || strcmp(target_algo, "all") == 0) {
            algo_name = "flat";
            std::cerr << "运行 Flat Search..." << std::endl;
            // 传递 flat_search 函数指针，benchmark_search 内部会调用它
            results = benchmark_search(flat_search, base, test_query, test_gt, base_number, vecdim, num_queries_to_test, test_gt_d, k);
            success = true;
            std::cerr << "Flat Search 完成." << std::endl;
            if (strcmp(target_algo, "all") != 0) goto end_algo_run;
        }

        if (strcmp(target_algo, "simd") == 0 || strcmp(target_algo, "all") == 0) {
             if (strcmp(target_algo, "all") == 0) results.clear();
             algo_name = "simd";
             std::cerr << "运行 SIMD Search..." << std::endl;
             results = benchmark_search(simd_search, base, test_query, test_gt, base_number, vecdim, num_queries_to_test, test_gt_d, k);
             success = true;
             std::cerr << "SIMD Search 完成." << std::endl;
             if (strcmp(target_algo, "all") != 0) goto end_algo_run;
        }

        if (strcmp(target_algo, "sq") == 0 || strcmp(target_algo, "all") == 0) {
            if (strcmp(target_algo, "all") == 0) results.clear();
            algo_name = "sq";
            std::cerr << "创建 SQ 索引..." << std::endl;
            sq_quantizer_ptr = new ScalarQuantizer(base, base_number, vecdim); // 使用指针管理
            std::cerr << "SQ 索引完成. 运行 SQ Search..." << std::endl;

            // 创建一个 lambda 适配器，使其符合 benchmark_search 期望的签名
            // 它捕获 sq_quantizer_ptr
            auto sq_search_adapter = [&](float* /*base_ignored*/, const float* q, size_t /*b_num_ignored*/, size_t /*dim_ignored*/, size_t k_param) {
                // 实际调用 SQ 的成员函数，忽略传入的 base 相关参数
                return sq_quantizer_ptr->sq_search(q, k_param);
            };

            // 将适配器 lambda 传递给 benchmark_search
            results = benchmark_search(sq_search_adapter, base, test_query, test_gt, base_number, vecdim, num_queries_to_test, test_gt_d, k);
            success = true;
            std::cerr << "SQ Search 完成." << std::endl;
            if (strcmp(target_algo, "all") != 0) goto end_algo_run;
        }

        if (!success && strcmp(target_algo, "all") != 0) {
             std::cerr << "错误: 未知或未执行的算法 '" << target_algo << "'" << std::endl;
        }

    } catch (const std::exception& e) {
        std::cerr << "运行算法 " << algo_name << " 时发生错误: " << e.what() << std::endl;
        success = false;
    }

end_algo_run:

    // --- 输出结果 (如果成功运行了单个算法) ---
    if (success && !results.empty() && strcmp(target_algo, "all") != 0) {
        double avg_recall = 0.0;
        double avg_latency = 0.0;
        for(const auto& res : results) {
            avg_recall += res.recall;
            avg_latency += res.latency;
        }
        avg_recall /= results.size();
        avg_latency /= results.size();

        std::cout << algo_name << ","
                  << std::fixed << std::setprecision(5) << avg_recall << ","
                  << std::fixed << std::setprecision(3) << avg_latency
                  << std::endl;
    } else if (strcmp(target_algo, "all") == 0) {
         // std::cerr << "模式 'all' 不输出 CSV，请为每个算法单独运行。" << std::endl;
    } else if (!success) {
         // std::cout << target_algo << ",0.00000,0.000" << std::endl; // 输出失败标记
    }


    // --- 清理 ---
    delete[] test_query;
    delete[] test_gt;
    delete[] base;
    delete sq_quantizer_ptr; // 清理 SQ 对象

    return success ? 0 : 1;
}

// --- LoadData 函数定义 (需要你提供或确认已存在) ---
template<typename T>
T *LoadData(const std::string& data_path, size_t& n, size_t& d) {
    std::ifstream fin(data_path, std::ios::binary);
    if (!fin) {
        std::cerr << "错误: 无法打开数据文件 " << data_path << std::endl;
        n = 0; d = 0;
        return nullptr;
    }
    fin.read(reinterpret_cast<char*>(&n), sizeof(size_t)); // 假设文件中的大小是 size_t
    fin.read(reinterpret_cast<char*>(&d), sizeof(size_t)); // 假设文件中的大小是 size_t

    // 添加基本的健���性检查
    if (n == 0 || d == 0 || n > 1000000 || d > 10000) { // 设定一些合理的上限
         std::cerr << "错误: 从文件 " << data_path << " 读取的维度 (" << d << ") 或数量 (" << n << ") 无效。" << std::endl;
         n = 0; d = 0;
         fin.close();
         return nullptr;
    }


    size_t num_elements = n * d;
    T* data = nullptr;
    try {
         data = new T[num_elements];
    } catch (const std::bad_alloc& e) {
         std::cerr << "错误: 分配内存失败 (需要 " << num_elements * sizeof(T) << " 字节): " << e.what() << std::endl;
         n = 0; d = 0;
         fin.close();
         return nullptr;
    }


    size_t bytes_to_read = num_elements * sizeof(T);
    fin.read(reinterpret_cast<char*>(data), bytes_to_read);

    if (static_cast<size_t>(fin.gcount()) != bytes_to_read) {
        std::cerr << "警告: 从 " << data_path << " 读取数据不足。预期 " << bytes_to_read << " 字节，实际读取 " << fin.gcount() << " 字节。" << std::endl;
        // 可以选择是否继续，或者认为这是一个错误
        delete[] data;
        n = 0; d = 0;
        fin.close();
        return nullptr;
    }

    fin.close();
    std::cerr << "加载数据 " << data_path << " (数量=" << n << ", 维度=" << d << ")" << std::endl;
    return data;
}