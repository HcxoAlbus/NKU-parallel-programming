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

// --- 函数声明  ---
std::priority_queue<std::pair<float, uint32_t>> flat_search(float* base, const float* query, size_t base_number, size_t vecdim, size_t k);
std::priority_queue<std::pair<float, uint32_t>> simd_search(float* base, const float* query, size_t base_number, size_t vecdim, size_t k);
// ---
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


int main(int argc, char *argv[])
{
    size_t test_number = 0, base_number = 0;
    size_t test_gt_d = 0, vecdim = 0;
    
    std::string path = "/anndata/";
    std::string query_path =   "DEEP100K.query.fbin";
    std::string gt_path =   "DEEP100K.gt.query.100k.top100.bin";
    std::string base_path =   "DEEP100K.base.100k.fbin";

    auto test_query = LoadData<float>(query_path, test_number, vecdim);
    auto test_gt = LoadData<int>(gt_path, test_number, test_gt_d);
    auto base = LoadData<float>(base_path, base_number, vecdim);

    size_t num_queries_to_test = 2000;
    std::cout << "将测试前 " << num_queries_to_test << " 条查询。" << std::endl;


    const size_t k = 10;

    // --- 使用 Lambda 表达式进行绑定 ---

    // --- Flat Search ---
    // 捕获 base, base_number, vecdim
    auto flat_search_lambda = [&](const float* q, size_t k_param) {
        return flat_search(base, q, base_number, vecdim, k_param);
    };
    std::vector<SearchResult> results_flat = benchmark_search(
       flat_search_lambda, // 传递 lambda
       test_query, test_gt, base_number, vecdim, num_queries_to_test, test_gt_d, k);


    // --- SIMD Search ---
    auto simd_search_lambda = [&](const float* q, size_t k_param) {
        return simd_search(base, q, base_number, vecdim, k_param);
    };
    std::vector<SearchResult> results_simd = benchmark_search(
       simd_search_lambda,
       test_query, test_gt, base_number, vecdim, num_queries_to_test, test_gt_d, k);

    // --- PQ Search ---
    size_t nsub = 4;
    double train_ratio = 1.0;
    size_t rerank_k = 600;

    ProductQuantizer* pq_index_ptr = nullptr; // 使用指针以便在失败时保持 nullptr
    
    pq_index_ptr = new ProductQuantizer(base, base_number, vecdim, nsub, train_ratio);

    std::vector<SearchResult> results_pq;
    if (pq_index_ptr) { // 只有在索引成功创建时才进行测试
        std::cout << "使用 PQ 参数: nsub=" << nsub << ", train_ratio=" << train_ratio << ", rerank_k=" << rerank_k << std::endl;
        auto pq_search_lambda = [&](const float* q, size_t k_param) {
            // 使用指针访问成员函数
            return pq_index_ptr->search(q, base, k_param, rerank_k);
        };
        results_pq = benchmark_search(
           pq_search_lambda,
           test_query, test_gt, base_number, vecdim, num_queries_to_test, test_gt_d, k);
    } else {
        std::cerr << "跳过 PQ 搜索测试，因为索引创建失败。" << std::endl;
        results_pq.resize(num_queries_to_test, {0.0f, 0}); // 填充默认失败结果
    }


    // --- SQ Search ---
    ScalarQuantizer* sq_quantizer_ptr = nullptr;

    sq_quantizer_ptr = new ScalarQuantizer(base, base_number, vecdim);
    
    std::vector<SearchResult> results_sq;
    if (sq_quantizer_ptr) {
        auto sq_search_lambda = [&](const float* q, size_t k_param) {
            return sq_quantizer_ptr->sq_search(q, k_param);
        };
        results_sq = benchmark_search(
           sq_search_lambda,
           test_query, test_gt, base_number, vecdim, num_queries_to_test, test_gt_d, k);
    } else {
         std::cerr << "跳过 SQ 搜索测试，因为量化器创建失败。" << std::endl;
         results_sq.resize(num_queries_to_test, {0.0f, 0}); // 填充默认失败结果
    }


    // --- 打印结果 ---
    print_results("Flat Search (暴力搜索)", results_flat, num_queries_to_test);
    print_results("SIMD Search (SIMD优化)", results_simd, num_queries_to_test);
    if (pq_index_ptr) { // 仅当 PQ 测试运行时才打印结果
       print_results("PQ Search (乘积量化, rerank=" + std::to_string(rerank_k) + ")", results_pq, num_queries_to_test);
    } else {
       print_results("PQ Search (跳过)", results_pq, num_queries_to_test);
    }
    if (sq_quantizer_ptr) { // 仅当 SQ 测试运行时才打印结果
        print_results("SQ Search (标量量化)", results_sq, num_queries_to_test);
    } else {
        print_results("SQ Search (跳过)", results_sq, num_queries_to_test);
    }


    // --- 清理 ---
    delete[] test_query;
    delete[] test_gt;
    delete[] base;
    delete pq_index_ptr; 
    delete sq_quantizer_ptr;

    return 0;
}