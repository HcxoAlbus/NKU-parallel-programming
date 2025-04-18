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
#include "hnswlib/hnswlib/hnswlib.h"
#include "flat_scan.h"
#include "simd_anns.h"
#include "pq_anns.h"
#include "sq_anns.h"
#include <functional>
// #include "fastscan_pq_anns.h"
// 可以自行添加需要的头文件

using namespace hnswlib;

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
    float recall; // 召回率，表示搜索结果中正确的最近邻占比。
    // 召回率 = (查询结果中正确的个数) / k
    int64_t latency; // 查询延迟，单位为us
};

void build_index(float* base, size_t base_number, size_t vecdim)
{
    // 控制索引构建时的搜索范围，值越大，索引质量越高，但构建时间也会增加。
    const int efConstruction = 150; // 为防止索引构建时间过长，efc建议设置200以下
    // 控制每个节点的最大邻居数，值越大，图的连通性越强，但内存占用也会增加。
    const int M = 16; // M建议设置为16以下

    // 创建HNSW对象
    HierarchicalNSW<float> *appr_alg;
    // 创建 InnerProductSpace 对象，表示向量空间，使用内积作为相似性度量。
    InnerProductSpace ipspace(vecdim);
    appr_alg = new HierarchicalNSW<float>(&ipspace, base_number, M, efConstruction);
    // 添加第一个点，指定向量ID为0
    // 第一个点通常需要单独添加，因为它是索引的起点
    appr_alg->addPoint(base, 0);
    // 使用OpenMP并行化循环
    #pragma omp parallel for
    // 并行地将数据集中的剩余的所有向量添加到HNSW索引中。
    for(int i = 1; i < base_number; ++i) {
        appr_alg->addPoint(base + 1ll*vecdim*i, i);//1ll：将int转换为long long，避免溢出
    }
    // 将构建好的HNSW索引保存到磁盘，以便后续加载和使用。
    char path_index[1024] = "files/hnsw.index";
    appr_alg->saveIndex(path_index);
}
// 封装查询过程的函数模板
template<typename SearchFunc>
std::vector<SearchResult> benchmark_search(
    SearchFunc search_func,
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
    
    // 查询测试代码，遍历查询向量
    for(int i = 0; i < test_number; ++i) {
        // 秒与微秒的转换常量
        const unsigned long Converter = 1000 * 1000;
        struct timeval val;
        gettimeofday(&val, NULL);

        // 调用传入的搜索函数
        auto res = search_func(base, test_query + i*vecdim, base_number, vecdim, k);

        // 计算延迟
        struct timeval newVal;
        gettimeofday(&newVal, NULL);
        int64_t diff = (newVal.tv_sec * Converter + newVal.tv_usec) - (val.tv_sec * Converter + val.tv_usec);

        // 构建 ground truth 的集合
        std::set<uint32_t> gtset;
        for(int j = 0; j < k; ++j){
            int t = test_gt[j + i*test_gt_d];
            gtset.insert(t);
        }

        // 计算召回率
        size_t acc = 0;
        while (res.size()) {   
            int x = res.top().second;
            if(gtset.find(x) != gtset.end()){
                ++acc;
            }
            res.pop();
        }
        float recall = (float)acc/k;
        
        // 保存结果
        results[i] = {recall, diff};
    }
    
    return results;
}

// 打印测试结果的辅助函数
void print_results(const std::string& method_name, const std::vector<SearchResult>& results, size_t test_number) {
    float avg_recall = 0, avg_latency = 0;
    for(int i = 0; i < test_number; ++i) {
        avg_recall += results[i].recall;
        avg_latency += results[i].latency;
    }
    
    std::cout << "=== " << method_name << " ===" << std::endl;
    std::cout << "Average recall: " << avg_recall / test_number << std::endl;
    std::cout << "Average latency (us): " << avg_latency / test_number << std::endl;
    std::cout << std::endl;
}
int main(int argc, char *argv[])
{
    // 查询向量的数量和向量的总数
    size_t test_number = 0, base_number = 0;
    // ground truth的维度和查询向量的维度
    size_t test_gt_d = 0, vecdim = 0;

    std::string data_path = "/data"; 
    auto test_query = LoadData<float>("DEEP100K.query.fbin", test_number, vecdim);
    auto test_gt = LoadData<int>("DEEP100K.gt.query.100k.top100.bin", test_number, test_gt_d);
    auto base = LoadData<float>("DEEP100K.base.100k.fbin", base_number, vecdim);
    // 只测试前2000条查询，避免测试时间过长
    test_number = 2000;
    // 查询时返回的最邻近向量的数量
    const size_t k = 10;
    // 测试不同的查询方法并保存结果
    
    // 1. 测试 flat_search
    //std::vector<SearchResult> results_flat = benchmark_search(
      //  flat_search, base, test_query, test_gt, base_number, vecdim, test_number, test_gt_d, k);
    
    // 2. 测试 simd_search
    //std::vector<SearchResult> results_simd = benchmark_search(
      //  simd_search, base, test_query, test_gt, base_number, vecdim, test_number, test_gt_d, k);
    
    // // 3. 测试 pq_search
     //std::vector<SearchResult> results_pq = benchmark_search(
       //  pq_search, base, test_query, test_gt, base_number, vecdim, test_number, test_gt_d, k);
    
    // 4. 测试 sq_search
    // Instantiate the ScalarQuantizer ONCE before benchmarking
    ScalarQuantizer quantizer(base, base_number, vecdim); // Changed variable name for clarity

    // Use std::bind to create an adapter for the member function.
    // benchmark_search passes 5 arguments: base, query, base_number, vecdim, k
    // We need to call quantizer.sq_search(query, k)
    // std::placeholders::_2 corresponds to the 'query' argument
    // std::placeholders::_5 corresponds to the 'k' argument
    auto sq_search_bound = std::bind(&ScalarQuantizer::sq_search, 
                                     &quantizer, // Pass the object instance
                                     std::placeholders::_2, // Map query
                                     std::placeholders::_5); // Map k

    std::vector<SearchResult> results_sq = benchmark_search(
       sq_search_bound, // Pass the bound function object
       base, test_query, test_gt, base_number, vecdim, test_number, test_gt_d, k);
    
    // 5. 测试 fast_pq_search
    // std::vector<SearchResult> results_fast_pq = benchmark_search(
    //     fastscan_pq_search, base, test_query, test_gt, base_number, vecdim, test_number, test_gt_d, k);
    
    // 打印每种方法的测试结果
    //print_results("Flat Search", results_flat, test_number);
    //print_results("SIMD Search", results_simd, test_number);
    // print_results("PQ Search", results_pq, test_number);
    print_results("SQ Search", results_sq, test_number);
    // print_results("Fast PQ Search", results_fast_pq, test_number);
    
    return 0;

    // 如果你需要保存索引，可以在这里添加你需要的函数，你可以将下面的注释删除来查看pbs是否将build.index返回到你的files目录中
    // 要保存的目录必须是files/*
    // 每个人的目录空间有限，不需要的索引请及时删除，避免占空间太大
    // 不建议在正式测试查询时同时构建索引，否则性能波动会较大
    // 下面是一个构建hnsw索引的示例
    // build_index(base, base_number, vecdim);

    
}
