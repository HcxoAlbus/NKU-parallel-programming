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


int main(int argc, char *argv[])
{
    // 查询向量的数量和向量的总数
    size_t test_number = 0, base_number = 0;
    // ground truth的维度和查询向量的维度
    size_t test_gt_d = 0, vecdim = 0;

    std::string data_path = "/anndata/"; 
    auto test_query = LoadData<float>(data_path + "DEEP100K.query.fbin", test_number, vecdim);
    auto test_gt = LoadData<int>(data_path + "DEEP100K.gt.query.100k.top100.bin", test_number, test_gt_d);
    auto base = LoadData<float>(data_path + "DEEP100K.base.100k.fbin", base_number, vecdim);
    // 只测试前2000条查询，避免测试时间过长
    test_number = 2000;
    // 查询时返回的最邻近向量的数量
    const size_t k = 10;
    // 存储每次查询的召回率和延迟
    std::vector<SearchResult> results;
    results.resize(test_number);

    // 如果你需要保存索引，可以在这里添加你需要的函数，你可以将下面的注释删除来查看pbs是否将build.index返回到你的files目录中
    // 要保存的目录必须是files/*
    // 每个人的目录空间有限，不需要的索引请及时删除，避免占空间太大
    // 不建议在正式测试查询时同时构建索引，否则性能波动会较大
    // 下面是一个构建hnsw索引的示例
    // build_index(base, base_number, vecdim);

    
    // 查询测试代码，遍历查询向量
    // 对每个查询向量执行搜索，并计算召回率和延迟。
    for(int i = 0; i < test_number; ++i) {
        // 秒与微秒的转换常量
        // 1秒 = 1000毫秒 = 1000 * 1000微秒
        const unsigned long Converter = 1000 * 1000;
        //是一个结构体，存储秒和微秒。
        //gettimeofday 获取当前时间
        struct timeval val;
        int ret = gettimeofday(&val, NULL);

        // 该文件已有代码中你只能修改该函数的调用方式
        // 可以任意修改函数名，函数参数或者改为调用成员函数，但是不能修改函数返回值。
        // 对第 i 个查询向量执行搜索，返回一个优先队列 res，存储最近邻的结果。

        auto res = flat_search(base, test_query + i*vecdim, base_number, vecdim, k);

        // 通过获取当前时间的秒和微秒来计算延迟
        struct timeval newVal;
        ret = gettimeofday(&newVal, NULL);
        int64_t diff = (newVal.tv_sec * Converter + newVal.tv_usec) - (val.tv_sec * Converter + val.tv_usec);

        // 构建 ground truth 的集合，用于后续计算召回率。
        std::set<uint32_t> gtset;
        for(int j = 0; j < k; ++j){
            int t = test_gt[j + i*test_gt_d];
            gtset.insert(t);
        }

        // 遍历搜索结果 res，检查是否在 ground truth 集合中。
        // 计算召回率 recall。
        size_t acc = 0;
        while (res.size()) {   
            int x = res.top().second;
            if(gtset.find(x) != gtset.end()){
                ++acc;
            }
            res.pop();
        }
        float recall = (float)acc/k;

        // 将召回率和延迟存储到 results 容器中。
        results[i] = {recall, diff};
    }
    // 计算平均召回率和平均延迟
    float avg_recall = 0, avg_latency = 0;
    for(int i = 0; i < test_number; ++i) {
        avg_recall += results[i].recall;
        avg_latency += results[i].latency;
    }

    // 浮点误差可能导致一些精确算法平均recall不是1
    std::cout << "average recall: "<<avg_recall / test_number<<"\n";
    std::cout << "average latency (us): "<<avg_latency / test_number<<"\n";
    return 0;
}
