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
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cublas_v2.h>
#include <algorithm>
#include "flat_scan.h"
// 可以自行添加需要的头文件

// CUDA错误检查宏
#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << " - " << cudaGetErrorString(error) << std::endl; \
            exit(1); \
        } \
    } while(0)

// cuBLAS错误检查宏
#define CUBLAS_CHECK(call) \
    do { \
        cublasStatus_t status = call; \
        if (status != CUBLAS_STATUS_SUCCESS) { \
            std::cerr << "cuBLAS error at " << __FILE__ << ":" << __LINE__ << std::endl; \
            exit(1); \
        } \
    } while(0)

// GPU核函数：将内积转换为距离
__global__ void convert_to_distance_kernel(float* distances, int total) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total) {
        distances[idx] = 1.0f - distances[idx];
    }
}

// GPU核函数：从距离矩阵中找出每个查询的top-k最近邻
__global__ void find_topk_kernel(float* distances, int* results, 
                                 int n, int m, int k) {
    int query_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (query_idx < m) {
        // 使用简单的选择排序找到top-k
        for (int i = 0; i < k; i++) {
            int min_idx = -1;
            float min_dist = 1e10f;
            
            // 找到当前最小的未选择的距离
            for (int j = 0; j < n; j++) {
                float dist = distances[j * m + query_idx];
                bool already_selected = false;
                
                // 检查是否已经被选择
                for (int l = 0; l < i; l++) {
                    if (results[query_idx * k + l] == j) {
                        already_selected = true;
                        break;
                    }
                }
                
                if (!already_selected && dist < min_dist) {
                    min_dist = dist;
                    min_idx = j;
                }
            }
            
            if (min_idx != -1) {
                results[query_idx * k + i] = min_idx;
            }
        }
    }
}

// 改进的GPU核函数：使用更高效的top-k选择算法
__global__ void find_topk_kernel_improved(float* distances, int* results, 
                                         int n, int m, int k) {
    int query_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (query_idx < m) {
        // 为每个线程分配局部内存来存储候选结果
        float topk_distances[10]; // 假设k<=10，避免动态分配
        int topk_indices[10];
        
        // 初始化为最大值
        for (int i = 0; i < k; i++) {
            topk_distances[i] = 1e10f;
            topk_indices[i] = -1;
        }
        
        // 遍历所有数据点
        for (int i = 0; i < n; i++) {
            float dist = distances[i * m + query_idx];
            
            // 检查是否应该插入到top-k中
            if (dist < topk_distances[k-1]) {
                // 找到插入位置
                int insert_pos = k-1;
                for (int j = 0; j < k-1; j++) {
                    if (dist < topk_distances[j]) {
                        insert_pos = j;
                        break;
                    }
                }
                
                // 向后移动元素
                for (int j = k-1; j > insert_pos; j--) {
                    topk_distances[j] = topk_distances[j-1];
                    topk_indices[j] = topk_indices[j-1];
                }
                
                // 插入新元素
                topk_distances[insert_pos] = dist;
                topk_indices[insert_pos] = i;
            }
        }
        
        // 将结果写入全局内存
        for (int i = 0; i < k; i++) {
            results[query_idx * k + i] = topk_indices[i];
        }
    }
}

// 简化版GPU核函数：用于调试
__global__ void find_topk_simple(float* distances, int* results, 
                                 int n, int m, int k) {
    int query_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (query_idx < m) {
        // 简单的气泡排序找最小的k个
        for (int i = 0; i < k; i++) {
            int min_idx = 0;
            float min_dist = distances[0 * m + query_idx];
            
            // 找到未被选择的最小距离
            for (int j = 1; j < n; j++) {
                float dist = distances[j * m + query_idx];
                bool selected = false;
                
                // 检查j是否已被选择
                for (int prev = 0; prev < i; prev++) {
                    if (results[query_idx * k + prev] == j) {
                        selected = true;
                        break;
                    }
                }
                
                if (!selected && dist < min_dist) {
                    min_dist = dist;
                    min_idx = j;
                }
            }
            
            results[query_idx * k + i] = min_idx;
        }
    }
}

// GPU版本的批量ANNS搜索
std::vector<std::vector<int>> gpu_batch_search(float* base, float* queries, 
                                               size_t base_number, size_t vecdim, 
                                               size_t query_number, size_t k) {
    // 创建cuBLAS句柄
    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));
    
    // 分配GPU内存
    float *d_base, *d_queries, *d_distances;
    int *d_results;
    
    size_t base_size = base_number * vecdim * sizeof(float);
    size_t query_size = query_number * vecdim * sizeof(float);
    size_t distance_size = base_number * query_number * sizeof(float);
    size_t result_size = query_number * k * sizeof(int);
    
    CUDA_CHECK(cudaMalloc(&d_base, base_size));
    CUDA_CHECK(cudaMalloc(&d_queries, query_size));
    CUDA_CHECK(cudaMalloc(&d_distances, distance_size));
    CUDA_CHECK(cudaMalloc(&d_results, result_size));
    
    // 复制数据到GPU
    CUDA_CHECK(cudaMemcpy(d_base, base, base_size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_queries, queries, query_size, cudaMemcpyHostToDevice));
    
    // 使用cuBLAS进行矩阵乘法：base (n×d) × queries^T (d×m) = distances (n×m)
    const float alpha = 1.0f, beta = 0.0f;
    
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));
    
    // 执行矩阵乘法计算内积
    // By swapping A and B, and transposing them, we get the result in row-major order
    // C(m,n) = A(k,m)^T * B(k,n)
    CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N,
                            query_number, base_number, vecdim,
                            &alpha,
                            d_queries, vecdim,
                            d_base, vecdim,
                            &beta,
                            d_distances, query_number));
    
    // 将内积转换为距离：distance = 1 - inner_product
    int total_elements = base_number * query_number;
    int threads_per_block = 256;
    int blocks = (total_elements + threads_per_block - 1) / threads_per_block;
    
    convert_to_distance_kernel<<<blocks, threads_per_block>>>(d_distances, total_elements);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // 找到每个查询的top-k最近邻
    int topk_threads = 256;
    int topk_blocks = (query_number + topk_threads - 1) / topk_threads;
    find_topk_kernel_improved<<<topk_blocks, topk_threads>>>(d_distances, d_results,
                                                     base_number, query_number, k);
    
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    
    float gpu_time;
    CUDA_CHECK(cudaEventElapsedTime(&gpu_time, start, stop));
    std::cout << "GPU batch processing time: " << gpu_time << " ms" << std::endl;
    
    // 复制结果回CPU
    std::vector<int> h_results(query_number * k);
    CUDA_CHECK(cudaMemcpy(h_results.data(), d_results, result_size, cudaMemcpyDeviceToHost));
    
    // 整理结果格式
    std::vector<std::vector<int>> results(query_number);
    for (int i = 0; i < query_number; i++) {
        results[i].resize(k);
        for (int j = 0; j < k; j++) {
            results[i][j] = h_results[i * k + j];
        }
    }
    
    // 清理资源
    CUDA_CHECK(cudaFree(d_base));
    CUDA_CHECK(cudaFree(d_queries));
    CUDA_CHECK(cudaFree(d_distances));
    CUDA_CHECK(cudaFree(d_results));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUBLAS_CHECK(cublasDestroy(handle));
    
    return results;
}

// 添加一个调试版本的GPU搜索函数
std::vector<std::vector<int>> gpu_batch_search_debug(float* base, float* queries, 
                                                     size_t base_number, size_t vecdim, 
                                                     size_t query_number, size_t k) {
    // 首先测试单个查询以验证算法正确性
    if (query_number == 1) {
        std::cout << "Debug: Single query test" << std::endl;
        
        // 在CPU上验证前几个距离计算
        float first_distances[5];
        for (int i = 0; i < 5; i++) {
            float dist = 0;
            for (int d = 0; d < vecdim; d++) {
                dist += base[i * vecdim + d] * queries[d];
            }
            first_distances[i] = 1.0f - dist;
            std::cout << "Base[" << i << "] distance: " << first_distances[i] << std::endl;
        }
    }
    
    // 创建cuBLAS句柄
    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));
    
    // 分配GPU内存
    float *d_base, *d_queries, *d_distances;
    int *d_results;
    
    size_t base_size = base_number * vecdim * sizeof(float);
    size_t query_size = query_number * vecdim * sizeof(float);
    size_t distance_size = base_number * query_number * sizeof(float);
    size_t result_size = query_number * k * sizeof(int);
    
    CUDA_CHECK(cudaMalloc(&d_base, base_size));
    CUDA_CHECK(cudaMalloc(&d_queries, query_size));
    CUDA_CHECK(cudaMalloc(&d_distances, distance_size));
    CUDA_CHECK(cudaMalloc(&d_results, result_size));
    
    // 初始化结果数组
    CUDA_CHECK(cudaMemset(d_results, -1, result_size));
    
    // 复制数据到GPU
    CUDA_CHECK(cudaMemcpy(d_base, base, base_size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_queries, queries, query_size, cudaMemcpyHostToDevice));
    
    // 使用cuBLAS进行矩阵乘法
    const float alpha = 1.0f, beta = 0.0f;
    
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));
    
    // 执行矩阵乘法计算内积
    // By swapping A and B, and transposing them, we get the result in row-major order
    // C(m,n) = A(k,m)^T * B(k,n)
    CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N,
                            query_number, base_number, vecdim,
                            &alpha,
                            d_queries, vecdim,
                            d_base, vecdim,
                            &beta,
                            d_distances, query_number));
    
    // 将内积转换为距离
    int total_elements = base_number * query_number;
    int threads_per_block = 256;
    int blocks = (total_elements + threads_per_block - 1) / threads_per_block;
    
    convert_to_distance_kernel<<<blocks, threads_per_block>>>(d_distances, total_elements);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // 如果是单查询调试，检查GPU计算的距离
    if (query_number == 1) {
        float* h_distances = new float[base_number];
        CUDA_CHECK(cudaMemcpy(h_distances, d_distances, base_number * sizeof(float), cudaMemcpyDeviceToHost));
        
        std::cout << "GPU computed distances (first 5):" << std::endl;
        for (int i = 0; i < 5; i++) {
            std::cout << "GPU Base[" << i << "] distance: " << h_distances[i] << std::endl;
        }
        delete[] h_distances;
    }
    
    // 找到每个查询的top-k最近邻
    int topk_threads = min(256, (int)query_number);
    int topk_blocks = (query_number + topk_threads - 1) / topk_threads;
    find_topk_simple<<<topk_blocks, topk_threads>>>(d_distances, d_results,
                                                    base_number, query_number, k);
    
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    
    float gpu_time;
    CUDA_CHECK(cudaEventElapsedTime(&gpu_time, start, stop));
    std::cout << "GPU batch processing time: " << gpu_time << " ms" << std::endl;
    
    // 复制结果回CPU
    std::vector<int> h_results(query_number * k);
    CUDA_CHECK(cudaMemcpy(h_results.data(), d_results, result_size, cudaMemcpyDeviceToHost));
    
    // 调试：打印第一个查询的结果
    if (query_number >= 1) {
        std::cout << "First query GPU results: ";
        for (int i = 0; i < k; i++) {
            std::cout << h_results[i] << " ";
        }
        std::cout << std::endl;
    }
    
    // 整理结果格式
    std::vector<std::vector<int>> results(query_number);
    for (int i = 0; i < query_number; i++) {
        results[i].resize(k);
        for (int j = 0; j < k; j++) {
            results[i][j] = h_results[i * k + j];
        }
    }
    
    // 清理资源
    CUDA_CHECK(cudaFree(d_base));
    CUDA_CHECK(cudaFree(d_queries));
    CUDA_CHECK(cudaFree(d_distances));
    CUDA_CHECK(cudaFree(d_results));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUBLAS_CHECK(cublasDestroy(handle));
    
    return results;
}

// 简化的调试版GPU函数
std::vector<std::vector<int>> gpu_batch_search_simple(float* base, float* queries, 
                                                     size_t base_number, size_t vecdim, 
                                                     size_t query_number, size_t k) {
    // 创建cuBLAS句柄
    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));
    
    // 分配GPU内存
    float *d_base, *d_queries, *d_distances;
    int *d_results;
    
    size_t base_size = base_number * vecdim * sizeof(float);
    size_t query_size = query_number * vecdim * sizeof(float);
    size_t distance_size = base_number * query_number * sizeof(float);
    size_t result_size = query_number * k * sizeof(int);
    
    CUDA_CHECK(cudaMalloc(&d_base, base_size));
    CUDA_CHECK(cudaMalloc(&d_queries, query_size));
    CUDA_CHECK(cudaMalloc(&d_distances, distance_size));
    CUDA_CHECK(cudaMalloc(&d_results, result_size));
    
    // 初始化结果数组
    CUDA_CHECK(cudaMemset(d_results, 0, result_size));
    
    // 复制数据到GPU
    CUDA_CHECK(cudaMemcpy(d_base, base, base_size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_queries, queries, query_size, cudaMemcpyHostToDevice));
    
    const float alpha = 1.0f, beta = 0.0f;
    
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));
    
    // 执行矩阵乘法计算内积
    // By swapping A and B, and transposing them, we get the result in row-major order
    // C(m,n) = A(k,m)^T * B(k,n)
    CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N,
                            query_number, base_number, vecdim,
                            &alpha,
                            d_queries, vecdim,
                            d_base, vecdim,
                            &beta,
                            d_distances, query_number));
    
    // 将内积转换为距离
    int total_elements = base_number * query_number;
    int threads_per_block = 256;
    int blocks = (total_elements + threads_per_block - 1) / threads_per_block;
    
    convert_to_distance_kernel<<<blocks, threads_per_block>>>(d_distances, total_elements);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // 找到每个查询的top-k最近邻
    int topk_threads = 256;
    int topk_blocks = (query_number + topk_threads - 1) / topk_threads;
    find_topk_simple<<<topk_blocks, topk_threads>>>(d_distances, d_results,
                                                    base_number, query_number, k);
    
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    
    float gpu_time;
    CUDA_CHECK(cudaEventElapsedTime(&gpu_time, start, stop));
    std::cout << "GPU batch processing time: " << gpu_time << " ms" << std::endl;
    
    // 复制结果回CPU
    std::vector<int> h_results(query_number * k);
    CUDA_CHECK(cudaMemcpy(h_results.data(), d_results, result_size, cudaMemcpyDeviceToHost));
    
    // 整理结果格式
    std::vector<std::vector<int>> results(query_number);
    for (int i = 0; i < query_number; i++) {
        results[i].resize(k);
        for (int j = 0; j < k; j++) {
            results[i][j] = h_results[i * k + j];
        }
    }
    
    // 清理资源
    CUDA_CHECK(cudaFree(d_base));
    CUDA_CHECK(cudaFree(d_queries));
    CUDA_CHECK(cudaFree(d_distances));
    CUDA_CHECK(cudaFree(d_results));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUBLAS_CHECK(cublasDestroy(handle));
    
    return results;
}

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


int main(int argc, char *argv[])
{
    // 查询向量的数量和向量的总数
    size_t test_number = 0, base_number = 0;
    // ground truth的维度和查询向量的维度
    size_t test_gt_d = 0, vecdim = 0;

    //std::string data_path = "/anndata/"; 
    auto test_query = LoadData<float>("DEEP100K.query.fbin", test_number, vecdim);
    auto test_gt = LoadData<int>("DEEP100K.gt.query.100k.top100.bin", test_number, test_gt_d);
    auto base = LoadData<float>("DEEP100K.base.100k.fbin", base_number, vecdim);
    
    // 只测试前2000条查询，避免测试时间过长
    test_number = 2000;
    // 查询时返回的最邻近向量的数量
    const size_t k = 10;
    
    std::cout << "=== CPU 暴力搜索测试 (逐个查询) ===" << std::endl;
    
    // 存储每次查询的召回率和延迟
    std::vector<SearchResult> cpu_results;
    cpu_results.resize(test_number);
    
    // CPU暴力搜索测试代码，遍历查询向量
    // 对每个查询向量执行搜索，并计算召回率和延迟。
    auto cpu_start_time = std::chrono::high_resolution_clock::now();
    
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
        cpu_results[i] = {recall, diff};
    }
    
    auto cpu_end_time = std::chrono::high_resolution_clock::now();
    auto cpu_total_time = std::chrono::duration_cast<std::chrono::milliseconds>(cpu_end_time - cpu_start_time);
    
    // 计算CPU平均召回率和平均延迟
    float cpu_avg_recall = 0, cpu_avg_latency = 0;
    for(int i = 0; i < test_number; ++i) {
        cpu_avg_recall += cpu_results[i].recall;
        cpu_avg_latency += cpu_results[i].latency;
    }

    // 浮点误差可能导致一些精确算法平均recall不是1
    std::cout << "CPU average recall: " << cpu_avg_recall / test_number << std::endl;
    std::cout << "CPU average latency (us): " << cpu_avg_latency / test_number << std::endl;
    std::cout << "CPU total time: " << cpu_total_time.count() << " ms" << std::endl;
    
    std::cout << "\n=== GPU 批量搜索测试 ===" << std::endl;
    
    // GPU批量测试 - 不同的batch大小
    std::vector<int> batch_sizes = {50, 100, 200, 500, 1000, 2000};
    
    for (int batch_size : batch_sizes) {
        if (batch_size > test_number) continue;
        
        std::cout << "\n--- Batch Size: " << batch_size << " ---" << std::endl;
        
        auto gpu_start_time = std::chrono::high_resolution_clock::now();
        
        // 执行GPU批量搜索 - 使用简化版本进行调试
        auto gpu_results = gpu_batch_search_simple(base, test_query, base_number, vecdim, batch_size, k);
        
        auto gpu_end_time = std::chrono::high_resolution_clock::now();
        auto gpu_total_time = std::chrono::duration_cast<std::chrono::milliseconds>(gpu_end_time - gpu_start_time);
        
        // 计算GPU召回率
        float gpu_avg_recall = 0;
        for (int i = 0; i < batch_size; i++) {
            std::set<uint32_t> gtset;
            for(int j = 0; j < k; ++j){
                int t = test_gt[j + i*test_gt_d];
                gtset.insert(t);
            }
            
            int correct = 0;
            for (int j = 0; j < k; j++) {
                if (gtset.find(gpu_results[i][j]) != gtset.end()) {
                    correct++;
                }
            }
            gpu_avg_recall += (float)correct / k;
        }
        gpu_avg_recall /= batch_size;
        
        std::cout << "GPU average recall: " << gpu_avg_recall << std::endl;
        std::cout << "GPU total time: " << gpu_total_time.count() << " ms" << std::endl;
        std::cout << "GPU throughput: " << (float)batch_size / gpu_total_time.count() * 1000 << " queries/second" << std::endl;
        
        // 计算加速比
        float cpu_time_for_batch = (cpu_total_time.count() * 1000) / test_number;
        std::cout << "Speedup: " << cpu_time_for_batch / gpu_total_time.count() << "x" << std::endl;
    }
    
    // 释放分配的内存
    delete[] test_query;
    delete[] test_gt;
    delete[] base;

    return 0;
}