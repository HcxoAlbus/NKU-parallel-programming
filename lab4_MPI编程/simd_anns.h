#pragma once // 防止头文件重复包含
#include <queue>
#include <arm_neon.h>  // 引入ARM NEON SIMD指令集
#include <cstdint>   // 引入stdint.h头文件，提供固定宽度整数类型

// SIMD工具类，处理8个float32数据
struct simd8float32 {
    float32x4x2_t data;  // 2个128位向量，共8个float

    simd8float32() = default; // 默认构造函数

    // 从x指向的地址开始，加载8个float32数据
    // 这里使用了NEON的vld1q_f32函数来加载数据
    explicit simd8float32(const float* x)
        : data{vld1q_f32(x), vld1q_f32(x + 4)} {}

    // 元素乘法
    // 使用NEON的vmulq_f32函数来进行向量乘法
    simd8float32 operator*(const simd8float32& other) const {
        simd8float32 result;
        result.data.val[0] = vmulq_f32(this->data.val[0], other.data.val[0]);
        result.data.val[1] = vmulq_f32(this->data.val[1], other.data.val[1]);
        return result;
    }

    // 元素加法
    // 使用NEON的vaddq_f32函数来进行向量加法
    simd8float32 operator+(const simd8float32& other) const {
        simd8float32 result;
        result.data.val[0] = vaddq_f32(this->data.val[0], other.data.val[0]);
        result.data.val[1] = vaddq_f32(this->data.val[1], other.data.val[1]);
        return result;
    }

    // 构造函数：从两个向量构建
    simd8float32(float32x4_t a, float32x4_t b) {
        data.val[0] = a;
        data.val[1] = b;
    }

    // 将8个float元素加总为一个float，方便内积计算时求和
    /*例如，
    data.val[0] = {1.0, 2.0, 3.0, 4.0};
    data.val[1] = {5.0, 6.0, 7.0, 8.0};
    sum_vec = {1.0+5.0, 2.0+6.0, 3.0+7.0, 4.0+8.0} = {6.0, 8.0, 10.0, 12.0}; // 先垂直相加
    sum_low = {6.0, 8.0}; sum_high = {10.0, 12.0}; // 分离高低部分
    sum_low = {6.0+10.0, 8.0+12.0} = {16.0, 20.0}; // 配对相加
    sum_low = {16.0+20.0, 16.0+20.0} = {36.0, 36.0}; // 再次配对相加
    */
    float horizontal_sum() const {
        float32x4_t sum_vec = vaddq_f32(data.val[0], data.val[1]); // 先加两个向量
        float32x2_t sum_low = vget_low_f32(sum_vec); // 提取和的低64位
        float32x2_t sum_high = vget_high_f32(sum_vec); // 提取和的高64位
        sum_low = vadd_f32(sum_low, sum_high); // 将低64位和高64位相加
        sum_low = vpadd_f32(sum_low, sum_low); // 对低64位的相邻元素进行水平加法
        // 返回sum_low的第一个元素，即所有8个float元素的和
        return vget_lane_f32(sum_low, 0);
    }
};

// SIMD优化的内积距离计算
// x: 第一个向量的指针
// y: 第二个向量的指针
// dim: 向量维度
// 返回: 内积距离 (1 - 内积)
inline float inner_product_distance_simd(const float* x, const float* y, size_t dim) {
    float dot = 0.0f; // 初始化内积结果
    size_t i = 0;
    
    // 使用SIMD处理每8个元素
    for (; i + 7 < dim; i += 8) {
        simd8float32 a(&x[i]); // 加载x的8个元素
        simd8float32 b(&y[i]); // 加载y的8个元素
        simd8float32 product = a * b; // 计算元素乘积
        dot += product.horizontal_sum(); // 将乘积结果水平加和并累加到dot
    }
    
    // 处理剩余元素 (如果维度不是8的倍数)
    for (; i < dim; i++) {
        dot += x[i] * y[i]; // 标量计算
    }
    
    return 1.0f - dot; // 返回内积距离
}

// SIMD优化的ANNS搜索
// base: 指向基向量数据的指针 (N * D)
// query: 指向查询向量数据的指针 (D)
// base_number: 基向量的数量 (N)
// vecdim: 向量维度 (D)
// k: 要查找的最近邻数量
// 返回: 一个包含 <距离, 索引> 对的优先队列 (最大堆)
std::priority_queue<std::pair<float, uint32_t>> simd_search(
    float* base, const float* query, size_t base_number, size_t vecdim, size_t k) {
    
    // 创建一个最大堆，用于存储 <距离, 索引> 对
    std::priority_queue<std::pair<float, uint32_t>> q;

    // 遍历所有基向量
    for (size_t i = 0; i < base_number; ++i) {
        // 使用SIMD计算内积距离
        float dis = inner_product_distance_simd(base + i * vecdim, query, vecdim);

        // 如果堆的大小小于 k，直接将当前向量加入堆中
        if (q.size() < k) {
            q.push({dis, static_cast<uint32_t>(i)});
        } else {
            // 如果当前向量的距离小于堆顶元素 (即当前 k 个最近邻中最远的那个)
            if (dis < q.top().first) {
                q.push({dis, static_cast<uint32_t>(i)}); // 将当前向量加入堆中
                q.pop();          // 移除原堆顶的最远向量
            }
        }
    }
    return q; // 返回包含 top-k 结果的堆
}