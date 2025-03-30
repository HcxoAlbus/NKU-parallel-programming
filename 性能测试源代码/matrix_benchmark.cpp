#include <iostream>
#include <vector>
// 用于高精度计时
#include <chrono>
// 用于格式化输出
#include <iomanip>
#include <cmath>
#include <random>
// 常见的算法库
#include <algorithm>
// 提供函数对象支持
#include <functional>
#include <string>
// 使用命名空间
using namespace std;
using namespace std::chrono;

// 定义矩阵最大大小
const int MAX_N = 4096;

// 使用高精度计时器，测试某一代码块的运行时间
class Timer {
private:
    high_resolution_clock::time_point start_time;
    // 记录计时操作的名称
    string operation_name;

public:
    // 初始化operation_name，并记录当前时间为start_time
    Timer(const string& name) : operation_name(name) {
        start_time = high_resolution_clock::now();
    }
    // 对象销毁时，计算到start_time的时间差，并输出
    ~Timer() {
        auto end_time = high_resolution_clock::now();
        duration<double> time_span = duration_cast<duration<double>>(end_time - start_time);
        cout << operation_name << ": " << fixed << setprecision(6) << time_span.count() << " 秒" << endl;
    }
    // 返回从start_time到当前时间的时间差
    double getElapsedTime() {
        auto end_time = high_resolution_clock::now();
        duration<double> time_span = duration_cast<duration<double>>(end_time - start_time);
        return time_span.count();
    }
};

class MatrixOperations {
private:
    int n; // 矩阵维度
    vector<double> a; // 向量a
    vector<vector<double>> b; // 矩阵b
    vector<double> sum1, sum2, sum3, sum4; // 存储不同算法结果的向量

    // 初始化数据（使用确定性模式以便结果验证）
    void initializeData() {
        // 使用C++标准库的随机数生成器类mt19937
        mt19937 rng(12345); // 固定种子，使得每次生成的随机数序列相同
        // 使用均匀分布生成器uniform_real_distribution
        uniform_real_distribution<double> dist(0.5, 1.5);

        // 直接指定向量a和矩阵b的值，便于判断算法正确性
        for (int i = 0; i < n; i++) {
            a[i] = 1.0 + (i % 10) / 10.0; // 产生固定模式：1.0, 1.1, 1.2, ..., 1.9, 1.0, ...
            for (int j = 0; j < n; j++) {
                b[i][j] = 0.5 + ((i+j) % 10) / 20.0; // 确定性值
            }
        }
    }

public:
    MatrixOperations(int size) : n(size) {
        // 初始化向量和矩阵
        a.resize(n);
        b.resize(n, vector<double>(n));
        sum1.resize(n);
        sum2.resize(n);
        sum3.resize(n);
        sum4.resize(n);

        // 初始化数据
        initializeData();
    }

    // 平凡算法：逐列访问矩阵元素
    void algorithm1() {
        for (int i = 0; i < n; i++) {
            sum1[i] = 0.0;
            for (int j = 0; j < n; j++) {
                sum1[i] += b[j][i] * a[j];
            }
        }
    }

    // 优化算法：逐行访问矩阵元素
    void algorithm2() {
        for (int i = 0; i < n; i++) {
            sum2[i] = 0.0;
        }
        for (int j = 0; j < n; j++) {
            for (int i = 0; i < n; i++) {
                sum2[i] += b[j][i] * a[j];
            }
        }
    }

    // 平凡算法的循环展开版本(展开4次)
    void algorithm1_unroll4() {
        for (int i = 0; i < n; i++) {
            sum3[i] = 0.0;
            int j = 0;
            for (; j <= n-4; j += 4) {
                sum3[i] += b[j][i] * a[j];
                sum3[i] += b[j+1][i] * a[j+1];
                sum3[i] += b[j+2][i] * a[j+2];
                sum3[i] += b[j+3][i] * a[j+3];
            }
            // 处理剩余元素
            for (; j < n; j++) {
                sum3[i] += b[j][i] * a[j];
            }
        }
    }

    // 优化算法的循环展开版本(展开4次)
    void algorithm2_unroll4() {
        for (int i = 0; i < n; i++) {
            sum4[i] = 0.0;
        }
        
        for (int j = 0; j < n; j++) {
            double a_val = a[j]; // 缓存a[j]值以减少内存访问
            int i = 0;
            for (; i <= n-4; i += 4) {
                sum4[i] += b[j][i] * a_val;
                sum4[i+1] += b[j][i+1] * a_val;
                sum4[i+2] += b[j][i+2] * a_val;
                sum4[i+3] += b[j][i+3] * a_val;
            }
            // 处理剩余元素
            for (; i < n; i++) {
                sum4[i] += b[j][i] * a_val;
            }
        }
    }

    // 验证结果是否正确
    bool verifyResults() {
        // 定义误差范围，避免浮点数比较时的精度问题
        const double epsilon = 1e-6;
        // 验证4个数组的所有元素的值在误差范围内是否相等
        for (int i = 0; i < n; i++) {
            if (fabs(sum1[i] - sum2[i]) > epsilon || 
                fabs(sum1[i] - sum3[i]) > epsilon || 
                fabs(sum1[i] - sum4[i]) > epsilon) {
                cout << "验证失败：位置 " << i << "的值不匹配" << endl;
                cout << "sum1[" << i << "] = " << sum1[i] << endl;
                cout << "sum2[" << i << "] = " << sum2[i] << endl;
                cout << "sum3[" << i << "] = " << sum3[i] << endl;
                cout << "sum4[" << i << "] = " << sum4[i] << endl;
                return false;
            }
        }
        return true;
    }

    // 运行性能测试，传入参数为测试次数（多次测试取平均值，缓解实验误差）
    void runPerformanceTest(int iterations) {
        // 大小为4的vector容器，存储4个算法的平均运行时间
        vector<double> times(4);

        cout << "\n矩阵大小: " << n << " x " << n << ", 测试迭代次数: " << iterations << endl;

        // 预热运行，避免程序第一次运行时的额外的开销
        algorithm1();
        algorithm2();
        algorithm1_unroll4();
        algorithm2_unroll4();

        // 测试算法1（平凡算法）
        cout << "\n测试算法1（平凡算法）..." << endl;
        for (int i = 0; i < iterations; i++) {
            Timer timer("运行 #" + to_string(i+1));
            algorithm1();
            times[0] += timer.getElapsedTime();
        }
        cout << "算法1平均执行时间: " << fixed << setprecision(6) << times[0] / iterations << " 秒" << endl;

        // 测试算法2（优化算法）
        cout << "\n测试算法2（优化算法）..." << endl;
        for (int i = 0; i < iterations; i++) {
            Timer timer("运行 #" + to_string(i+1));
            algorithm2();
            times[1] += timer.getElapsedTime();
        }
        cout << "算法2平均执行时间: " << fixed << setprecision(6) << times[1] / iterations << " 秒" << endl;

        // 测试算法1循环展开版本
        cout << "\n测试算法1（循环展开4次）..." << endl;
        for (int i = 0; i < iterations; i++) {
            Timer timer("运行 #" + to_string(i+1));
            algorithm1_unroll4();
            times[2] += timer.getElapsedTime();
        }
        cout << "算法1循环展开版本平均执行时间: " << fixed << setprecision(6) << times[2] / iterations << " 秒" << endl;

        // 测试算法2循环展开版本
        cout << "\n测试算法2（循环展开4次）..." << endl;
        for (int i = 0; i < iterations; i++) {
            Timer timer("运行 #" + to_string(i+1));
            algorithm2_unroll4();
            times[3] += timer.getElapsedTime();
        }
        cout << "算法2循环展开版本平均执行时间: " << fixed << setprecision(6) << times[3] / iterations << " 秒" << endl;

        // 验证结果
        cout << "\n验证计算结果..." << endl;
        if (verifyResults()) {
            cout << "验证通过: 所有算法结果匹配" << endl;
        }

        // 输出性能比较
        cout << "\n性能比较:" << endl;
        cout << "算法1 vs 算法2: 算法2快 " 
             << fixed << setprecision(2) << (times[0] / times[1]) << " 倍" << endl;
        cout << "算法1 vs 算法1循环展开: 循环展开快 " 
             << fixed << setprecision(2) << (times[0] / times[2]) << " 倍" << endl;
        cout << "算法2 vs 算法2循环展开: 循环展开快 " 
             << fixed << setprecision(2) << (times[1] / times[3]) << " 倍" << endl;
        cout << "算法1循环展开 vs 算法2循环展开: 算法2循环展开快 " 
             << fixed << setprecision(2) << (times[2] / times[3]) << " 倍" << endl;
    }
};

int main(int argc, char* argv[]) {
    int n = 1024; // 默认矩阵大小
    int iterations = 5; // 默认测试次数

    // 处理命令行参数
    if (argc > 1) n = atoi(argv[1]);
    if (argc > 2) iterations = atoi(argv[2]);

    // 边界检查
    if (n <= 0 || n > MAX_N) {
        cout << "矩阵大小必须在 1-" << MAX_N << " 范围内" << endl;
        return 1;
    }
    if (iterations <= 0) {
        cout << "迭代次数必须大于 0" << endl;
        return 1;
    }

    try {
        MatrixOperations matrixOps(n);
        matrixOps.runPerformanceTest(iterations);
    } catch (const exception& e) {
        cerr << "发生错误: " << e.what() << endl;
        return 1;
    }

    return 0;
}
