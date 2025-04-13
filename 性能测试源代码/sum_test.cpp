#include <iostream>
#include <vector>
#include <chrono>
#include <iomanip>
#include <random>
#include <functional>
#include <algorithm>
#include <cmath>
#include <string>

// 使用命名空间
using namespace std;
using namespace std::chrono;

// 定义最大数组大小
const int MAX_N = 100000000;

// 高精度计时器
class Timer {
private:
    high_resolution_clock::time_point start_time;
    string operation_name;

public:
    Timer(const string& name) : operation_name(name) {
        start_time = high_resolution_clock::now();
    }

    ~Timer() {
        auto end_time = high_resolution_clock::now();
        duration<double> time_span = duration_cast<duration<double>>(end_time - start_time);
        cout << operation_name << ": " << fixed << setprecision(9) << time_span.count() << " 秒" << endl;
    }

    double getElapsedTime() {
        auto end_time = high_resolution_clock::now();
        duration<double> time_span = duration_cast<duration<double>>(end_time - start_time);
        return time_span.count();
    }
};

class SumAlgorithms {
private:
    vector<double> data; // 待求和的数组
    int n; // 数组大小

    // 初始化数据（确定性随机）
    void initializeData() {
        mt19937 rng(12345); // 固定种子
        uniform_real_distribution<double> dist(-1.0, 1.0);

        for (int i = 0; i < n; i++) {
            data[i] = dist(rng);
        }
    }

public:
    SumAlgorithms(int size) : n(size) {
        // 确保n是2的幂，便于递归算法实现
        if (size & (size - 1)) {
            n = 1 << static_cast<int>(log2(size) + 1);
            cout << "调整数组大小为 " << n << " (2的幂)" << endl;
        }
        data.resize(n);
        initializeData();
    }

    // 算法1: 平凡算法（链式）
    double algorithm1() {
        double sum = 0.0;
        for (int i = 0; i < n; i++) {
            sum += data[i];
        }
        return sum;
    }

    // 算法2: 多链路式
    double algorithm2() {
        double sum1 = 0.0, sum2 = 0.0;
        for (int i = 0; i < n; i += 2) {
            sum1 += data[i];
            if (i + 1 < n) { // 防止越界
                sum2 += data[i + 1];
            }
        }
        return sum1 + sum2;
    }

    // 算法3: 递归方式
    double algorithm3() {
        // 创建数据副本以免修改原始数据
        vector<double> temp = data;
        int size = n;
        
        // 递归辅助函数
        function<void(vector<double>&, int)> recursiveSum = [&](vector<double>& arr, int m) {
            if (m == 1) return;
            
            for (int i = 0; i < m / 2; i++) {
                arr[i] += arr[m - i - 1];
            }
            
            recursiveSum(arr, m / 2);
        };
        
        recursiveSum(temp, size);
        return temp[0];
    }

    // 算法4: 二重循环方式
    double algorithm4() {
        // 创建数据副本以免修改原始数据
        vector<double> temp = data;
        
        for (int m = n; m > 1; m /= 2) {
            for (int i = 0; i < m / 2; i++) {
                temp[i] = temp[i * 2] + temp[i * 2 + 1];
            }
        }
        
        return temp[0];
    }

    // 运行性能测试
    void runPerformanceTest(int iterations) {
        vector<double> results(4);
        vector<double> times(4, 0.0);

        cout << "\n数组大小: " << n << ", 测试迭代次数: " << iterations << endl;

        // 预热运行
        results[0] = algorithm1();
        results[1] = algorithm2();
        results[2] = algorithm3();
        results[3] = algorithm4();

        // 测试算法1（平凡算法）
        cout << "\n测试算法1（平凡算法）..." << endl;
        for (int i = 0; i < iterations; i++) {
            Timer timer("运行 #" + to_string(i+1));
            results[0] = algorithm1();
            times[0] += timer.getElapsedTime();
        }
        cout << "算法1平均执行时间: " << fixed << setprecision(9) << times[0] / iterations << " 秒" << endl;
        cout << "算法1结果: " << results[0] << endl;

        // 测试算法2（多链路式）
        cout << "\n测试算法2（多链路式）..." << endl;
        for (int i = 0; i < iterations; i++) {
            Timer timer("运行 #" + to_string(i+1));
            results[1] = algorithm2();
            times[1] += timer.getElapsedTime();
        }
        cout << "算法2平均执行时间: " << fixed << setprecision(9) << times[1] / iterations << " 秒" << endl;
        cout << "算法2结果: " << results[1] << endl;

        // 测试算法3（递归方式）
        cout << "\n测试算法3（递归方式）..." << endl;
        for (int i = 0; i < iterations; i++) {
            Timer timer("运行 #" + to_string(i+1));
            results[2] = algorithm3();
            times[2] += timer.getElapsedTime();
        }
        cout << "算法3平均执行时间: " << fixed << setprecision(9) << times[2] / iterations << " 秒" << endl;
        cout << "算法3结果: " << results[2] << endl;

        // 测试算法4（二重循环）
        cout << "\n测试算法4（二重循环）..." << endl;
        for (int i = 0; i < iterations; i++) {
            Timer timer("运行 #" + to_string(i+1));
            results[3] = algorithm4();
            times[3] += timer.getElapsedTime();
        }
        cout << "算法4平均执行时间: " << fixed << setprecision(9) << times[3] / iterations << " 秒" << endl;
        cout << "算法4结果: " << results[3] << endl;

        // 验证结果一致性
        cout << "\n验证计算结果..." << endl;
        bool consistent = true;
        for (int i = 1; i < 4; i++) {
            if (abs(results[0] - results[i]) > 1e-6) {
                cout << "警告: 算法" << i+1 << "结果与算法1不一致!" << endl;
                cout << "差异: " << abs(results[0] - results[i]) << endl;
                consistent = false;
            }
        }
        if (consistent) {
            cout << "验证通过: 所有算法结果一致" << endl;
        }

        // 输出性能比较
        cout << "\n性能比较:" << endl;
        cout << "算法1 vs 算法2: 算法2快 " 
             << fixed << setprecision(2) << (times[0] / times[1]) << " 倍" << endl;
        cout << "算法1 vs 算法3: 算法3快 " 
             << fixed << setprecision(2) << (times[0] / times[2]) << " 倍" << endl;
        cout << "算法1 vs 算法4: 算法4快 " 
             << fixed << setprecision(2) << (times[0] / times[3]) << " 倍" << endl;
        cout << "算法2 vs 算法3: 算法" << (times[1] < times[2] ? "2" : "3") << "快 " 
             << fixed << setprecision(2) << (times[1] < times[2] ? (times[2] / times[1]) : (times[1] / times[2])) << " 倍" << endl;
        cout << "算法2 vs 算法4: 算法" << (times[1] < times[3] ? "2" : "4") << "快 " 
             << fixed << setprecision(2) << (times[1] < times[3] ? (times[3] / times[1]) : (times[1] / times[3])) << " 倍" << endl;
        cout << "算法3 vs 算法4: 算法" << (times[2] < times[3] ? "3" : "4") << "快 " 
             << fixed << setprecision(2) << (times[2] < times[3] ? (times[3] / times[2]) : (times[2] / times[3])) << " 倍" << endl;
    }
};

int main(int argc, char* argv[]) {
    int n = 1000000; // 默认数组大小
    int iterations = 5; // 默认测试次数

    // 处理命令行参数
    if (argc > 1) n = atoi(argv[1]);
    if (argc > 2) iterations = atoi(argv[2]);

    // 边界检查
    if (n <= 0 || n > MAX_N) {
        cout << "数组大小必须在 1-" << MAX_N << " 范围内" << endl;
        return 1;
    }
    if (iterations <= 0) {
        cout << "迭代次数必须大于 0" << endl;
        return 1;
    }

    try {
        SumAlgorithms sumAlgs(n);
        sumAlgs.runPerformanceTest(iterations);
    } catch (const exception& e) {
        cerr << "发生错误: " << e.what() << endl;
        return 1;
    }

    return 0;
}