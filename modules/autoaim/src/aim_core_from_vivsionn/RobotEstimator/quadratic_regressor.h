#pragma once
#include <deque>
#include <vector>
#include <cmath>
#include <Eigen/Dense>

class QuadraticRegressor {
private:
    struct DataPoint {
        double t; // 绝对时间
        double y; // 值
    };
    
    std::deque<DataPoint> history;
    int max_history_size;

    // [新增] 缓存拟合系数和基准时间
    // y = a * dt^2 + b * dt + c
    double a = 0, b = 0, c = 0;
    double t_base = 0; 
    bool coeffs_dirty = true; // 标记是否需要重新计算（可选，但在 update 中直接计算更简单）

    // [新增] 内部私有函数：更新拟合系数
    void fit() {
        int n = history.size();
        if (n < 3) return; // 数据太少，由 predict 处理降级逻辑

        t_base = history.back().t; // 以最新数据时间为 0 点，避免数值稳定性问题
        
        Eigen::Matrix3d A = Eigen::Matrix3d::Zero();
        Eigen::Vector3d B = Eigen::Vector3d::Zero();
        
        for (const auto& p : history) {
            double dt = p.t - t_base;
            double dt2 = dt * dt;
            double dt3 = dt2 * dt;
            double dt4 = dt3 * dt;
            
            A(0, 0) += dt4; A(0, 1) += dt3; A(0, 2) += dt2;
            A(1, 0) += dt3; A(1, 1) += dt2; A(1, 2) += dt;
            A(2, 0) += dt2; A(2, 1) += dt;  A(2, 2) += 1.0;
            
            B(0) += dt2 * p.y;
            B(1) += dt  * p.y;
            B(2) += p.y;
        }

        // 使用 LDLT 分解求解，速度快且稳定
        Eigen::Vector3d coeffs = A.ldlt().solve(B);
        a = coeffs(0);
        b = coeffs(1);
        c = coeffs(2);
    }

public:
    QuadraticRegressor(int window_size = 20) : max_history_size(window_size) {}

    int size() const {
        return history.size();
    }

    void clear() {
        history.clear();
        a = b = c = 0;
    }

    // [修改] Update 时顺便更新系数 (每帧只算1次)
    void update(double t, double y) {
        history.push_back({t, y});
        if (history.size() > max_history_size) {
            history.pop_front();
        }
        // 数据量足够时，立即更新系数
        if (history.size() >= 3) {
            fit();
        }
    }

    // [修改] Predict 变为极其廉价的查表计算 (每帧算几百次也没事)
    // 加上 const 修饰符，解决编译报错
    double predict(double future_time) const {
        int n = history.size();
        
        // 降级处理
        if (n < 1) return 0.0;
        if (n == 1) return history.back().y;
        if (n == 2) { 
            double t0 = history[0].t;
            double t1 = history[1].t;
            double y0 = history[0].y;
            double y1 = history[1].y;
            if (std::abs(t1 - t0) < 1e-6) return y1;
            double slope = (y1 - y0) / (t1 - t0);
            return y1 + slope * (future_time - t1);
        }

        // 直接使用缓存的系数进行预测
        double dt_pred = future_time - t_base;
        return a * dt_pred * dt_pred + b * dt_pred + c;
    }
};