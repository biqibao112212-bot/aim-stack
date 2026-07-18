// RobotEstimator/extendedkalmanfilter.h
#pragma once
#include <Eigen/Dense>

namespace rm
{

// 定义参数结构体，用于初始化不同的模型
struct EKFParams {
    double q_xyz;   // 位置过程噪声
    double q_yaw;   // 角速度过程噪声
    double q_r;     // 半径过程噪声
    double r_xyz;   // 位置观测噪声系数
    double r_yaw;   // Yaw观测噪声
};

enum KFC
{
    EKF_CENTER_X = 0, EKF_CENTER_V_X,
    EKF_CENTER_Y,     EKF_CENTER_V_Y,
    EKF_ARMOR_Z,      EKF_ARMOR_V_Z,
    EKF_ARMOR_YAW,    EKF_ARMOR_V_YAW,
    EKF_ROTATION_RADIUS,
};

class ExtendedKalmanFilter
{
public:
    ExtendedKalmanFilter() = default;
    
    explicit ExtendedKalmanFilter(const Eigen::MatrixXd & P0, const EKFParams & params);

    void setState(const Eigen::VectorXd & x0);
    void setdt(double dt) { _dt = dt; }

    Eigen::MatrixXd predict();
    Eigen::MatrixXd update(const Eigen::VectorXd& z);

    // 获取当前观测的似然度 (用于 IMM 权重计算)
    double getLikelihood() const { return _likelihood; }
    
    // 获取状态 (用于 IMM 交互)
    Eigen::VectorXd getState() const { return x_post; }

    // [修复] 协方差矩阵的 Get 和 Set 接口 (确保只定义一次)
    Eigen::MatrixXd getP() const { return P_post; }
    void setP(const Eigen::MatrixXd & P) { P_post = P; }

private:
    double _dt;
    double _likelihood; // 存储似然度
    EKFParams _params;  // 存储当前模型的参数

    Eigen::VectorXd f(const Eigen::VectorXd& x);
    Eigen::VectorXd h(const Eigen::VectorXd& x);
    Eigen::MatrixXd jacobian_f(const Eigen::VectorXd& x);
    Eigen::MatrixXd jacobian_h(const Eigen::VectorXd& x);
    Eigen::MatrixXd update_Q();
    Eigen::MatrixXd update_R(const Eigen::VectorXd& z);

    Eigen::MatrixXd F, H, Q, R, P_pri, P_post, K, I;
    Eigen::VectorXd x_pri, x_post;
    int n;
};
} // namespace rm