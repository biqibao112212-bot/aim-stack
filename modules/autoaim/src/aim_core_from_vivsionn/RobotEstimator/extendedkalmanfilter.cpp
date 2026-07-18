#include "extendedkalmanfilter.h"
#include <iostream>
#include <cmath>

namespace rm
{

ExtendedKalmanFilter::ExtendedKalmanFilter(const Eigen::MatrixXd & P0, const EKFParams & params)
    : _params(params), P_post(P0), n(P0.rows()), I(Eigen::MatrixXd::Identity(n, n)), x_pri(n), x_post(n)
{
    _likelihood = 0.0;
}

void ExtendedKalmanFilter::setState(const Eigen::VectorXd & x0) { 
    x_post = x0; 
}

Eigen::MatrixXd ExtendedKalmanFilter::predict()
{
    F = jacobian_f(x_post);
    Q = update_Q();
    x_pri = f(x_post);
    
    // P_pri = F * P_post * F.transpose() + Q
    P_pri = F * P_post * F.transpose() + Q;
    
    x_post = x_pri; 
    P_post = P_pri;
    return x_pri;
}

Eigen::MatrixXd ExtendedKalmanFilter::update(const Eigen::VectorXd & z)
{
    H = jacobian_h(x_pri);
    R = update_R(z);

    // 1. Calculate measurement residual
    Eigen::VectorXd y = z - h(x_pri);
    
    // Normalize yaw angle residual (-PI ~ PI)
    while(y(3) > M_PI) y(3) -= 2*M_PI;
    while(y(3) < -M_PI) y(3) += 2*M_PI;

    // 2. Calculate residual covariance S = HPH' + R
    Eigen::MatrixXd S = H * P_pri * H.transpose() + R;
    
    // 3. Calculate Optimal Kalman Gain
    K = P_pri * H.transpose() * S.inverse();

    // 4. Update State
    x_post = x_pri + K * y;

    // 5. Update Covariance P = (I - KH)P
    // [Fix] Explicitly evaluate K*H to avoid lazy-evaluation dimension issues
    Eigen::MatrixXd KH = K * H;

    // [Safety Check] Ensure I is correct size (n x n)
    if (I.rows() != n || I.cols() != n) {
        // This handles cases where I might have been zero-initialized during copy/move
        I = Eigen::MatrixXd::Identity(n, n);
    }

    // Now perform subtraction with guaranteed dimensions
    P_post = (I - KH) * P_pri;

    // 6. Calculate Likelihood for IMM
    double detS = S.determinant();
    if (std::abs(detS) < 1e-9) detS = 1e-9;
    
    double mahal = y.transpose() * S.inverse() * y;
    _likelihood = std::exp(-0.5 * mahal) / std::sqrt(std::pow(2 * M_PI, 4) * detS);

    if (_likelihood < 1e-50) _likelihood = 1e-50;

    return x_post;
}

// === Process Functions ===

Eigen::VectorXd ExtendedKalmanFilter::f(const Eigen::VectorXd& x) {
    Eigen::VectorXd x_new = x;
    x_new(0) += x(1) * _dt;
    x_new(2) += x(3) * _dt;
    x_new(4) += x(5) * _dt;
    x_new(6) += x(7) * _dt;
    return x_new;
}

Eigen::VectorXd ExtendedKalmanFilter::h(const Eigen::VectorXd& x) {
    Eigen::VectorXd z(4);
    double xc = x(0), yc = x(2), yaw = x(6), r = x(8);
    z(0) = xc - r * cos(yaw);
    z(1) = yc - r * sin(yaw);
    z(2) = x(4);
    z(3) = x(6);
    return z;
}

Eigen::MatrixXd ExtendedKalmanFilter::jacobian_f(const Eigen::VectorXd& x) {
    Eigen::MatrixXd f = Eigen::MatrixXd::Identity(9, 9);
    f(0, 1) = _dt; f(2, 3) = _dt; f(4, 5) = _dt; f(6, 7) = _dt;
    return f;
}

Eigen::MatrixXd ExtendedKalmanFilter::jacobian_h(const Eigen::VectorXd& x) {
    Eigen::MatrixXd h(4, 9);
    double yaw = x(6), r = x(8);
    h << 1, 0, 0, 0, 0, 0, r*sin(yaw), 0, -cos(yaw),
         0, 0, 1, 0, 0, 0, -r*cos(yaw), 0, -sin(yaw),
         0, 0, 0, 0, 1, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 1, 0, 0;
    return h;
}

Eigen::MatrixXd ExtendedKalmanFilter::update_Q() {
    Eigen::MatrixXd q(9, 9);
    double t = _dt;
    double x = _params.q_xyz;
    double y = _params.q_yaw;
    double r = _params.q_r;

    double q_x_x = pow(t, 4)/4*x, q_x_vx = pow(t, 3)/2*x, q_vx_vx = pow(t, 2)*x;
    double q_y_y = pow(t, 4)/4*y, q_y_vy = pow(t, 3)/2*y, q_vy_vy = pow(t, 2)*y;
    double q_r = pow(t, 4)/4*r;

    q << q_x_x, q_x_vx, 0, 0, 0, 0, 0, 0, 0,
         q_x_vx, q_vx_vx, 0, 0, 0, 0, 0, 0, 0,
         0, 0, q_x_x, q_x_vx, 0, 0, 0, 0, 0,
         0, 0, q_x_vx, q_vx_vx, 0, 0, 0, 0, 0,
         0, 0, 0, 0, q_x_x, q_x_vx, 0, 0, 0,
         0, 0, 0, 0, q_x_vx, q_vx_vx, 0, 0, 0,
         0, 0, 0, 0, 0, 0, q_y_y, q_y_vy, 0,
         0, 0, 0, 0, 0, 0, q_y_vy, q_vy_vy, 0,
         0, 0, 0, 0, 0, 0, 0, 0, q_r;
    return q;
}

Eigen::MatrixXd ExtendedKalmanFilter::update_R(const Eigen::VectorXd& z) {
    Eigen::DiagonalMatrix<double, 4> r;
    double d = std::sqrt(z[0]*z[0] + z[1]*z[1] + z[2]*z[2]);
    
    double xyz_sigma = _params.r_xyz + 0.02 * d; 
    double yaw_sigma = _params.r_yaw; 

    r.diagonal() << xyz_sigma*xyz_sigma, xyz_sigma*xyz_sigma, xyz_sigma*xyz_sigma, yaw_sigma*yaw_sigma;
    return r;
}

} // namespace rm