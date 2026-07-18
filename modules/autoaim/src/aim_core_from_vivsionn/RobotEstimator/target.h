#ifndef RM_TARGET_HPP
#define RM_TARGET_HPP

#include <Eigen/Dense>
#include <vector>
#include <cmath>
#include <iostream>

namespace rm
{

class Target
{
public:
    Target() = default;
    Target(int armor_num);
    bool jumped = false;

    // [核心接口] 同步外部 Estimator 计算好的状态
    void sync_state(const Eigen::VectorXd& rm_state);

    // [核心接口] 预测未来状态 (仅使用 CV 模型)
    void predict(double dt);

    Eigen::VectorXd ekf_x() const;
    std::vector<Eigen::Vector4d> armor_xyza_list() const;

private:
    int armor_num_ = 4; 
    
    // 内部状态向量 (11维)
    // [0]xc, [1]vxc, [2]yc, [3]vyc, [4]za, [5]vza, [6]yaw, [7]vyaw, [8]r.
    // Four-armors: [9] secondary radius delta, [10] secondary height delta.
    // Outpost: [9] id1 height delta, [10] id2 height delta.
    Eigen::VectorXd x_; 

    // 辅助函数
    Eigen::Vector3d get_armor_pos(double yaw_base, int id) const;
};

} // namespace rm

#endif // RM_TARGET_HPP
