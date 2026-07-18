#include "second_order_position_mpc.h"

#include <Eigen/Dense>

#include <iostream>

int main()
{
    rm::SecondOrderPositionMPCConfig config;
    config.model_dt_s = 0.004;
    config.horizon = 12;
    config.track_q = 10.0;
    config.command_q = 1.0;
    config.delta_r = 20.0;
    config.wn_rad_s = 20.0;
    config.zeta = 0.8;
    config.max_rate_deg_s = 720.0;
    config.max_lead_deg = 4.0;
    config.max_state_rate_deg_s = 720.0;

    rm::SecondOrderPositionMPC mpc;
    mpc.configure(config);
    mpc.reset(0.0, 0.0);

    const double command = mpc.update(5.0, 0.0, 0.0, config.model_dt_s);
    if (!std::isfinite(command)) {
        std::cerr << "MPC smoke failed: non-finite command" << std::endl;
        return 1;
    }

    std::cout << "MPC smoke command_deg=" << command
              << " solve_us=" << mpc.lastSolveUs() << std::endl;
    return 0;
}

