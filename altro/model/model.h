//
// Created by 廖田志浩 on 2024/6/6.
//

#ifndef CILQR_MODEL_H
#define CILQR_MODEL_H

#include "../ilqr_system_declaration.hpp"

template<typename T, unsigned int M, unsigned int N>
class ModelBase {
public:
    OCP_VARIABLES(T, M, N)
    ModelBase() = default;
    virtual ~ModelBase() = default;

    virtual inline A JacobianX(const State &x,
                               const Control &u,
                               const double dt) const = 0;

    virtual inline B JacobianU(const State &x,
                               const Control &u,
                               const double dt) const = 0;

    virtual inline State ForwardCalculation(const State &x,
                                            const Control &u,
                                            const double dt) const = 0;

    State RK4Methode(const State &x, const Control &u, const double dt) const {
        State K1 = ForwardCalculation(x, u, dt);
        State K2 = ForwardCalculation(x + 0.5 * K1 * dt, u, dt);
        State K3 = ForwardCalculation(x + 0.5 * K2 * dt, u, dt);
        State K4 = ForwardCalculation(x + K3 * dt, u, dt);
        return x + 1 / 6 * dt * (K1 + K2 + K3 + K4);
    }

//    virtual inline void SetTimer(const double timer) = 0;
//
//    virtual inline void UpdateTimer(const double dt) = 0;

//protected:
//    double timer_;
};

#endif //CILQR_MODEL_H
