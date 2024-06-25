//
// Created by Sommer  on 2024/5/30.
//

#include "alilqr_solver.h"
#include "clock.hpp"
#include <algorithm>
#include <cmath>
#include <numeric>


template<typename T, unsigned int M, unsigned int N>
void ILQRSolver<T, M, N>::Solve(const State &x0, const Controls &u0, States &x_res_seq, Controls &u_res_seq) const {
    std::unique_ptr<ILQRSolverState<T, M, N>> ilqr_state =
            std::make_unique<ILQRSolverState<T, M, N>>(step_sizes().size());
    InitTraj(x0, u0, ilqr_state.get());
    GenerateTrajectory(ilqr_state.get());
    x_res_seq = ilqr_state->x_seq_;
    u_res_seq = ilqr_state->u_seq_;
}

template<typename T, unsigned int M, unsigned int N>
double ILQRSolver<T, M, N>::InitTraj(const State &x_0, const Controls &u_0, ILQRSolverState<T, M, N> *ilqr_state) const {
    ilqr_state->x0_ = x_0;
    ilqr_state->x_seq_[0] = x_0;
    ilqr_state->u_seq_ = u_0;
    ilqr_state->dV_ = Vector2d::Zero();

    double cost_ini = RollOut(ilqr_state->x0_, ilqr_state->u_seq_, ilqr_state);
    ilqr_state->cost_ = cost_ini;

    return ilqr_state->cost_;
}

template<typename T, unsigned int M, unsigned int N>
void ILQRSolver<T, M, N>::GenerateTrajectory(ILQRSolverState<T, M, N> *ilqr_state) const {
    States x_old;
    Controls u_old;

    bool reinit_derivatives = true; 
    for (int iter = 0; iter < kMaxIter_; iter++) {
        x_old = ilqr_state->x_seq_;
        u_old = ilqr_state->u_seq_;

        if (reinit_derivatives) {
            InitializeILQRSolverState(ilqr_state);
            reinit_derivatives = false;
        }
        
        // update Vx, Vxx, l, L, dV with backward_pass
        BackwardProcess(ilqr_state);
        double dcost = 0.0;
        double ratio = 0.0;
        double expected = 0.0;
        double new_cost = 0.0;
        bool is_fwd_pass_done =
                ForwardProcess(x_old, u_old, &new_cost, &dcost, &expected, &ratio,
                               ilqr_state);

        ilqr_state->g_norm_ = GetGradientNorm(ilqr_state->k_, ilqr_state->u_seq_);
        if (ilqr_state->g_norm_ < kTolGrad_ and dcost > 0.0) {
            ilqr_state->status_ = SolverStatus::OCPSolved;
            cout << "SUCCESS: gradient norm < tolGrad\n";
            break;
        }

        if (iter == 0) {
            if (ilqr_state->status_ != SolverStatus::OCPSolved)
                ilqr_state->status_ = SolverStatus::MaxIterReached;
            cout << "AL-iLQR is terminated\n";
        }

        if (is_fwd_pass_done) {
            reinit_derivatives = true;
            if (dcost < kTolFun_) {
                ilqr_state->status_ = SolverStatus::OCPSolved;
                cout << "SUCCESS: cost change < tolFun\n";
                break;
            }
        } else { 
            IncreaseRho(ilqr_state->drho_, ilqr_state->rho_);
            if (ilqr_state->rho_ > krhoMax_) {
                cout << "EXIT: rho > rhoMax\n";
                break;
            }
        }

        UpdateDualsAndPenalties(ilqr_state);

        if (iter == kMaxIter_) {
            if (ilqr_state->status_ != SolverStatus::OCPSolved)
                ilqr_state->status_ = SolverStatus::MaxIterReached;
            cout << "EXIT: Maximum iterations reached\n";
        }
    }

    problem().GetCostUnion()->EvaluateConstraints(ilqr_state->x_seq(), ilqr_state->u_seq());
    double max_violation = problem().GetCostUnion()->GetMaxViolation();
    cout << "Solver Status : ";
    ShowSolverState(ilqr_state->status_);
    cout << "max_violation : " << max_violation << "\n";
   return;
}

template<typename T, unsigned M, unsigned N>
void ILQRSolver<T, M, N>::UpdateDualsAndPenalties(ILQRSolverState<T, M, N> *ilqr_state) const {
    States x = ilqr_state->x_seq();
    Controls u = ilqr_state->u_seq();
    for (auto &cons_ptr : problem().GetCostUnion()->GetEqConstraints()) {
        cons_ptr->UpdateDual();
        cons_ptr->UpdatePenalty();
    }

    for (auto &cons_ptr : problem().GetCostUnion()->GetIneqConstraints()) {
        cons_ptr->UpdateDual();
        cons_ptr->UpdatePenalty();
    }
}

template<typename T, unsigned int M, unsigned int N>
bool ILQRSolver<T, M, N>::ForwardProcess(const States &x_old, const Controls &u_old, double *new_cost, double *dcost,
                                         double *expected, double *ratio, ILQRSolverState<T, M, N> *ilqr_state) const {
    bool is_fwd_pass_done = false;
    for (const auto &alpha : alpha_vec_) {
        Controls u_plu_seq_feedforward = ilqr_state->u_seq_;
        for (size_t j = 0; j < ilqr_state->u_seq_.size(); j++) {
            u_plu_seq_feedforward[j] += ilqr_state->k_[j] * alpha;
        }
        *new_cost = RollOut(ilqr_state->x0_, u_plu_seq_feedforward, ilqr_state);

        *dcost = ilqr_state->cost_ - *new_cost;
        *expected = -alpha * (ilqr_state->dV_(0) + alpha * ilqr_state->dV_(1));

        if (*expected > 0) {
            *ratio = *dcost / *expected;
        } else {
            *ratio = sgn(*dcost);
            ilqr_state->status_ = SolverStatus::CostIncrease;
            cout << "Warning: non-positive expected reduction\n";
        }

        if (*ratio > kRatioMin_ and *ratio < krhoMax_) {
            is_fwd_pass_done = true;
            DecreaseRho(ilqr_state->drho_, ilqr_state->rho_);
            ilqr_state->cost_ = *new_cost; // accept step
            break;
        }
        ilqr_state->x_seq_ = x_old;
        ilqr_state->u_seq_ = u_old;
    }

    return is_fwd_pass_done;
}

template<typename T, unsigned int M, unsigned int N>
double ILQRSolver<T, M, N>::RollOut(const State &x0, const Controls &u, ILQRSolverState<T, M, N> *ilqr_state) const {
    double total_cost = 0;

    State x_curr = x0;
    Control u_curr;

    const uint32_t step_nums = step_sizes_.size();
    States x_new(step_nums + 1);
    x_new[0] = x0;
    double timer = 0.0;
    problem().GetModel()->SetTimer(timer);
    for (uint32_t t = 0; t < step_nums + 1; t++) {
        u_curr = u[t];
        if (ilqr_state->K_.size() > 0)
            // apply LQR control gains after first iteration
            u_curr += ilqr_state->K_[t] * (x_new[t] - ilqr_state->x_seq_[t]);

        ilqr_state->u_seq_[t] = u_curr; // no clamping since no control limit
        total_cost += problem().Evaluate(t, x_curr, u_curr);
        x_curr = problem().GetModel()->ForwardCalculation(x_curr, u_curr, step_sizes_[t]);
        if (use_timer_)
            problem().GetModel()->UpdateTimer(step_sizes_[t]);
        if (t < step_nums)
            x_new[t + 1] = x_curr;
    }

    ilqr_state->x_seq_ = x_new;
    return total_cost;
}

template<typename T, unsigned int M, unsigned int N>
void ILQRSolver<T, M, N>::BackwardProcess(ILQRSolverState<T, M, N> *ilqr_state) const {
    const uint32_t step_nums = step_sizes().size();
    if (step_nums == 0) {
        cout << "iLQR backward_pass: planning horizon length is zero\n";
        return;
    }
    while (true) {
        bool is_cost_psd = true;
        ilqr_state->dV_.setZero();
        int max_reg_count = 0;
        for (int i = static_cast<int>(step_nums - 1); i >= 0; i--) {
            ilqr_state->Qx_ = ilqr_state->lx_[i] + (ilqr_state->fx_[i].transpose() *
                                                    ilqr_state->Vx_[i + 1]) + ilqr_state->cx_[i];
            ilqr_state->Qu_ = ilqr_state->lu_[i] + (ilqr_state->fu_[i].transpose() *
                                                    ilqr_state->Vx_[i + 1]) + ilqr_state->cu_[i];
            ilqr_state->Qxx_ =
                    ilqr_state->lxx_[i] + (ilqr_state->fx_[i].transpose() *
                                           ilqr_state->Vxx_[i + 1] * ilqr_state->fx_[i])
                                           + ilqr_state->cxx_[i];
            ilqr_state->Qux_ = ilqr_state->lxu_[i].transpose() +
                               (ilqr_state->fu_[i].transpose() *
                                ilqr_state->Vxx_[i + 1] * ilqr_state->fx_[i])
                                + ilqr_state->cxu_[i].transpose();
            ilqr_state->Quu_ =
                    ilqr_state->luu_[i] + (ilqr_state->fu_[i].transpose() *
                                           ilqr_state->Vxx_[i + 1] * ilqr_state->fu_[i])
                                           + ilqr_state->cuu_[i];
            ilqr_state->QuuF_ = ilqr_state->luu_[i] +
                                ilqr_state->fu_[i].transpose() *
                                (ilqr_state->Vxx_[i + 1] + (ilqr_state->rho_ * MatrixLXX::Identity()))
                                * ilqr_state->fu_[i];

            MatrixLUU QuuF_inv;
            double QuuF_det;
            bool QuuF_invertible;
            double tol = 1.0e-6;
            ilqr_state->QuuF_.computeInverseAndDetWithCheck(QuuF_inv, QuuF_det,
                                                            QuuF_invertible, tol);
            if (!QuuF_invertible) {
                is_cost_psd = false;
                break;
            }

            ilqr_state->k_i_ = -QuuF_inv * ilqr_state->Qu_;
            ilqr_state->K_i_ = -QuuF_inv * ilqr_state->Qux_;

            ilqr_state->dV_(0) += ilqr_state->k_i_.transpose() * ilqr_state->Qu_;
            ilqr_state->dV_(1) += 0.5 * (ilqr_state->k_i_.transpose() *
                                         ilqr_state->Quu_ * ilqr_state->k_i_)[0];
            ilqr_state->Vx_[i] =
                    ilqr_state->Qx_ +
                    ilqr_state->K_i_.transpose() * ilqr_state->Quu_ * ilqr_state->k_i_ +
                    ilqr_state->K_i_.transpose() * ilqr_state->Qu_ +
                    ilqr_state->Qux_.transpose() * ilqr_state->k_i_;
            ilqr_state->Vxx_[i] =
                    ilqr_state->Qxx_ +
                    ilqr_state->K_i_.transpose() * ilqr_state->Quu_ * ilqr_state->K_i_ +
                    ilqr_state->K_i_.transpose() * ilqr_state->Qux_ +
                    ilqr_state->Qux_.transpose() * ilqr_state->K_i_;
            ilqr_state->Vxx_[i] =
                    0.5 * (ilqr_state->Vxx_[i] + ilqr_state->Vxx_[i].transpose());


            ilqr_state->k_[i] = ilqr_state->k_i_;
            ilqr_state->K_[i] = ilqr_state->K_i_;
        }

        if (is_cost_psd) {
            break;
        } else {
            IncreaseRho(ilqr_state->drho_, ilqr_state->rho_);
            if (ilqr_state->rho_ > krhoMax_) {
                ++max_reg_count;
                if (max_reg_count > kMaxRegCount_) {
                    ilqr_state->status_ = SolverStatus::RegularizationFailed;
                    break;
                }
            }
        }
    }
}

template<typename T, unsigned int M, unsigned int N>
void ILQRSolver<T, M, N>::IncreaseRho(double &drho, double &rho) const {
    drho = std::max(drho * krhoFactor_, krhoFactor_);
    rho = std::max(rho * drho, krhoMin_);
    rho = std::min(rho, krhoMax_);
}

template<typename T, unsigned int M, unsigned int N>
void ILQRSolver<T, M, N>::DecreaseRho(double &drho, double &rho) const {
    drho = std::min(drho / krhoFactor_, 1 / krhoFactor_);
    rho = std::max(rho * drho, krhoMin_);
    rho = std::min(rho, krhoMax_);
}

template<typename T, unsigned int M, unsigned int N>
double ILQRSolver<T, M, N>::GetGradientNorm(const VecUs &l, const Controls &u) const {
    std::vector<double> vals(l.size());
    for (uint32_t i = 0; i < l.size(); i++) {
        VectorXd v = l[i].cwiseAbs().array() / (u[i].cwiseAbs().array() + 1);
        vals[i] = v.maxCoeff();
    }
    return std::accumulate(vals.begin(), vals.end(), 0.0) / vals.size();
}

template<typename T, unsigned M, unsigned N>
double ILQRSolver<T, M, N>::GetMaxViolation() const {
    return problem().GetCostUnion()->GetMaxViolation();
}

template<typename T, unsigned M, unsigned N>
void ILQRSolver<T, M, N>::InitializeILQRSolverState(ILQRSolverState<T, M, N> *ilqr_state) const {
    const uint32_t step_nums = step_sizes().size();
    States x = ilqr_state->x_seq();
    Controls u = ilqr_state->u_seq();
    CalcKinematicsDerivatives(x, u, &(ilqr_state->fx_), &(ilqr_state->fu_));
    for (uint32_t t = 0; t < step_nums; ++t) {
        (void) problem().GetCostUnion()->Evaluate(t, x[t], u[t]);
        problem().GetCostUnion()->CalcCostGradient(t, x[t], u[t],
                                                   ilqr_state->lx_[t], ilqr_state->lu_[t]);
        problem().GetCostUnion()->CalcAugLagConstraintGradient(t, x[t], u[t],
                                                         ilqr_state->cx_[t], ilqr_state->cu_[t]);
        problem().GetCostUnion()->CalcCostHessian(t, x[t], u[t],
                                                  ilqr_state->lxx_[t],
                                                  ilqr_state->luu_[t],
                                                  ilqr_state->lxu_[t]);
        problem().GetCostUnion()->CalcAugLagConstraintHessian(t, x[t], u[t],
                                                              ilqr_state->cxx_[t],
                                                              ilqr_state->cuu_[t],
                                                              ilqr_state->cxu_[t]);

    }
    problem().GetCostUnion()->CalcTerminalCostToGo(x[step_nums]);
    ilqr_state->Vx_[step_nums] = problem().GetCostUnion()->GetFinalCostToGoGradient();
    ilqr_state->Vxx_[step_nums] = problem().GetCostUnion()->GetFinalCostToGoHessian();
}

template<typename T, unsigned int M, unsigned int N>
void ILQRSolver<T, M, N>::CalcKinematicsDerivatives(const States &x, const Controls &u, MatrixLXXs *f_x,
                                                    MatrixLXUs *f_u) const {
    const uint32_t step_nums = step_sizes().size();
    for (uint32_t t = 0; t < step_nums; t++) {
        (*f_x)[t] =
                problem().GetModel()->JacobianX(x[t], u[t], step_sizes_[t]);
        (*f_u)[t] =
                problem().GetModel()->JacobianU(x[t], u[t], step_sizes_[t]);
    }
}

template<typename T, unsigned M, unsigned N>
bool ILQRSolver<T, M, N>::IsTerminated() {
}

template<typename T, unsigned int M, unsigned int N>
const int ILQRSolver<T, M, N>::kMaxIter_ = 20;

template<typename T, unsigned int M, unsigned int N>
const double ILQRSolver<T, M, N>::kTolFun_ = 1e-6;

template<typename T, unsigned int M, unsigned int N>
const double ILQRSolver<T, M, N>::kTolGrad_ = 1e-6;

template<typename T, unsigned int M, unsigned int N>
const double ILQRSolver<T, M, N>::krhoFactor_ = 1.6;

template<typename T, unsigned int M, unsigned int N>
const double ILQRSolver<T, M, N>::krhoMax_ = 1e8;

template<typename T, unsigned int M, unsigned int N>
const double ILQRSolver<T, M, N>::krhoMin_ = 1e-8;

template<typename T, unsigned int M, unsigned int N>
const double ILQRSolver<T, M, N>::krhoMinGrad_ = 1e-5;

template<typename T, unsigned int M, unsigned int N>
const double ILQRSolver<T, M, N>::kRatioMin_ = 1.0e-8;

template<typename T, unsigned M, unsigned N>
const double ILQRSolver<T, M, N>::kRationMax_ = 10.0;

template<typename T, unsigned M, unsigned N>
const double ILQRSolver<T, M, N>::kViolationTol_ = 1e-2;

template<typename T, unsigned M, unsigned N>
const int ILQRSolver<T, M, N>::kMaxRegCount_ = 20;

template<typename T, unsigned int M, unsigned int N>
const std::array<double, 11> ILQRSolver<T, M, N>::alpha_vec_{
        1.0000, 0.50, 0.25, 0.125, 0.0631, 0.0316,
        0.0158, 0.0079, 0.0040, 0.0020, 0.0010};

template class ILQRSolver<double, 5, 1>;

template class ILQRSolver<double, 4, 2>;

template class ILQRSolver<double, 6, 2>;