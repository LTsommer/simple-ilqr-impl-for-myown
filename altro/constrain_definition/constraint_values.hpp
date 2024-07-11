//
// Created by 廖田志浩 on 2024/6/20.
//

#ifndef ALILQR_CONSTRAINT_VALUES_HPP
#define ALILQR_CONSTRAINT_VALUES_HPP

#include "constraint.hpp"
#include "../ilqr_system_declaration.hpp"
#include <iostream>

template <typename T, unsigned M, unsigned N, class ConsType>
class ConstraintValue {
public:
    static constexpr double kDefaultPenalty = 1.0;
    static constexpr double kDefaultPenaltyFactor = 2.0;
    static constexpr double kDefaultLambda = 1.0e-2;
    static constexpr int dim = Eigen::Dynamic;
public:
    OCP_VARIABLES(T, M, N)
    ConstraintValue(const int horizon) : horizon_(horizon + 1){
        lambda_.setZero(horizon_);
        penalty_.setZero(horizon_);
        cons_val_mat_.setZero(horizon_);
        ctg_val_mat_.setZero(horizon_);
        violations_.setZero(horizon_);
    };

    ~ConstraintValue() = default;

    void UpdateDual() {
        for (int t = 0; t < horizon_; ++t)
            ConsType::Name::UpdateDual(cons_val_mat_(t), penalty_(t), lambda_(t));
    }

    void UpdatePenalty() {
        for (int t = 0; t < horizon_; ++t)
            ConsType::Name::UpdatePenalty(cons_val_mat_(t), lambda_(t), penalty_factor_, penalty_(t));
    }

    void UpdateConfig(const CostTermConfig &config) {
        cons_ptr_->UpdateConfig(config);
    }

    void SetPenalty(const double penalty) {
        penalty_.setConstant(penalty);
    }

    void SetDual(const double lambda) {
        lambda_.setConstant(lambda);
    }

    double GetLambda(const int step) const {return lambda_(step);}

    double GetPenalty(const int step) const {return penalty_(step);}

//    const Eigen::Matrix<T, dim, 1> &GetLambda() const {return lambda_;}

    double GetMaxPenalty() const {return penalty_.template lpNorm<Eigen::Infinity>();}

    double GetMaxViolation() const {return violations_.template lpNorm<Eigen::Infinity>();}

    void LoadConstraint(ConstraintPtr<T, M, N, ConsType> && cons) {
        cons_ptr_.swap(cons);
    };

    double ConsVal() { return cons_val_; }

    double AugLagVal() { return ctg_val_; }

    VecX GradientX() { return grad_x_; }

    VecU GradientU() { return grad_u_; }

    bool Evaluate(const int step, const State &x, const Control &u) {
//        cons_ptr_->GetConstraintType();
        bool success = cons_ptr_->Evaluate(step, x, u, cons_val_);
        cons_val_mat_(step) = cons_val_;
        violations_(step) = cons_val_ < 0.0 ? 0.0 : cons_val_;
        if (success) {
            ctg_val_ = (lambda_(step) + 0.5 * cons_val_ * penalty_(step)) * cons_val_;
        }
        else {
            ctg_val_ = 0.0;
        }
        ctg_val_mat_(step) = ctg_val_;
        return success;
    }

//    double MaxViolation() {
//        auto iter = std::max_element(violations_.begin(), violations_.end());
//        double max_val = violations_[std::distance(violations_.begin(), iter)];
//        violations_.clear();
//        return max_val;
//    }
    std::type_index GetTypeIndex() {return cons_ptr_->GetTypeIndex();}

    std::string GetName() {return cons_ptr_->GetName();}

    void Gradient(const int step, const State &x, const Control &u) {
        grad_x_.setZero();
        grad_u_.setZero();
        cons_ptr_->Gradient(step, x, u, grad_x_, grad_u_);
    }

    void Hessian(const int step, const State &x, const Control &u) {
        hessian_xx_.setZero();
        hessian_uu_.setZero();
        hessian_xu_.setZero();
        cons_ptr_->Hessian(step, x, u, hessian_xx_, hessian_uu_, hessian_xu_);
    }

private:
    int horizon_;
    Eigen::Matrix<T, dim, 1> lambda_;   // penalty scale
    double penalty_factor_ = kDefaultPenaltyFactor;
    double cons_val_;
    double ctg_val_;
    Eigen::Matrix<T, dim, 1> cons_val_mat_;  // c(x)
    Eigen::Matrix<T, dim, 1> ctg_val_mat_;  // (lambda + 0.5 * penalty * c(x)) * c(x)
    Eigen::Matrix<T, dim, 1> penalty_;
    MatrixLXX hessian_xx_;
    MatrixLUU hessian_uu_;
    MatrixLXU hessian_xu_;
    VecX grad_x_;
    VecU grad_u_;
    ConstraintPtr<T, M, N, ConsType> cons_ptr_;
    Eigen::Matrix<T, dim, 1> violations_;
};

template<typename T, unsigned M, unsigned N, class ConsType>
using ConstraintValuePtr = std::unique_ptr<ConstraintValue<T, M, N, ConsType>>;


#endif //ALILQR_CONSTRAINT_VALUES_HPP
