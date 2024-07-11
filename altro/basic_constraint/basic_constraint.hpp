//
// Created by 廖田志浩 on 2024/6/20.
//

#ifndef ALILQR_BASIC_CONSTRAINT_H
#define ALILQR_BASIC_CONSTRAINT_H

#include "../ilqr_system_declaration.hpp"
#include "../constrain_definition/constraint_formulation.h"
#include "../constrain_definition/constraint.hpp"
#include <iostream>

using std::cout;


template<typename T, unsigned M, unsigned int N>
class LateralOffsetConstraint final : public Constraint<T, M, N, Inequality> {
public:
    OCP_VARIABLES(T, M, N);
    LateralOffsetConstraint(const CostTermConfig &config, const bool is_lb) {
        // g(x) = a_[i] * x + b_[i] < 0
        //
        const auto &p_bound = config.p_bound;
        size_t num_steps = p_bound.size();
        bound_types_ = is_lb ? config.boundary_types[0] : config.boundary_types[1];

        std::vector<Vector2d> line_vecs(num_steps); // p1p2 or p3p4
        std::vector<double> c1(num_steps);          // 1/||p1p2|| or 1/||p3p4||
        std::vector<double> c2(num_steps);          // c1 * (p2 x p1) or c1 * (p2 x p1)
        a_.resize(num_steps);
        b_.resize(num_steps);
        int start_idx = is_lb ? 0 : 2; // p1: 0, p3: 2
        int lb_coeff = start_idx - 1;  // if lb: 1, rb: -1
        // safe_dist - distance_to_boundary <= 0
        for (size_t i = 0; i < num_steps; i++) {
            line_vecs[i] = p_bound[i][start_idx + 1] - p_bound[i][start_idx];
            c1[i] = 1.0 / line_vecs[i].norm();
            c2[i] = c1[i] *
                    cross_prod(p_bound[i][start_idx + 1], p_bound[i][start_idx]);
            a_[i](kXPos) =
                    c1[i] * lb_coeff *
                    (p_bound[i][start_idx + 1](kYPos) - p_bound[i][start_idx](kYPos));
            a_[i](kYPos) =
                    c1[i] * lb_coeff *
                    (p_bound[i][start_idx](kXPos) - p_bound[i][start_idx + 1](kXPos));
            double safe_dist = is_lb ? config.left_safe_dist[i] : config.right_safe_dist[i];
//            cout << "safe dist" << safe_dist << "\n";
            b_[i] = lb_coeff * c2[i] + safe_dist;
        }
        Constraint<T, M, N, Inequality>::SetHorizon(config.steps.size());
    }

    void UpdateConfig(const CostTermConfig &config, const bool is_lb) {
        // g(x) = a_[i] * x + b_[i] > 0
        const auto &p_bound = config.p_bound;
        size_t num_steps = p_bound.size();
        bound_types_ = is_lb ? config.boundary_types[0] : config.boundary_types[1];

        std::vector<Vector2d> line_vecs(num_steps); // p1p2 or p3p4
        std::vector<double> c1(num_steps);          // 1/||p1p2|| or 1/||p3p4||
        std::vector<double> c2(num_steps);          // c1 * (p2 x p1) or c1 * (p2 x p1)
        a_.resize(num_steps);
        b_.resize(num_steps);
        int start_idx = is_lb ? 0 : 2; // p1: 0, p3: 2
        int lb_coeff = start_idx - 1;  // if lb: 1, rb: -1
        // safe_dist - distance_to_boundary <= 0
        for (size_t i = 0; i < num_steps; i++) {
            line_vecs[i] = p_bound[i][start_idx + 1] - p_bound[i][start_idx];
            c1[i] = 1.0 / line_vecs[i].norm();
            c2[i] = c1[i] *
                    cross_prod(p_bound[i][start_idx + 1], p_bound[i][start_idx]);
            a_[i](kXPos) =
                    c1[i] * lb_coeff *
                    (p_bound[i][start_idx + 1](kYPos) - p_bound[i][start_idx](kYPos));
            a_[i](kYPos) =
                    c1[i] * lb_coeff *
                    (p_bound[i][start_idx](kXPos) - p_bound[i][start_idx + 1](kXPos));
            double safe_dist = is_lb ? config.left_safe_dist[i] : config.right_safe_dist[i];
            b_[i] = lb_coeff * c2[i] + safe_dist;
        }
    }

    virtual std::type_index GetTypeIndex() const override {
        return std::type_index(typeid(LateralOffsetConstraint));
    }

    virtual std::string GetName() const override {return std::string("LateralOffsetConstraint");}

    virtual bool Evaluate(const int step,
                          const State &x,
                          const Control &u,
                          double &val) const override {
        if (step >= a_.size()) {
            return false;
        }

        if (step < bound_types_.size() &&
            bound_types_[step] == BoundType::Default) {
            val = 0.0;
            return true;
        }

        Vector2d p = x.template head<2>();
        val = a_[step].dot(p) + b_[step];
        return true;
    }

    virtual bool Gradient(const int step, const State &x, const Control &u,
                          Eigen::Ref<VecX> lx, Eigen::Ref<VecU> lu) const override {
        if (step >= a_.size()) {
            return false;
        }

        lx.setZero();
        if (step < bound_types_.size() &&
            bound_types_[step] == BoundType::Default) {
            return true;
        }
        lx(0) = a_[step](0);
        lx(1) = a_[step](1);

        lu.setZero();
        return true;
    }

    virtual bool Hessian(const int step, const State &x, const Control &u,
                         Eigen::Ref<MatrixLXX> lxx,
                         Eigen::Ref<MatrixLUU> luu,
                         Eigen::Ref<MatrixLXU> lxu) const override {
        lxx.setZero();
        luu.setZero();
        lxu.setZero();
        return true;
    }

private:
    std::vector<Vector2d> a_;
    std::vector<double> b_;
    std::vector<BoundType> bound_types_;
};

/*
  J_safe = w_plb * f(g_plb(x)) + w_prb * f(g_prb(x))
  where f() is the penalty function, currently using one-sided quadratic
  function, and g_plb(x), g_prb(x) represents the safe constraints
 */
template<typename T, unsigned M, unsigned int N>
class SafeConstraint final: public Constraint<T, M, N, Inequality>{
public:
    OCP_VARIABLES(T, M, N);
    SafeConstraint(const CostTermConfig &config)
            : w_plb_(config.weights.at(WeightPlb)),
              w_prb_(config.weights.at(WeightPrb)),
              steps_num_(config.p_bound.size()),
              line_segment_constraint_lb_(config, true),
              line_segment_constraint_rb_(config, false) {
        Constraint<T, M, N, Inequality>::SetHorizon(config.steps.size());
    }

    ~SafeConstraint() = default;

    virtual std::string GetName() const override {
        std::string name("SafeConstraint");
        return name;
    }

    virtual std::type_index GetTypeIndex() const override {
        return std::type_index(typeid(SafeConstraint));
    }

    virtual bool Evaluate(const int step, const State &x, const Control &u,
                          double &val) const override {
        val = 0.0;
        if (step > steps_num_) {
            return false;
        }

        // using one side quadratic cost f(g(x))
        double g_lb{0.0};
        double g_rb{0.0};
        line_segment_constraint_lb_.Evaluate(step, x, u, g_lb);
        line_segment_constraint_rb_.Evaluate(step, x, u, g_rb);


        double f_lb = one_side_cost_.Evaluate(g_lb);
        double f_rb = one_side_cost_.Evaluate(g_rb);

//        if ((step >= 5 and step <= 15) or (step >= 25 and step <= 28))


        val = w_plb_ * f_lb + w_prb_ * f_rb;
//        std::cout << "f_lb = " << f_lb << "\n"
//                  << "f_rb = " << f_rb << "\n"
//                  << "val = " << val << "\n";
        return true;
    };

    // f(g)' = f' * g'
    virtual bool Gradient(const int step, const State &x, const Control &u,
                          Eigen::Ref<VecX> lx, Eigen::Ref<VecU> lu) const override {
        lx.setZero();
        lu.setZero();
        if (step > steps_num_) {
            return false;
        }

        double g_lb{0.0};
        double g_rb{0.0};
        line_segment_constraint_lb_.Evaluate(step, x, u, g_lb);
        line_segment_constraint_rb_.Evaluate(step, x, u, g_rb);

        VecX grad_gx_lb, grad_gx_rb;
        VecU grad_gu_lb, grad_gu_rb;
        line_segment_constraint_lb_.Gradient(step, x, u, grad_gx_lb, grad_gu_lb);
        line_segment_constraint_rb_.Gradient(step, x, u, grad_gx_rb, grad_gu_rb);

        double grad_f_lb{0.0};
        double grad_f_rb{0.0};
        one_side_cost_.Gradient(g_lb, grad_f_lb);
        one_side_cost_.Gradient(g_rb, grad_f_rb);

        lx = w_plb_ * grad_f_lb * grad_gx_lb + w_prb_ * grad_f_rb * grad_gx_rb;
        lu = w_plb_ * grad_f_lb * grad_gu_lb + w_prb_ * grad_f_rb * grad_gu_rb;
        return true;
    };

    // f(g)'' = f'' * g' * g' + f' * g''
    virtual bool Hessian(const int step, const State &x, const Control &u,
                         Eigen::Ref<MatrixLXX> lxx,
                         Eigen::Ref<MatrixLUU> luu,
                         Eigen::Ref<MatrixLXU> lxu) const override {
        lxx.setZero();
        luu.setZero();
        lxu.setZero();
        if (step > steps_num_) {
            return false;
        }

        double g_lb{0.0};
        double g_rb{0.0};
        line_segment_constraint_lb_.Evaluate(step, x, u, g_lb);
        line_segment_constraint_rb_.Evaluate(step, x, u, g_rb);

        VecX grad_gx_lb, grad_gx_rb;
        VecU grad_gu_lb, grad_gu_rb;
        line_segment_constraint_lb_.Gradient(step, x, u, grad_gx_lb, grad_gu_lb);
        line_segment_constraint_rb_.Gradient(step, x, u, grad_gx_rb, grad_gu_rb);

        double grad_f_lb{0.0};
        double grad_f_rb{0.0};
        one_side_cost_.Gradient(g_lb, grad_f_lb);
        one_side_cost_.Gradient(g_rb, grad_f_rb);

        MatrixLXX hess_gxx_lb, hess_gxx_rb;
        MatrixLUU hess_guu_lb, hess_guu_rb;
        MatrixLXU hess_gxu_lb, hess_gxu_rb;
        line_segment_constraint_lb_.Hessian(step, x, u, hess_gxx_lb, hess_guu_lb, hess_gxu_lb);
        line_segment_constraint_rb_.Hessian(step, x, u, hess_gxx_rb, hess_guu_rb, hess_gxu_rb);

        double hess_f_lb{0.0};
        double hess_f_rb{0.0};
        one_side_cost_.Hessian(g_lb, hess_f_lb);
        one_side_cost_.Hessian(g_rb, hess_f_rb);

        lxx = w_plb_ * (hess_f_lb * grad_gx_lb * grad_gx_lb.transpose() +
                        grad_f_lb * hess_gxx_lb) +
              w_prb_ * (hess_f_rb * grad_gx_rb * grad_gx_rb.transpose() +
                        grad_f_rb * hess_gxx_rb);
        luu = w_plb_ * (hess_f_lb * grad_gu_lb * grad_gu_lb.transpose() +
                        grad_f_lb * hess_guu_lb) +
              w_prb_ * (hess_f_rb * grad_gu_rb * grad_gu_rb.transpose() +
                        grad_f_rb * hess_guu_rb);
        lxu = w_plb_ * (hess_f_lb * grad_gx_lb * grad_gu_lb.transpose() +
                        grad_f_lb * hess_gxu_lb) +
              w_prb_ * (hess_f_rb * grad_gx_rb * grad_gu_rb.transpose() +
                        grad_f_rb * hess_gxu_rb);
        return true;
    };

private:
    const double w_plb_;       // path left bound weight
    const double w_prb_;       // path right bound weight
    const uint32_t steps_num_; // total number of steps
//    const OneSideQuadraticCost one_side_quadratic_cost_;
    const OneSideCost one_side_cost_;
    const LateralOffsetConstraint<T, M, N> line_segment_constraint_lb_;
    const LateralOffsetConstraint<T, M, N> line_segment_constraint_rb_;
};


/*
  g(u) = a_ * u + b_[i] > 0
  represents u_min < u < u_max
  for u_min < u, g(u) = -u + u_min < 0
      u < u_max, g(u) = u - u_max < 0
 */
template<typename T, unsigned M, unsigned int N>
class ControlLimitConstraint final : public Constraint<T, M, N, Inequality> {
public:
    OCP_VARIABLES(T, M, N);
    ControlLimitConstraint(const CostTermConfig &config, const bool is_lb) {
        // g(u) = a_ * u + b_[i] < 0
        step_num_ = config.steps.size();
        if (is_lb) {
            a_ = -1;
            b_ = config.u_min;
        } else {
            a_ = 1;
            for (size_t i = 0; i < step_num_; i++) {
                b_.emplace_back(-config.u_max[i]);
            }
        }
    }

    virtual std::string GetName() const override {return std::string("ControlLimitConstraint");}

    virtual std::type_index GetTypeIndex() const override {
        return std::type_index(typeid(ControlLimitConstraint));
    }

    bool Evaluate(const int step,
                  const State &x,
                  const Control &u,
                  double &cost_val) const override {
        if (step >= step_num_) {
            cost_val = 0.0;
            return false;
        }
        cost_val = a_ * u(kDDkappa) + b_[step];
        return true;
    }

    bool Gradient(const int step, const State &x, const Control &u,
                  Eigen::Ref<VecX> lx, Eigen::Ref<VecU> lu) const override {
        lx.setZero();
        lu.setZero();
        if (step > b_.size())
            return false;
        lu(kDDkappa) = a_;
        return true;
    }

    bool Hessian(const int step, const State &x, const Control &u,
                 Eigen::Ref<MatrixLXX> lxx,
                 Eigen::Ref<MatrixLUU> luu,
                 Eigen::Ref<MatrixLXU> lxu) const override {
        lxx.setZero();
        luu.setZero();
        lxu.setZero();
        if (step >= step_num_)
            return false;
        return true;
    }

private:
    int a_;
    std::vector<double> b_;
    int step_num_;
};


/*
  J_ulimit = w_clb * f(g_clb(u)) + w_cub * f(g_cub(u))
  where f() is the penalty function, currently using one-sided quadratic
  function, and g_clb(u), g_cub(u) represents the control limit constraints
 */
template<typename T, unsigned M, unsigned int N>
class ControlConstraint final : public Constraint<T, M, N, Inequality> {
public:
    OCP_VARIABLES(T, M, N);
    ControlConstraint<T, M, N>(const CostTermConfig &config)
            : w_clb_(config.weights.at(WeightClb)),
              w_cub_(config.weights.at(WeightCub)),
              control_limit_constraint_lb_(config, true),
              control_limit_constraint_ub_(config, false) {
        Constraint<T, M, N, Inequality>::SetHorizon(config.steps.size());
    }

    ~ControlConstraint() = default;

    virtual std::string GetName() const override {
        std::string name("ControlConstraint");
        return name;
    }

    virtual std::type_index GetTypeIndex() const override {
        return std::type_index(typeid(ControlConstraint));
    }

    bool Evaluate(const int step,
                  const State &x,
                  const Control &u,
                  double &cost_val) const override {
        if (step >= Constraint<T, M, N, Inequality>::Horizon()) {
            cost_val = 0.0;
            return false;
        }

        // using one side quadratic cost f(g(u))
        double g_lb{0.0};
        double g_ub{0.0};
        control_limit_constraint_lb_.Evaluate(step, x, u, g_lb);
        control_limit_constraint_ub_.Evaluate(step, x, u, g_ub);

        double f_lb = one_side_cost_.Evaluate(g_lb);
        double f_ub = one_side_cost_.Evaluate(g_ub);

        cost_val = w_clb_ * f_lb + w_cub_ * f_ub;
        return true;
    };

    bool Gradient(const int step, const State &x, const Control &u,
                  Eigen::Ref<VecX> lx, Eigen::Ref<VecU> lu) const override {

        lx.setZero();
        lu.setZero();
        if (step >= Constraint<T, M, N, Inequality>::Horizon()) {
            return false;
        }

        double g_lb{0.0};
        double g_ub{0.0};
        control_limit_constraint_lb_.Evaluate(step, x, u, g_lb);
        control_limit_constraint_ub_.Evaluate(step, x, u, g_ub);

        VecX grad_gx_lb, grad_gx_ub;
        VecU grad_gu_lb, grad_gu_ub;
        control_limit_constraint_lb_.Gradient(step, x, u, grad_gx_lb, grad_gu_lb);
        control_limit_constraint_ub_.Gradient(step, x, u, grad_gx_ub, grad_gu_ub);

        double grad_f_lb{0.0};
        double grad_f_ub{0.0};
        one_side_cost_.Gradient(g_lb, grad_f_lb);
        one_side_cost_.Gradient(g_ub, grad_f_ub);

        lx = w_clb_ * grad_f_lb * grad_gx_lb + w_cub_ * grad_f_ub * grad_gx_ub;
        lu = w_clb_ * grad_f_lb * grad_gu_lb + w_cub_ * grad_f_ub * grad_gu_ub;
        return true;
    };

    bool Hessian(const int step, const State &x, const Control &u,
                 Eigen::Ref<MatrixLXX> lxx,
                 Eigen::Ref<MatrixLUU> luu,
                 Eigen::Ref<MatrixLXU> lxu) const override {
        lxx.setZero();
        luu.setZero();
        lxu.setZero();
        if (step >= Constraint<T, M, N, Inequality>::Horizon()) {
            return false;
        }

        double g_lb{0.0};
        double g_ub{0.0};
        control_limit_constraint_lb_.Evaluate(step, x, u, g_lb);
        control_limit_constraint_ub_.Evaluate(step, x, u, g_ub);

        VecX grad_gx_lb, grad_gx_ub;
        VecU grad_gu_lb, grad_gu_ub;
        control_limit_constraint_lb_.Gradient(step, x, u, grad_gx_lb, grad_gu_lb);
        control_limit_constraint_ub_.Gradient(step, x, u, grad_gx_ub, grad_gu_ub);

        double grad_f_lb{0.0};
        double grad_f_ub{0.0};
        one_side_cost_.Gradient(g_lb, grad_f_lb);
        one_side_cost_.Gradient(g_ub, grad_f_ub);

        MatrixLXX hess_gxx_lb, hess_gxx_ub;
        MatrixLUU hess_guu_lb, hess_guu_ub;
        MatrixLXU hess_gxu_lb, hess_gxu_ub;
        control_limit_constraint_lb_.Hessian(step, x, u, hess_gxx_lb, hess_guu_lb, hess_gxu_lb);
        control_limit_constraint_ub_.Hessian(step, x, u, hess_gxx_ub, hess_guu_ub, hess_gxu_ub);

        double hess_f_lb{0.0};
        double hess_f_ub{0.0};
        one_side_cost_.Hessian(g_lb, hess_f_lb);
        one_side_cost_.Hessian(g_ub, hess_f_ub);

        lxx = w_clb_ * (hess_f_lb * grad_gx_lb * grad_gx_lb.transpose() +
                        grad_f_lb * hess_gxx_lb) +
              w_cub_ * (hess_f_ub * grad_gx_ub * grad_gx_ub.transpose() +
                        grad_f_ub * hess_gxx_ub);
        luu = w_clb_ * (hess_f_lb * grad_gu_lb * grad_gu_lb.transpose() +
                        grad_f_lb * hess_guu_lb) +
              w_cub_ * (hess_f_ub * grad_gu_ub * grad_gu_ub.transpose() +
                        grad_f_ub * hess_guu_ub);
        lxu = w_clb_ * (hess_f_lb * grad_gx_lb * grad_gu_lb.transpose() +
                        grad_f_lb * hess_gxu_lb) +
              w_cub_ * (hess_f_ub * grad_gx_ub * grad_gu_ub.transpose() +
                        grad_f_ub * hess_gxu_ub);

        return true;
    };

private:
    const double w_clb_;       // control lower bound weight
    const double w_cub_;       // control upper bound weight
//    const OneSideQuadraticCost one_side_quadratic_cost_;
    const OneSideCost one_side_cost_;
    const ControlLimitConstraint<T, M, N> control_limit_constraint_lb_;
    const ControlLimitConstraint<T, M, N> control_limit_constraint_ub_;
};


/*
  g(u) = a_ * u + b_[i] > 0
  represents u_min < u < u_max
  for u_min < u, g(u) = -u + u_min < 0
      u < u_max, g(u) = u - u_max < 0
 */
template<typename T, unsigned M, unsigned N>
class HeadingTrackConstraint final : public Constraint<T, M, N, Inequality> {
public:
    OCP_VARIABLES(T, M, N);
    HeadingTrackConstraint(const CostTermConfig &config, const bool is_lb):
    yaw(config.yaw)
    {
        // g(u) = a_ * u + b_[i] < 0
        // -(u - ref) + u_min <= 0
        // (u - ref) -u_max <=0
        size_t num_steps = config.p_ref.size();
        if (is_lb) {
            a_ = -1;
            for (size_t t = 0; t < num_steps; t++) {
                b_.emplace_back(yaw[t] - config.yaw_diff_max);
            }
        } else {
            a_ = 1;
            for (size_t t = 0; t < num_steps; t++) {
                b_.emplace_back(-yaw[t] - config.yaw_diff_max);
            }
        }
        step_num_ = config.steps.size();
    }

    virtual std::type_index GetTypeIndex() const override {
        return std::type_index(typeid(HeadingTrackConstraint));
    }

    virtual std::string GetName() const override {return std::string("HeadingTrackConstraint");}

    bool Evaluate(const int step,
                  const State &x,
                  const Control &u,
                  double &cost_val) const override {
        if (step > step_num_) {
            cost_val = 0.0;
            return false;
        }
        cost_val = a_ * x(kTheta) + b_[step];
        return true;
    }

    bool Gradient(const int step, const State &x, const Control &u,
                  Eigen::Ref<VecX> lx, Eigen::Ref<VecU> lu) const override {
        lx.setZero();
        lu.setZero();
        if (step > step_num_)
            return false;
        lx(kTheta) = a_;
        return true;
    }

    bool Hessian(const int step, const State &x, const Control &u,
                 Eigen::Ref<MatrixLXX> lxx,
                 Eigen::Ref<MatrixLUU> luu,
                 Eigen::Ref<MatrixLXU> lxu) const override {
        lxx.setZero();
        luu.setZero();
        lxu.setZero();
        if (step > step_num_)
            return false;
        return true;
    }

private:
    int a_;
    vector<double> b_;
    vector<double> yaw;
    int step_num_;
};

template<typename T, unsigned M, unsigned N>
class HeadingConstraint final : public Constraint<T, M, N, Inequality> {
public:
    OCP_VARIABLES(T, M, N);
    HeadingConstraint(const CostTermConfig &config):
    w_clb_(config.weights.at(WeightClb)),
    w_cub_(config.weights.at(WeightCub)),
    yaw(config.yaw),
    heading_track_lb_(config, true),
    heading_track_ub_(config, false)
    {
        step_num_ = config.steps.size();
    }

    virtual std::type_index GetTypeIndex() const override {
        return std::type_index(typeid(HeadingConstraint));
    }

    virtual std::string GetName() const override {return std::string("HeadingConstraint");}

    bool Evaluate(const int step,
                  const State &x,
                  const Control &u,
                  double &cost_val) const override {
        if (step > step_num_) {
            cost_val = 0.0;
            return false;
        }
        // using one side quadratic cost f(g(u))
        double g_lb{0.0};
        double g_ub{0.0};
        heading_track_lb_.Evaluate(step, x, u, g_lb);
        heading_track_ub_.Evaluate(step, x, u, g_ub);

        double f_lb = one_side_cost_.Evaluate(g_lb);
        double f_ub = one_side_cost_.Evaluate(g_ub);

        cost_val = w_clb_ * f_lb + w_cub_ * f_ub;
        return true;
    }

    bool Gradient(const int step, const State &x, const Control &u,
                  Eigen::Ref<VecX> lx, Eigen::Ref<VecU> lu) const override {
        lx.setZero();
        lu.setZero();
        if (step > step_num_)
            return false;

        double g_lb{0.0};
        double g_ub{0.0};
        heading_track_lb_.Evaluate(step, x, u, g_lb);
        heading_track_ub_.Evaluate(step, x, u, g_ub);

        VecX grad_gx_lb, grad_gx_ub;
        VecU grad_gu_lb, grad_gu_ub;
        heading_track_lb_.Gradient(step, x, u, grad_gx_lb, grad_gu_lb);
        heading_track_ub_.Gradient(step, x, u, grad_gx_ub, grad_gu_ub);

        double grad_f_lb{0.0};
        double grad_f_ub{0.0};
        one_side_cost_.Gradient(g_lb, grad_f_lb);
        one_side_cost_.Gradient(g_ub, grad_f_ub);

        lx = w_clb_ * grad_f_lb * grad_gx_lb + w_cub_ * grad_f_ub * grad_gx_ub;
        lu = w_clb_ * grad_f_lb * grad_gu_lb + w_cub_ * grad_f_ub * grad_gu_ub;
        return true;
    }

    // (f(g))'' = (f'g')' = f''g'g' + f'g''
    bool Hessian(const int step, const State &x, const Control &u,
                 Eigen::Ref<MatrixLXX> lxx,
                 Eigen::Ref<MatrixLUU> luu,
                 Eigen::Ref<MatrixLXU> lxu) const override {
        lxx.setZero();
        luu.setZero();
        lxu.setZero();
        if (step > step_num_)
            return false;

        double g_lb{0.0};
        double g_ub{0.0};
        heading_track_lb_.Evaluate(step, x, u, g_lb);
        heading_track_ub_.Evaluate(step, x, u, g_ub);

        VecX grad_gx_lb, grad_gx_ub;
        VecU grad_gu_lb, grad_gu_ub;
        heading_track_lb_.Gradient(step, x, u, grad_gx_lb, grad_gu_lb);
        heading_track_ub_.Gradient(step, x, u, grad_gx_ub, grad_gu_ub);

        double grad_f_lb{0.0};
        double grad_f_ub{0.0};
        one_side_cost_.Gradient(g_lb, grad_f_lb);
        one_side_cost_.Gradient(g_ub, grad_f_ub);

        MatrixLXX hess_gxx_lb, hess_gxx_ub;
        MatrixLUU hess_guu_lb, hess_guu_ub;
        MatrixLXU hess_gxu_lb, hess_gxu_ub;
        heading_track_lb_.Hessian(step, x, u, hess_gxx_lb, hess_guu_lb, hess_gxu_lb);
        heading_track_ub_.Hessian(step, x, u, hess_gxx_ub, hess_guu_ub, hess_gxu_ub);

        double hess_f_lb{0.0};
        double hess_f_ub{0.0};
        one_side_cost_.Hessian(g_lb, hess_f_lb);
        one_side_cost_.Hessian(g_ub, hess_f_ub);

        lxx = w_clb_ * (hess_f_lb * grad_gx_lb * grad_gx_lb.transpose() +
                grad_f_lb * hess_gxx_lb) +
              w_cub_ * (hess_f_ub * grad_gx_ub * grad_gx_ub.transpose() +
                grad_f_ub * hess_gxx_ub);
        luu = w_clb_ * (hess_f_lb * grad_gu_lb * grad_gu_lb.transpose() +
                grad_f_lb * hess_guu_lb) +
              w_cub_ * (hess_f_ub * grad_gu_ub * grad_gu_ub.transpose() +
                grad_f_ub * hess_guu_ub);
        lxu = w_clb_ * (hess_f_lb * grad_gx_lb * grad_gu_lb.transpose() +
                grad_f_lb * hess_gxu_lb) +
              w_cub_ * (hess_f_ub * grad_gx_ub * grad_gu_ub.transpose() +
                grad_f_ub * hess_gxu_ub);
        return true;
    }

private:
    const double w_clb_;       // path left bound weight
    const double w_cub_;
    vector<double> yaw;
    int step_num_;
    HeadingTrackConstraint<T, M, N> heading_track_lb_;
    HeadingTrackConstraint<T, M, N> heading_track_ub_;
    OneSideCost one_side_cost_;
};

/*
  g(u) = a_ * u + b_[i] > 0
  represents u_min < u < u_max
  for u_min < u, g(u) = -u + u_min < 0
      u < u_max, g(u) = u - u_max < 0
 */
template<typename T, unsigned M, unsigned N>
class KappaLimitsConstraint final : public Constraint<T, M, N, Inequality> {
public:
    OCP_VARIABLES(T, M, N)
    KappaLimitsConstraint(const CostTermConfig &config, const bool is_lb) {
        steps_num_ = config.steps.size();
        if (is_lb) {
            a_ = -1;
            b_ = config.kappa_min;
        }
        else {
            a_ = 1;
            b_ = -config.kappa_max;
        }
    }
    ~KappaLimitsConstraint() = default;

    std::string GetName() const override {return std::string("KappaLimitsConstraint");}

    std::type_index GetTypeIndex() const override {
        return std::type_index(typeid(KappaLimitsConstraint));
    }

    bool Evaluate(const int step,
                  const State &x,
                  const Control &u,
                  double &cost_val) const override {
        if (step > steps_num_ ) {
            return false;
        }
        cost_val = a_ * x(kKappa) + b_;
        return true;
    }

    bool Gradient(const int step, const State &x, const Control &u,
                  Eigen::Ref<VecX> lx, Eigen::Ref<VecU> lu) const override {
        lx.setZero();
        lu.setZero();
        if (step > steps_num_) {
            return false;
        }
        lx(kKappa) = a_;
        return true;
    }

    bool Hessian(const int step, const State &x, const Control &u,
                 Eigen::Ref<MatrixLXX> lxx,
                 Eigen::Ref<MatrixLUU> luu,
                 Eigen::Ref<MatrixLXU> lxu) const override {
        lxx.setZero();
        luu.setZero();
        lxu.setZero();
        if (step > steps_num_) {
            return false;
        }
        return true;
    }

private:
    int a_;
    double b_;
    int steps_num_;
};

/*
  J_ulimit = w_clb * f(g_clb(u)) + w_cub * f(g_cub(u))
  where f() is the penalty function, currently using one-sided quadratic
  function, and g_clb(u), g_cub(u) represents the control limit constraints
 */
template<typename T, unsigned M, unsigned N>
class KappaConstraint final : public Constraint<T, M, N, Inequality> {
public:
    OCP_VARIABLES(T, M, N)
    KappaConstraint(const CostTermConfig &config):
            w_clb_(config.weights.at(WeightClb)),
            w_cub_(config.weights.at(WeightCub)),
            kappa_limit_constraint_lb_(config, true),
            kappa_limit_constraint_ub_(config, false)
    {
        steps_num_ = config.steps.size();
    }

    ~KappaConstraint() = default;

    std::string GetName() const override {return std::string("KappaConstraint");}

    std::type_index GetTypeIndex() const override {
        return std::type_index(typeid(KappaConstraint));
    }

    bool Evaluate(const int step,
                  const State &x,
                  const Control &u,
                  double &cost_val) const override {
        if (step > steps_num_) {
            cost_val = 0.0;
            return false;
        }

        double g_lb{0.0};
        double g_ub{0.0};
        kappa_limit_constraint_lb_.Evaluate(step, x, u, g_lb);
        kappa_limit_constraint_ub_.Evaluate(step, x, u, g_ub);

        double f_lb = one_side_cost_.Evaluate(g_lb);
        double f_ub = one_side_cost_.Evaluate(g_ub);

        cost_val = w_clb_ * f_lb + w_cub_ * f_ub;
        return true;
    };

    bool Gradient(const int step, const State &x, const Control &u,
                  Eigen::Ref<VecX> lx, Eigen::Ref<VecU> lu) const override {
        lx.setZero();
        lu.setZero();
        if (step > steps_num_) {
            return false;
        }

        double g_lb{0.0};
        double g_ub{0.0};
        kappa_limit_constraint_lb_.Evaluate(step, x, u, g_lb);
        kappa_limit_constraint_ub_.Evaluate(step, x, u, g_ub);

        VecX grad_gx_lb, grad_gx_ub;
        VecU grad_gu_lb, grad_gu_ub;
        kappa_limit_constraint_lb_.Gradient(step, x, u, grad_gx_lb, grad_gu_lb);
        kappa_limit_constraint_ub_.Gradient(step, x, u, grad_gx_ub, grad_gu_ub);

        double grad_f_lb{0.0};
        double grad_f_ub{0.0};
        one_side_cost_.Gradient(g_lb, grad_f_lb);
        one_side_cost_.Gradient(g_ub, grad_f_ub);

        lx = w_clb_ * grad_f_lb * grad_gx_lb + w_cub_ * grad_f_ub * grad_gx_ub;
        lu = w_clb_ * grad_f_lb * grad_gu_lb + w_cub_ * grad_f_ub * grad_gu_ub;
        return true;
    };

    bool Hessian(const int step, const State &x, const Control &u,
                 Eigen::Ref<MatrixLXX> lxx,
                 Eigen::Ref<MatrixLUU> luu,
                 Eigen::Ref<MatrixLXU> lxu) const override {
        lxx.setZero();
        luu.setZero();
        lxu.setZero();
        if (step > steps_num_)
            return false;

        double g_lb{0.0};
        double g_ub{0.0};
        kappa_limit_constraint_lb_.Evaluate(step, x, u, g_lb);
        kappa_limit_constraint_ub_.Evaluate(step, x, u, g_ub);

        VecX grad_gx_lb, grad_gx_ub;
        VecU grad_gu_lb, grad_gu_ub;
        kappa_limit_constraint_lb_.Gradient(step, x, u, grad_gx_lb, grad_gu_lb);
        kappa_limit_constraint_ub_.Gradient(step, x, u, grad_gx_ub, grad_gu_ub);

        double grad_f_lb{0.0};
        double grad_f_ub{0.0};
        one_side_cost_.Gradient(g_lb, grad_f_lb);
        one_side_cost_.Gradient(g_ub, grad_f_ub);

        MatrixLXX hess_gxx_lb, hess_gxx_ub;
        MatrixLUU hess_guu_lb, hess_guu_ub;
        MatrixLXU hess_gxu_lb, hess_gxu_ub;
        kappa_limit_constraint_lb_.Hessian(step, x, u, hess_gxx_lb, hess_guu_lb, hess_gxu_lb);
        kappa_limit_constraint_ub_.Hessian(step, x, u, hess_gxx_ub, hess_guu_ub, hess_gxu_ub);

        double hess_f_lb{0.0};
        double hess_f_ub{0.0};
        one_side_cost_.Hessian(g_lb, hess_f_lb);
        one_side_cost_.Hessian(g_ub, hess_f_ub);

        lxx = w_clb_ * (hess_f_lb * grad_gx_lb * grad_gx_lb.transpose() +
                        grad_f_lb * hess_gxx_lb) +
              w_cub_ * (hess_f_ub * grad_gx_ub * grad_gx_ub.transpose() +
                        grad_f_ub * hess_gxx_ub);
        luu = w_clb_ * (hess_f_lb * grad_gu_lb * grad_gu_lb.transpose() +
                        grad_f_lb * hess_guu_lb) +
              w_cub_ * (hess_f_ub * grad_gu_ub * grad_gu_ub.transpose() +
                        grad_f_ub * hess_guu_ub);
        lxu = w_clb_ * (hess_f_lb * grad_gx_lb * grad_gu_lb.transpose() +
                        grad_f_lb * hess_gxu_lb) +
              w_cub_ * (hess_f_ub * grad_gx_ub * grad_gu_ub.transpose() +
                        grad_f_ub * hess_gxu_ub);

        return true;
    };

private:
    const double w_clb_;       // control lower bound weight
    const double w_cub_;       // control upper bound weight
    const OneSideCost one_side_cost_;
    const KappaLimitsConstraint<T, M, N> kappa_limit_constraint_lb_;
    const KappaLimitsConstraint<T, M, N> kappa_limit_constraint_ub_;
    int steps_num_;
};

#endif //ALILQR_BASIC_CONSTRAINT_H
