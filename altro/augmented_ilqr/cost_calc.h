//
// Created by 廖田志浩 on 2024/5/30.
//

#ifndef CILQR_COST_CALC_H
#define CILQR_COST_CALC_H

#include "cost_function.hpp"
#include "cost_term_config.hpp"
#include "constraint_values.hpp"
#include "../frenet_coordinate/frenet_coordinate_system.h"
#include <iostream>

using std::cout;
using std::endl;

template<typename T, unsigned int M, unsigned int N>
class CostUnion {
public:
    OCP_VARIABLES(T, M, N)
    using CostFuncs = std::vector<CostFuncPtr<T, M, N>>;

    template<class ConsType>
    using ConstraintValues = std::vector<ConstraintValuePtr<T, M, N, ConsType>>;

    static constexpr int dim = Eigen::Dynamic;

    template<int size>
    using VectorNd = Eigen::Matrix<T, size, 1>;

public:
    explicit CostUnion(const CostTermConfig & config): config_(config),
    horizon_(config.steps.size())
    {
        FrenetCoordinateSystemParameters fcs_params;
        fcs_params.Init();
        fcs_.reset(new FrenetCoordinateSystem(config.ref_x, config.ref_y, fcs_params));
        num_of_eq_cons_ = 0;
        num_of_ineq_cons_ = 0;
    };

    CostUnion &operator=(const CostUnion&) = delete;
    CostUnion(const CostUnion&) = delete;

    virtual ~CostUnion() = default;

    void SetHorizon(const int horizon) {horizon_ = horizon;}

    double Evaluate(const int step,
                    const State &x,
                    const Control &u) const;

    void CalcCostGradient(const int step,
                          const State &x,
                          const Control &u,
                          Eigen::Ref<VecX> lx,
                          Eigen::Ref<VecU> lu) const;

    void CalcCostHessian(const int step, const State &x, const Control &u,
                         Eigen::Ref<MatrixLXX> lxx,
                         Eigen::Ref<MatrixLUU> luu,
                         Eigen::Ref<MatrixLXU> lxu) const;

    void CalcAugLagConstraintGradient(const int step, const State &x, const Control &u,
                                      Eigen::Ref<VecX> lx, Eigen::Ref<VecU> lu) const;

    void CalcAugLagConstraintHessian(const int step, const State &x, const Control &u,
                                     Eigen::Ref<MatrixLXX> cxx,
                                     Eigen::Ref<MatrixLUU> cuu,
                                     Eigen::Ref<MatrixLXU> cxu) const;

    void CalcTerminalCostToGo(const State &x);

    double GetMaxViolation() {
        int num_of_cons = GetNumOfConstraints();
        violations_.setZero(num_of_cons);
        for (int i = 0; i < num_of_cons; ++i) {
            if (i < num_of_eq_cons_) {
                violations_(i) = eqs_[i]->GetMaxViolation();
                cout << eqs_[i]->GetName() << ": " << violations_(i) << "\n";
            }
            else {
                int j = i - num_of_eq_cons_;
                violations_(i) = ineqs_[j]->GetMaxViolation();
                cout << ineqs_[j]->GetName() << ": " << violations_(i) << "\n";
            }
        }
        cout << "\n";
        return violations_.template lpNorm<Eigen::Infinity>();
    };

    void EvaluateConstraints(const States &x, const Controls &u);

    VecX GetFinalCostToGoGradient() const {return pN;}

    MatrixLXX GetFinalCostToGoHessian() const {return PN;}

    void AddCostTerm(CostFuncPtr<T, M, N> &&cost_term) {
        type_names_[cost_term->GetTypeIndex()] = cost_term->GetName();
        std::cout << cost_term->GetName() << " is loaded\n";
        cost_terms_.emplace_back(std::move(cost_term));
    }

    void AddEqConstraint(ConstraintValuePtr<T, M, N, Equality> &&constraint_value) {
        type_names_[constraint_value->GetTypeIndex()] = constraint_value->GetName();
        std::cout << constraint_value->GetName() << " is loaded\n";
        eqs_.emplace_back(std::move(constraint_value));
        ++num_of_eq_cons_;
    }

    void AddIneqConstraint(ConstraintValuePtr<T, M, N, Inequality> &&constraint_value) {
        type_names_[constraint_value->GetTypeIndex()] = constraint_value->GetName();
        std::cout << constraint_value->GetName() << " is loaded\n";
        ineqs_.emplace_back(std::move(constraint_value));
        ++num_of_ineq_cons_;
    }

    int GetNumOfConstraints() const { return num_of_eq_cons_ + num_of_ineq_cons_;}

    TypeNames GetTypeNames() const {return type_names_;}

    void UpdateConfig(const States &x_res_seq) {
        double v0 = config_.velocity.front();
        vector<Vector2d> first_bound = config_.p_bound.front();
        vector<Vector2d> last_bound = config_.p_bound.back();
        config_.p_bound.clear();
        config_.p_ref.clear();
        config_.yaw.clear();
        config_.ref_x.clear();
        config_.ref_y.clear();
        config_.right_safe_dist.clear();
        config_.left_safe_dist.clear();
        int size = x_res_seq.size();
        config_.p_bound.resize(size);
        config_.p_ref.resize(size);
        config_.yaw.resize(size);
        double start_s = config_.start_s;
        double lane_width = config_.lane_width;
        double buffer_dis = config_.buffer_dis;
        Vec2d prev(x_res_seq[0](0), x_res_seq[0](1));
        for (int t = 0; t < size; ++t) {
            double x = x_res_seq[t](0);
            double y = x_res_seq[t](1);
            Vec2d frenet_pt;
            (void) fcs_->CartCoord2FrenetCoord(Vec2d(x, y), frenet_pt);
            double s = frenet_pt.x();
            Point2D ref_point = fcs_->GetRefCurvePoint(s);
            Vector2d ref;
            ref << ref_point.x, ref_point.y;
            config_.p_ref.emplace_back(ref);
            config_.ref_x.emplace_back(ref_point.x);
            config_.ref_y.emplace_back(ref_point.y);
            double yaw = fcs_->GetRefCurveHeading(s);
            config_.yaw[t] = yaw;
            double k = fcs_->GetRefCurveCurvature(s);
//            double v = std::min(config_.vel_max, std::min(config_.velocity[t], 1.0 / k));
//            if (t > 0)
//                config_.velocity[t] = v;
            auto min_iter = std::upper_bound(config_.bound_s.begin(), config_.bound_s.end(), s);
            int bound_index = std::distance(config_.bound_s.begin(), min_iter);
            if (bound_index == 0)
                config_.p_bound[t] = first_bound;
            else {
                vector<Vector2d> p_bound{config_.left_boundary[bound_index - 1],
                                         config_.left_boundary[bound_index],
                                         config_.right_boundary[bound_index - 1],
                                         config_.right_boundary[bound_index]};
                config_.p_bound[t] = p_bound;
            }
            Vector2d left_bound_point = config_.p_bound[t][0];
            Vector2d right_bound_point = config_.p_bound[t][2];
            Vec2d left_frenet_pt, right_frenet_pt;
            (void) fcs_->CartCoord2FrenetCoord(Vec2d(left_bound_point(0), left_bound_point(1)), left_frenet_pt);
            (void) fcs_->CartCoord2FrenetCoord(Vec2d(right_bound_point(0), right_bound_point(1)), right_frenet_pt);
            double l_l = left_frenet_pt.y();
            double r_l = right_frenet_pt.y();
            double left_safe_buffer = l_l > r_l ? config_.max_safe_buffer : config_.min_safe_buffer;
            double right_safe_buffer = r_l > l_l ? config_.max_safe_buffer : config_.min_safe_buffer;
            config_.left_safe_dist.emplace_back(config_.half_veh_length + left_safe_buffer);
            config_.right_safe_dist.emplace_back(config_.half_veh_length + right_safe_buffer);
        }

        for(const auto &ct : cost_terms_) {
            ct->UpdateConfig(config_);
        }

        for (const auto &cons_ptr : eqs_) {
            cons_ptr->UpdateConfig(config_);
        }

        for (const auto &cons_ptr : ineqs_) {
            cons_ptr->UpdateConfig(config_);
        }
    };

    ConstraintValues<Equality> &GetEqConstraints() {return eqs_;}

    ConstraintValues<Inequality> &GetIneqConstraints() {return ineqs_;}

    const ConstraintValues<Equality> &GetImmutableEqConstraints() const {return eqs_;}

    const ConstraintValues<Inequality> &GetImmutableIneqConstraints() const {return ineqs_;}

    const CostFuncs &GetCostTerms() const {return cost_terms_;}

    const CostTermConfig &GetConfig() const {return config_;}
private:
    CostTermConfig config_;
    CostFuncs cost_terms_;
    ConstraintValues<Equality> eqs_;
    ConstraintValues<Inequality> ineqs_;
    VectorNd<dim> violations_;
    VecX pN;
    MatrixLXX PN;
    int horizon_;
    double max_violation_ = std::numeric_limits<double>::max();
    TypeNames type_names_;
    vector<double> max_penalty_;
    std::unique_ptr<FrenetCoordinateSystem> fcs_;
    int num_of_eq_cons_;
    int num_of_ineq_cons_;
};

template<typename T, unsigned M, unsigned N>
using CostUnionPtr = std::unique_ptr<CostUnion<T, M, N>>;

#endif //CILQR_COST_CALC_H
