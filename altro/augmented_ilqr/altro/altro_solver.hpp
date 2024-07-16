//
// Created by 廖田志浩 on 2024/6/25.
//

#ifndef ALILQR_ALTRO_SOLVER_HPP
#define ALILQR_ALTRO_SOLVER_HPP

#include "alilqr_solver.h"
#include "cost_calc.h"
#include "../clock.hpp"
#include "qdldl_interface.h"
#include <Eigen/Dense>
#include <iostream>

using std::cout;
using std::endl;

enum class JacSize {
    None,
    Gradient,
    Jacobian
};

template<typename...Args>
void print(const Args&... args) {
    using arr = int[];
    (void) arr{ 0, (std::cout << args << ", ", 0)...};
    cout << "\n";
    return;
}

template<typename T>
std::ostream& operator<<(std::ostream& os, const std::vector<T>& vec)
{
    for (const auto& element : vec)
        os << element << ' ';
    os << "\n";
    return os;
}

template<typename T, unsigned M, unsigned N>
class ALTROSolver {
public:
    using CostUnionType = CostUnion<T, M, N>;
    template<class ConsType>
    using ConstraintValues = typename CostUnionType::template ConstraintValues<ConsType>;
    constexpr static double kDefaultConstraintEpsilon = 1.0e-6;
    constexpr static double kTol = 1.0e-6;
    constexpr static double kConvRateTol = 1.1;
    constexpr static int kMaxLineSearchIter = 20;
    constexpr static int kMaxInnerIter = 10;
    constexpr static int kMaxOuterIter = 10;
    constexpr static double kZeroTol = 1.0e-6;
    constexpr static double kPrimalFactor = 1.0e-8;
public:
    OCP_VARIABLES(T, M, N)
    explicit ALTROSolver(std::unique_ptr<OCPInterface<T, M, N>> &&ocp_interface,
                        const vector<double> &steps)
            : step_sizes_(steps)
    {
//        problem_.reset(std::move(solver->ProblemPtr()));
        steps_num_ = steps.size();
        solver_.reset(new ALILQRSolver<T, M, N>(std::move(ocp_interface), steps));
    };

    virtual ~ALTROSolver() = default;

    void Solve(const State &x0, const Controls &u0,
               States &x_res_seq, Controls &u_res_seq) {
        StopWatch timer;
        timer.start();
        States alilqr_x;
        Controls alilqr_u;
        solver_->Solve(x0, u0, alilqr_x, alilqr_u);
        States x_old = alilqr_x;
        Controls u_old = alilqr_u;
        cout << "AL-iLQR Solver Status : ";
        solver_->ShowSolverState();
        cout << "AL-iLQR max_violation : " << solver_->GetMaxViolation() << "\n";
//        vector<vector<double>> all_violations;
        StopWatch proj_timer;
        proj_timer.start();
        double proj_consumption = 0.0;
        for (int t = 0; t < steps_num_; ++t) {
            State x = alilqr_x[t];
            Control u = alilqr_u[t];
            Projection(t, x, u, alilqr_x[t], alilqr_u[t]);
        }
        proj_consumption += proj_timer.elapsed_microseconds() / 1.0e3;
        x_res_seq = alilqr_x;
        u_res_seq = alilqr_u;
//        std::unordered_map<std::string, vector<double>> violation_record;
//        MatrixXd Violations = CalcAllViolations(x_res_seq, u_res_seq, violation_record);
//        cout << "Max Violation is " << Violations.lpNorm<Eigen::Infinity>() << "\n";
//        for (const auto &ele : violation_record) {
//            cout << ele.first << ": " << ele.second;
//        }
        solver_->Problem().GetCostUnionPtr()->EvaluateConstraints(x_res_seq, u_res_seq);
        cout << "ALTRO solver max violation : " << GetMaxViolation() << endl;
        double d = timer.elapsed_microseconds() / 1.0e3;
        cout << "altro Projection Process : " << proj_consumption << " ms\n";
        cout << "altro solver consumes : " << d << " ms\n";
    }

    void SetUpdateConfig(const bool update) {
        solver_->SetUpdateConfig(update);
    }

    MatrixXd CalcAllViolations(States &x, Controls &u,
                               std::unordered_map<std::string, vector<double>> &violation_record) const {
        vector<double> violations;
        int rows = 0;
        for (const auto &cons_ptr : solver_->Problem().GetCostUnionPtr()->GetImmutableEqConstraints()) {
            vector<double> vio;
            cout << cons_ptr->GetName() << endl;
            for (int t = 0; t < steps_num_; ++t) {
                bool success = cons_ptr->Evaluate(t, x[t], u[t]);
                if (success) {
                    double violation = cons_ptr->ConsVal();
                    violations.emplace_back(violation);
                    vio.emplace_back(violation);
                }
            }
            std::string name =
                    solver_->Problem().GetCostUnionPtr()->GetTypeNames().at(cons_ptr->GetTypeIndex());
            violation_record[name] = vio;
            ++rows;
        }

        for (const auto &cons_ptr : solver_->Problem().GetCostUnionPtr()->GetImmutableIneqConstraints()) {
            vector<double> vio;
            cout << cons_ptr->GetName() << endl;
            for (int t = 0; t < steps_num_; ++t) {
                bool success = cons_ptr->Evaluate(t, x[t], u[t]);
                if (success) {
                    double violation = cons_ptr->ConsVal();
                    violations.emplace_back(violation);
                    vio.emplace_back(violation);
                }
            }
            std::string name =
                    solver_->Problem().GetCostUnionPtr()->GetTypeNames().at(cons_ptr->GetTypeIndex());
            violation_record[name] = vio;
            ++rows;
        }

        Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
        Violations(violations.data(), rows, steps_num_);
        return Violations;
    }

private:
    void GetCostHessianAt(const int step, const State &x, const Control &u) {
        MatrixLXX lxx;
        MatrixLUU luu;
        MatrixLXU lxu;
        t_cost_hessian_.setZero();
        for (const auto &ct : solver_->Problem().GetCostUnionPtr()->GetCostTerms()) {
            MatrixLXX cur_lxx;
            MatrixLUU cur_luu;
            MatrixLXU cur_lxu;
            ct->Hessian(step, x, u, cur_lxx, cur_luu, cur_lxu);
//            cout << ct->GetName() << endl;
//            cout << "cur_lxx : " << endl << cur_lxx << endl;
//            cout << "cur_luu : " << endl << cur_luu << endl;
//            cout << "cur_lxu : " << endl << cur_lxu << endl;
            lxx += cur_lxx;
            luu += cur_luu;
            lxu += cur_lxu;
        }
        t_cost_hessian_ << lxx,              lxu,
                           lxu.transpose(),  luu;
        MatrixHZ primal_regularization;
        primal_regularization.setIdentity();
        t_cost_hessian_ += primal_regularization * kPrimalFactor;
//        cout << "t_cost_hessian" << t_cost_hessian_ << endl;
    }

    Eigen::MatrixXd GetKKTMatrix(const MatrixXd &D) {
        const int cj_rows = D.rows();
//        Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> zero_mat;
        Eigen::MatrixXd zero_mat(cj_rows, cj_rows);
        zero_mat.setZero();
//        Eigen::MatrixXd<T, M + N + cj_rows, M + N + cj_rows> KKT;
        Eigen::MatrixXd KKT(M + N + cj_rows, M + N + cj_rows);
        KKT.setZero();
        KKT << t_cost_hessian_, D.transpose(),
               D,               zero_mat;
        return KKT;
    }

    void QDLDLSolver(const MatrixXd &KKT, const MatrixXd &d, State &x, Control &u) {
        const QDLDL_int An = KKT.rows();
        vector<long long> Ap{0};
        vector<long long> Ai;
        vector<double> Ax;
        int ap = 0;
        for (int i = 0; i < KKT.cols(); ++i) {
            int ai = 0;
            for (int j = 0; j <= i; ++j) {
                if (std::fabs(KKT(j, i)) > kZeroTol) {
                    ++ai;
                    Ai.emplace_back(j);
                    Ax.emplace_back(KKT(j, i));
                }
            }
            ap += ai;
            Ap.emplace_back(ap);
        }
        Eigen::VectorXd b(KKT.rows());
        b.setZero();
        for (int i = 0; i < d.rows(); ++i) {
            b(M + N + i) = -d(i);
        }
//        vector<double> b(An, 0.0);
//        for (int i = 0; i < d.rows(); ++i) {
//            b[M + N + i] = -d(i);
//        }

//        cout << KKT << endl;
//        cout << "An : " << An << endl;
//        cout << "Ap size : " << Ap.size() << endl;
//        std::for_each(Ap.begin(), Ap.end(), [](const auto &num){cout << num << " ";});
//        cout << endl;
//        cout << "Ai size : " << Ai.size() << endl;
//        std::for_each(Ai.begin(), Ai.end(), [](const auto &num){cout << num << " ";});
//        cout << endl;
//        cout << "Ax size : " << Ax.size() << endl;
//        std::for_each(Ax.begin(), Ax.end(), [](const auto &num){cout << num << " ";});
//        cout << endl;
//        cout << "b size : " << b.size() << endl;
//        std::for_each(b.begin(), b.end(), [](const auto &num){cout << num << " ";});
//        cout << endl;
        vector<double> rst = QDLDLSolve(An, Ap.data(), Ai.data(), Ax.data(), b.data());
//        Eigen::VectorXd z_ = KKT.lu().solve(b).head(M + N);
//        cout << "eigen rst : " << z_.transpose() << endl;
//        std::for_each(rst.begin(), rst.end(), [](const double &num){cout << num << " ";});
//        cout << "\n";
        Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, 1>> z(rst.data(), M + N);


//        cout << "qdldl end\n";
        x = z.head(M);
        u = z.tail(N);
//        cout << "qdldl solve successfully\n";
    }

    /* in this ocp, inequality constraints are defined as g(x) < 0
     * active set is defined as g(x) < kDefaultConstraintEpsilon
     * true means that the size of Jacobian Matrix is larger than 1
     * false means that it is a grdient vector
     * */
    JacSize GetActiveConstraintsGradientAt(const int step, const State &x, const Control &u,
                                           MatrixXd &D, MatrixXd &d) {
//        cout << step << endl;
        VecX lx;
        VecU lu;
        lx.setZero();
        lu.setZero();
        vector<double> violations;
        MatrixZs JacVec;
        int index = 0;
//        eq_active_sets_.clear();
        ineq_active_sets_.clear();
        index = 0;
        for (auto &cons_ptr : solver_->Problem().GetCostUnionPtr()->GetImmutableIneqConstraints()) {
            bool success = cons_ptr->Evaluate(step, x, u);
            if (success) {
                double cons_val = cons_ptr->ConsVal();
                if (cons_val > kDefaultConstraintEpsilon) {
                    violations.emplace_back(cons_val);
                    cons_ptr->Gradient(step, x, u);
                    VecX cur_lx = cons_ptr->GradientX();
                    VecU cur_lu = cons_ptr->GradientU();
                    MatrixZ gz;
                    gz << cur_lx, cur_lu;
                    JacVec.emplace_back(gz);
                    ineq_active_sets_.emplace_back(index);
                    index++;
                }
            }
        }
        int rows = JacVec.size();
        if (rows == 0)
            return JacSize::None;
        D = Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, M + N, Eigen::RowMajor>>(
                reinterpret_cast<T*>(JacVec.data()), rows, M + N);

        d = Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, 1>>(violations.data(), rows, 1);
        if (D.rows() <= 1)
            return JacSize::Gradient;
        return JacSize::Jacobian;
    }

    void UpdateConstraintValues(const int step, const State &x, const Control &u,
                                MatrixXd& d) {
        vector<double> violations;
        for (const int &ind : eq_active_sets_) {
            bool success = solver_->Problem().GetCostUnionPtr()->GetImmutableEqConstraints()[ind]->Evaluate(step, x, u);
            if (success) {
                double cons_val = solver_->Problem().GetCostUnionPtr()->GetImmutableEqConstraints()[ind]->ConsVal();
                violations.emplace_back(cons_val);
            }
        }

        for (const int &ind : ineq_active_sets_) {
            bool success = solver_->Problem().GetCostUnionPtr()->GetImmutableIneqConstraints()[ind]->Evaluate(step, x, u);
            if (success) {
                double cons_val = solver_->Problem().GetCostUnionPtr()->GetImmutableIneqConstraints()[ind]->ConsVal();
                violations.emplace_back(cons_val);
            }
        }
        d = Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, 1>>(violations.data(), violations.size(), 1);
//        cout << "d=\n" << d << endl;
    }

//    Eigen::MatrixXd PseudoInverse(const Eigen::MatrixXd& mat, double tolerance = 1e-6) {
//        Eigen::JacobiSVD<Eigen::MatrixXd> svd(mat, Eigen::ComputeThinU | Eigen::ComputeThinV);
//        Eigen::VectorXd singularValues = svd.singularValues();
//        Eigen::MatrixXd singularValuesInv(mat.cols(), mat.rows());
//        singularValuesInv.setZero();
//
//        for (int i = 0; i < singularValues.size(); ++i) {
//            if (singularValues(i) > tolerance) {
//                singularValuesInv(i, i) = 1.0 / singularValues(i);
//            }
//        }
//
//        return svd.matrixV() * singularValuesInv * svd.matrixU().transpose();
//    }

    void Projection(const int step, State x, Control u,
                    State &altro_x, Control &altro_u) {
        GetCostHessianAt(step, x, u);
//        Eigen::LLT<MatrixXd> llt(t_cost_hessian_);
//        if (llt.info() != Eigen::Success) {
//            cout << "fail to QR decomposition\n";
//            return;
//        }
        MatrixXd D;
        MatrixXd d;
        JacSize linearized_active_cons =
                GetActiveConstraintsGradientAt(step, x, u, D, d);
//        cout << D << endl << endl;
        if (linearized_active_cons == JacSize::None)
            return;
        MatrixXd KKT = GetKKTMatrix(D);
        if (!KKT.allFinite() or KKT.array().isNaN().any())
            return;

        /*----------------- use Eigen to solve KKT equation -----------------*/
        Eigen::VectorXd b(KKT.rows());
        b.setZero();
        for (int i = 0; i < d.rows(); ++i) {
            b(M + N + i) = -d(i);
        }

        Eigen::VectorXd z = KKT.lu().solve(b).head(M + N);
        altro_x = x + z.head(M);
        altro_u = u + z.tail(N);

        /*----------------- use qdldl solver to solve KKT equation -----------------*/
//        State delta_x;
//        Control delta_u;
//        QDLDLSolver(KKT, d, delta_x, delta_u);
//        altro_x = x + delta_x;
//        altro_u = u + altro_u;
    }

//    void Projection(const int step, State x, Control u,
//                    State &polished_x, Control &polished_u) {
//        GetCostHessianAt(step, x, u);
//        Eigen::LLT<MatrixXd> llt(t_cost_hessian_);
//        MatrixXd H_inv;
//        if (llt.info() == Eigen::Success) {
//            H_inv = t_cost_hessian_.inverse();
//        }
//        else {
////            H_inv = t_cost_hessian_.completeOrthogonalDecomposition().pseudoInverse();
//            H_inv = PseudoInverse(t_cost_hessian_);
//        }
//        if (!H_inv.allFinite() or H_inv.array().isNaN().any() or H_inv.sum() < kTol) {
////            cout << "fail to calc Inverse\n";
//            return;
//        }
//        int loop = 0;
//        double v = std::numeric_limits<double>::epsilon();
//        while (v > kTol) {
//            MatrixXd D;
//            MatrixXd d;
//            JacSize linearized_active_cons = GetActiveConstraintsGradientAt(step, x, u, D, d);
////            cout << "H_inv : \n" << H_inv << endl;
//
//            if (linearized_active_cons == JacSize::None) {
////                cout << "No active constraints at step " << step << " found\n";
//                return;
//            }
//            MatrixXd mat = D * H_inv * D.transpose();
//            if (mat.sum() < kTol)
//                return;
//
//            MatrixXd S;
//            S.setZero();
//            MatrixXd S_inv;
//            double S_, S_inv_;
//            if (linearized_active_cons == JacSize::Jacobian) {
//                S = mat.llt().matrixL();
//                Eigen::LLT<MatrixXd> llt(S);
//                if (llt.info() == Eigen::Success) {
//                    S_inv = S.inverse();
//                }
//                else {
//                    S_inv = PseudoInverse(S);
//                }
//
//                if (!S_inv.allFinite() or S_inv.array().isNaN().any()) {
////                        cout << "Failed to calc inverse of Matrix S\n";
//                    return;
//                }
//            }
//            else if (linearized_active_cons == JacSize::Gradient){
//                S_ = std::sqrt(mat(0));
//                S_inv_ = 1.0 / S_;
////                cout << "S_ = " << S_ << endl;
//            }
//
//            v = d.lpNorm<Eigen::Infinity>();
//            if (v < kTol) {
//                break;
//            }
//
//            int inner_loop = 0;
//
//            MatrixXd delta_z;
//            double r = std::numeric_limits<double>::infinity();
//            while (v > kTol and r > kConvRateTol) {
//                // line search
//                double alpha = 1.0;
//                double gamma = 0.5;
//                if (linearized_active_cons == JacSize::Jacobian)
//                    delta_z = H_inv * D.transpose() * (S_inv * S_inv.transpose() * d);
//                else if (linearized_active_cons == JacSize::Gradient)
//                    delta_z = H_inv * D.transpose() * (S_inv_ * S_inv_ * d);
////                cout << "delta_z: \n" << delta_z << endl;
//                double v0;
//                State x_n;
//                Control u_n;
//                bool flag = false;
//                for (int i = 0; i < kMaxLineSearchIter; ++i) {
////                    cout << "line search " << i << endl;
//                    x_n = x + alpha * delta_z.topRows(M);
//                    u_n = u + alpha * delta_z.bottomRows(N);
//                    MatrixXd d_;
//                    UpdateConstraintValues(step, x_n, u_n, d_);
//                    v0 = d_.lpNorm<Eigen::Infinity>();
////                    cout << "v0=" << v0 << " v=" << v << endl;
//                    if (v0 < v) {
//                        cout << "line search successfully\n";
//                        x = x_n;
//                        u = u_n;
//                        flag = true;
//                        break;
//                    }
//                    alpha *= gamma;
//                }
//                // end line search
//                r = std::log(v0) / std::log(v);
//                if (flag) {
//                    v = v0;
//                    break;
//                }
////                if (v < kTol or r < kConvRateTol) {
////                    cout << "line search successfully\n";
//////                    cout << "r=" << r << endl;
//////                    cout << "v=" << v << endl;
////                    x = x_n;
////                    u = u_n;
////                    break;
////                }
//                inner_loop++;
//                if (inner_loop > kMaxInnerIter)
//                    break;
//            }
//            loop++;
//            if (loop > kMaxOuterIter)
//                break;
//        }
//        polished_x = x;
//        polished_u = u;
//    }

    const double GetMaxViolation() const {return solver_->Problem().GetCostUnionPtr()->GetMaxViolation();}

private:
//    std::unique_ptr<OCPInterface<T, M, N>> problem_;
    std::unique_ptr<ALILQRSolver<T, M, N>> solver_;
    std::vector<double> step_sizes_;
    MatrixHZ t_cost_hessian_;
    int steps_num_;
//    CostUnion<T, M, N> solver_->ProblemPtr()->GetCostUnionPtr();
    vector<int> eq_active_sets_;
    vector<int> ineq_active_sets_;
};


#endif //ALILQR_ALTRO_SOLVER_HPP
