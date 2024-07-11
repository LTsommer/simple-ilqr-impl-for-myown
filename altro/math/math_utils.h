#ifndef MODULES_PLANNING_MATH_UTIL_H_
#define MODULES_PLANNING_MATH_UTIL_H_

#include "../frenet_coordinate/types.h"
#include "box2d.h"
#include "line_segment2d.h"
#include "../frenet_coordinate/vec2d.h"
#include <limits>
#include <utility>
#include <vector>

#include <Eigen/Dense>



double Sqr(const double x);

/**
 * @brief Cross product between two 2-D vectors from the common start point,
 *        and end at two other points.
 * @param start_point The common start point of two vectors in 2-D.
 * @param end_point_1 The end point of the first vector.
 * @param end_point_2 The end point of the second vector.
 *
 * @return The cross product result.
 */
double CrossProd(const Vec2d &start_point, const Vec2d &end_point_1,
                 const Vec2d &end_point_2);

/**
 * @brief Inner product between two 2-D vectors from the common start point,
 *        and end at two other points.
 * @param start_point The common start point of two vectors in 2-D.
 * @param end_point_1 The end point of the first vector.
 * @param end_point_2 The end point of the second vector.
 *
 * @return The inner product result.
 */
double InnerProd(const Vec2d &start_point, const Vec2d &end_point_1,
                 const Vec2d &end_point_2);

/**
 * @brief Cross product between two vectors.
 *        One vector is formed by 1st and 2nd parameters of the function.
 *        The other vector is formed by 3rd and 4th parameters of the function.
 * @param x0 The x coordinate of the first vector.
 * @param y0 The y coordinate of the first vector.
 * @param x1 The x coordinate of the second vector.
 * @param y1 The y coordinate of the second vector.
 *
 * @return The cross product result.
 */
double CrossProd(const double x0, const double y0, const double x1,
                 const double y1);

/**
 * @brief Inner product between two vectors.
 *        One vector is formed by 1st and 2nd parameters of the function.
 *        The other vector is formed by 3rd and 4th parameters of the function.
 * @param x0 The x coordinate of the first vector.
 * @param y0 The y coordinate of the first vector.
 * @param x1 The x coordinate of the second vector.
 * @param y1 The y coordinate of the second vector.
 *
 * @return The inner product result.
 */
double InnerProd(const double x0, const double y0, const double x1,
                 const double y1);

/**
 * @brief Wrap angle to [0, 2 * PI).
 * @param angle the original value of the angle.
 * @return The wrapped value of the angle.
 */
double WrapAngle(const double angle);

/**
 * @brief Normalize angle to [-PI, PI).
 * @param angle the original value of the angle.
 * @return The normalized value of the angle.
 */
double NormalizeAngle(const double angle);

/**
 * @brief Calculate the difference between angle from and to
 * @param from the start angle
 * @param from the end angle
 * @return The difference between from and to. The range is between [-PI, PI).
 */
double AngleDiff(const double from, const double to);

/**
 * @brief Get a random integer between two integer values by a random seed.
 * @param s The lower bound of the random integer.
 * @param t The upper bound of the random integer.
 * @param random_seed The random seed.
 * @return A random integer between s and t based on the input random_seed.
 */
int RandomInt(const int s, const int t, unsigned int rand_seed = 1);

/**
 * @brief Get a random double between two integer values by a random seed.
 * @param s The lower bound of the random double.
 * @param t The upper bound of the random double.
 * @param random_seed The random seed.
 * @return A random double between s and t based on the input random_seed.
 */
double RandomDouble(const double s, const double t, unsigned int rand_seed = 1);

/**
 * @brief Compute squared value.
 * @param value The target value to get its squared value.
 * @return Squared value of the input value.
 */
template <typename T>
inline T Square(const T value) {
  return value * value;
}

/**
 * @brief Clamp a value between two bounds.
 *        If the value goes beyond the bounds, return one of the bounds,
 *        otherwise, return the original value.
 * @param value The original value to be clamped.
 * @param bound1 One bound to clamp the value.
 * @param bound2 The other bound to clamp the value.
 * @return The clamped value.
 */
template <typename T>
T Clamp(const T value, T bound1, T bound2) {
  if (bound1 > bound2) {
    std::swap(bound1, bound2);
  }

  if (value < bound1) {
    return bound1;
  } else if (value > bound2) {
    return bound2;
  }
  return value;
}

// Gaussian
double Gaussian(const double u, const double std, const double x);

// 2-dimension Gaussian
double Gaussian2d(const double u1, const double u2, const double std1,
                  const double std2, const double x1, const double x2,
                  const double rho);

// Sigmoid
double Sigmoid(const double x);

// Rotate a 2d vector counter-clockwise by theta
Eigen::Vector2d RotateVector2d(const Eigen::Vector2d &v_in, const double theta);

inline std::pair<double, double> RFUToFLU(const double x, const double y) {
  return std::make_pair(y, -x);
}

inline std::pair<double, double> FLUToRFU(const double x, const double y) {
  return std::make_pair(-y, x);
}

inline void L2Norm(int feat_dim, float *feat_data) {
  if (feat_dim == 0) {
    return;
  }
  // feature normalization
  float l2norm = 0.0f;
  for (int i = 0; i < feat_dim; ++i) {
    l2norm += feat_data[i] * feat_data[i];
  }
  if (l2norm == 0) {
    float val = 1.f / std::sqrt(static_cast<float>(feat_dim));
    for (int i = 0; i < feat_dim; ++i) {
      feat_data[i] = val;
    }
  } else {
    l2norm = std::sqrt(l2norm);
    for (int i = 0; i < feat_dim; ++i) {
      feat_data[i] /= l2norm;
    }
  }
}

/**
 * linear interpolation
 * vector x vector y double y
 **/
double interps(const std::vector<double> &y, const std::vector<double> &x,
               const double &x_interp);

/*
 * linear interpolation
 * vector x vector y vector xinterp
 */
void interpsVector(const std::vector<double> &y, const std::vector<double> &x,
                   const std::vector<double> &x_interps,
                   std::vector<double> &y_interps);

Pose2D calc_projection_point(const Pose2D &point1, const Pose2D &point2,
                             const Pose2D &point);

// Cartesian coordinates to Polar coordinates
std::pair<double, double> Cartesian2Polar(double x, double y);

LineSegment2d tf2d(const Pose2D &local_frame, const LineSegment2d &line);
Box2d tf2d(const Pose2D &local_frame, const Box2d &box);
Vec2d tf2d(const Pose2D &local_frame, const Vec2d &point);
Pose2D tf2d(const Pose2D &local_frame, const Pose2D &pose);

LineSegment2d tf2d_inv(const Pose2D &local_frame,
                       const LineSegment2d &line_local);
Box2d tf2d_inv(const Pose2D &local_frame, const Box2d &box_local);
Vec2d tf2d_inv(const Pose2D &local_frame, const Vec2d &point_local);
Pose2D tf2d_inv(const Pose2D &local_frame, const Pose2D &frame_local);

inline void trans_rot_2d(double x, double y, double x_offset, double y_offset,
                         double &x_local, double &y_local);

inline void rotate2d(double x, double y, double alpha, double origin_x,
                     double origin_y, double &x_local, double &y_local);

void get_rotate_matrix(float rotate_angle, float *rotate_matrix_ptr);

template <class T>
class IntervalMethodSolution {
 public:
  using Interval = std::pair<T, T>;
  static bool cmp(Interval a, Interval b) { return a.first < b.first; }
  std::vector<Interval> merge(std::vector<Interval> &intervals) {
    if (intervals.size() <= 1) {
      return intervals;
    }
    std::sort(intervals.begin(), intervals.end(), cmp);
    std::vector<Interval> res;
    res.push_back(intervals[0]);
    for (size_t i = 0; i < intervals.size(); i++) {
      if (res.back().second < intervals[i].first) {
        res.push_back(intervals[i]);
      } else {
        res.back().second = std::max(res.back().second, intervals[i].second);
      }
    }
    return res;
  }

  std::vector<Interval> intersect(const std::vector<Interval> &intervals,
                                  const Interval &section) {
    std::vector<Interval> res;
    for (auto &interval : intervals) {
      if (interval.second < section.first || interval.first > section.second) {
        continue;
      }
      res.push_back({std::max(interval.first, section.first),
                     std::min(interval.second, section.second)});
    }
    return res;
  }

  std::vector<Interval> intersect(const std::vector<Interval> &a,
                                  const std::vector<Interval> &b) {
    size_t i = 0;
    size_t j = 0;
    std::vector<Interval> result;
    while (i < a.size() && j < b.size()) {
      auto lo = std::max(a[i].first, b[j].first);
      auto hi = std::min(a[i].second, b[j].second);
      if (lo < hi) {
        result.push_back({lo, hi});
      }
      if (a[i].second < b[j].second) {
        i++;
      } else if (a[i].second > b[j].second) {
        j++;
      } else {
        i++;
        j++;
      }
    }
    return result;
  }

  void filter(std::vector<Interval> &intervals, T lower_length,
              T upper_length) {
    for (auto itor_s = intervals.begin(); itor_s != intervals.end();) {
      auto cur_length = std::abs(itor_s->second - itor_s->first);
      if (cur_length < lower_length || cur_length > upper_length) {
        itor_s = intervals.erase(itor_s);
      } else {
        ++itor_s;
      }
    }
  }

  std::vector<Interval> complement(const std::vector<Interval> &intervals,
                                   T lower_bound, T upper_bound) {
    std::vector<Interval> result;
    T sweep_line = lower_bound;
    for (auto &interval : intervals) {
      if (sweep_line >= upper_bound) {
        return result;
      }
      // if (internal.first >= upper_bound) {
      //   return result;
      // }
      if (sweep_line < interval.first) {
        result.push_back({sweep_line, std::min(upper_bound, interval.first)});
      }
      if (interval.second > sweep_line) {
        sweep_line = interval.second;
      }
    }
    if (sweep_line < upper_bound) {
      result.push_back({sweep_line, upper_bound});
    }
    return result;
  }

  T calculate_length(std::vector<Interval> &intervals) {
    T length{};
    for (auto itor_s = intervals.begin(); itor_s != intervals.end(); itor_s++) {
      length += std::abs(itor_s->second - itor_s->first);
    }
    return length;
  }

  bool is_in_intervals(const std::vector<Interval> &intervals, T t) {
    for (auto &interval : intervals) {
      if (t < interval.second && t > interval.first) {
        return true;
      }
    }
    return false;
  }
};



#endif /* MODULES_PLANNING_MATH_UTIL_H_ */
