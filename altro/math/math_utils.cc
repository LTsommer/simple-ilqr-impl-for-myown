#include "math_utils.h"

#include <cmath>
#include <utility>



double Sqr(const double x) { return x * x; }

// if CrossProd < 0: end_point2 is on the right side of line[start_point, end_point1]
// otherwise: end_point2 is on the left side of line[start_point, end_point1]
double CrossProd(const Vec2d &start_point, const Vec2d &end_point_1,
                 const Vec2d &end_point_2) {
  return (end_point_1 - start_point).CrossProd(end_point_2 - start_point);
}

double InnerProd(const Vec2d &start_point, const Vec2d &end_point_1,
                 const Vec2d &end_point_2) {
  return (end_point_1 - start_point).InnerProd(end_point_2 - start_point);
}

double CrossProd(const double x0, const double y0, const double x1,
                 const double y1) {
  return x0 * y1 - x1 * y0;
}

double InnerProd(const double x0, const double y0, const double x1,
                 const double y1) {
  return x0 * x1 + y0 * y1;
}

double WrapAngle(const double angle) {
  const double new_angle = std::fmod(angle, M_PI * 2.0);
  return new_angle < 0 ? new_angle + M_PI * 2.0 : new_angle;
}

double NormalizeAngle(const double angle) {
  double a = std::fmod(angle + M_PI, 2.0 * M_PI);
  if (a < 0.0) {
    a += (2.0 * M_PI);
  }
  return a - M_PI;
}

double AngleDiff(const double from, const double to) {
  return NormalizeAngle(to - from);
}

int RandomInt(const int s, const int t, unsigned int rand_seed) {
  if (s >= t) {
    return s;
  }
  return s + rand_r(&rand_seed) % (t - s + 1);
}

double RandomDouble(const double s, const double t, unsigned int rand_seed) {
  return s + (t - s) / 16383.0 * (rand_r(&rand_seed) & 16383);
}

// Gaussian
double Gaussian(const double u, const double std, const double x) {
  return (1.0 / std::sqrt(2 * M_PI * std * std)) *
         std::exp(-(x - u) * (x - u) / (2 * std * std));
}

// 2-dimension Gaussian
double Gaussian2d(const double u1, const double u2, const double std1,
                  const double std2, const double x1, const double x2,
                  const double rho) {
  return (1.0 / 2 * M_PI * std1 * std2 * std::sqrt(1 - rho * rho)) *
         std::exp(-((x1 - u1) * (x1 - u1) / (std1 * std1) +
                    (x2 - u2) * (x2 - u2) / (std2 * std2) -
                    2 * rho * (x1 - u1) * (x2 - u2) / (std1 * std2)) /
                  (2 * (1 - rho * rho)));
}

// Sigmoid
double Sigmoid(const double x) { return 1.0 / (1.0 + std::exp(-x)); }

Eigen::Vector2d RotateVector2d(const Eigen::Vector2d &v_in,
                               const double theta) {
  const double cos_theta = std::cos(theta);
  const double sin_theta = std::sin(theta);

  auto x = cos_theta * v_in.x() - sin_theta * v_in.y();
  auto y = sin_theta * v_in.x() + cos_theta * v_in.y();

  return {x, y};
}

double interps(const std::vector<double> &y, const std::vector<double> &x,
               const double &x_interp) {
  if (y.size() == 0) {
    return 0.0;
  } else if (y.size() == 1) {
    return y[0];
  } else {
    double s = x_interp;
    for (std::size_t j = 0; j + 1 < y.size(); j++) {
      if (s >= x[j] && s <= x[j + 1]) {
        return y[j] + (y[j + 1] - y[j]) / (x[j + 1] - x[j]) * (s - x[j]);
        break;
      } else if (j == (y.size() - 2)) {
        return y[j + 1];
      }
    }
  }
  return 0.0;
}

void interpsVector(const std::vector<double> &y, const std::vector<double> &x,
                   const std::vector<double> &x_interps,
                   std::vector<double> &y_interps) {
  assert(y.size() == x.size());
  for (std::size_t i = 0; i < x_interps.size(); i++) {
    if (y.size() == 0) {
      y_interps.push_back(0.);
    } else if (y.size() == 1) {
      y_interps.push_back(y[0]);
    } else {
      double s = x_interps[i];
      for (std::size_t j = 0; j + 1 < y.size(); j++) {
        if (s >= x[j] && s <= x[j + 1]) {
          y_interps.push_back(
              y[j] + (y[j + 1] - y[j]) / (x[j + 1] - x[j]) * (s - x[j]));
          break;
        } else if (j == (y.size() - 2)) {
          y_interps.push_back(y[j + 1]);
        }
      }
    }
  }
}

std::pair<double, double> Cartesian2Polar(double x, double y) {
  double r = std::sqrt(x * x + y * y);
  double theta = std::atan2(y, x);
  return std::make_pair(r, theta);
}

Pose2D calc_projection_point(const Pose2D &point1, const Pose2D &point2,
                             const Pose2D &point) {
  Pose2D projection_point;
  double k =
      ((point.x - point1.x) * (point2.x - point1.x) +
       (point.y - point1.y) * (point2.y - point1.y)) /
      std::pow(std::hypot(point2.x - point1.x, point2.y - point1.y), 2.0);
  k = std::min(std::max(0.0, k), 1.0);
  projection_point.x = point1.x + (point2.x - point1.x) * k;
  projection_point.y = point1.y + (point2.y - point1.y) * k;
  projection_point.theta = std::atan2(
      std::sin(point1.theta) * (1.0 - k) + std::sin(point2.theta) * k,
      std::cos(point1.theta) * (1.0 - k) + std::cos(point2.theta) * k);
  return projection_point;
}

LineSegment2d tf2d(const Pose2D &local_frame, const LineSegment2d &line) {
  return LineSegment2d(tf2d(local_frame, line.start()),
                       tf2d(local_frame, line.end()));
}

Box2d tf2d(const Pose2D &local_frame, const Box2d &box) {
  return Box2d(tf2d(local_frame, box.center()),
               NormalizeAngle(box.heading() - local_frame.theta), box.length(),
               box.width());
}

Vec2d tf2d(const Pose2D &local_frame, const Vec2d &point) {
  double x_local, y_local;
  trans_rot_2d(point.x(), point.y(), -local_frame.x, -local_frame.y, x_local,
               y_local);
  rotate2d(x_local, y_local, -local_frame.theta, 0.0, 0.0, x_local, y_local);
  return Vec2d(x_local, y_local);
}

Pose2D tf2d(const Pose2D &local_frame, const Pose2D &pose) {
  Pose2D Point_local;
  trans_rot_2d(pose.x, pose.y, -local_frame.x, -local_frame.y, Point_local.x,
               Point_local.y);
  rotate2d(Point_local.x, Point_local.y, -local_frame.theta, 0.0, 0.0,
           Point_local.x, Point_local.y);
  Point_local.theta = NormalizeAngle(pose.theta - local_frame.theta);
  return Point_local;
}

LineSegment2d tf2d_inv(const Pose2D &local_frame,
                       const LineSegment2d &line_local) {
  Pose2D tmp;
  auto global_frame_pose_local = tf2d(local_frame, tmp);
  return tf2d(global_frame_pose_local, line_local);
}

Box2d tf2d_inv(const Pose2D &local_frame, const Box2d &box_local) {
  Pose2D tmp;
  auto global_frame_pose_local = tf2d(local_frame, tmp);
  return tf2d(global_frame_pose_local, box_local);
}

Vec2d tf2d_inv(const Pose2D &local_frame, const Vec2d &point_local) {
  Pose2D tmp;
  auto global_frame_pose_local = tf2d(local_frame, tmp);
  return tf2d(global_frame_pose_local, point_local);
}

Pose2D tf2d_inv(const Pose2D &local_frame, const Pose2D &frame_local) {
  Pose2D tmp;
  auto global_frame_pose_local = tf2d(local_frame, tmp);
  return tf2d(global_frame_pose_local, frame_local);
}

inline void trans_rot_2d(double x, double y, double x_offset, double y_offset,
                         double &x_local, double &y_local) {
  x_local = x + x_offset;
  y_local = y + y_offset;
}

inline void rotate2d(double x, double y, double alpha, double origin_x,
                     double origin_y, double &x_local, double &y_local) {
  double cos_a = cos(alpha);
  double sin_a = sin(alpha);
  x_local = origin_x + (x - origin_x) * cos_a - (y - origin_y) * sin_a;
  y_local = origin_y + (x - origin_x) * sin_a + (y - origin_y) * cos_a;
}

void get_rotate_matrix(float rotate_angle, float *rotate_matrix_ptr) {
  float cos_theta = cos(M_PI / 2 - rotate_angle);
  float sin_theta = sin(M_PI / 2 - rotate_angle);
  rotate_matrix_ptr[0] = cos_theta;
  rotate_matrix_ptr[1] = sin_theta;
  rotate_matrix_ptr[2] = -sin_theta;
  rotate_matrix_ptr[3] = cos_theta;
}

