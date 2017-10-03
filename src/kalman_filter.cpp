#include "kalman_filter.h"

using Eigen::MatrixXd;
using Eigen::VectorXd;

KalmanFilter::KalmanFilter() = default;

KalmanFilter::~KalmanFilter() = default;

void KalmanFilter::Init(VectorXd &x_in, MatrixXd &P_in, MatrixXd &F_in,
                        MatrixXd &H_in, MatrixXd &R_laser_in, MatrixXd &R_radar_in, MatrixXd &Q_in) {
  x_ = x_in;
  P_ = P_in;
  F_ = F_in;
  // This is the measurement matrix for laser data! For radar data, we need to always compute it again using the
  // linear approximation related to the Taylor expansion.
  H_ = H_in;
  R_laser_ = R_laser_in;
  R_radar_ = R_radar_in;
  Q_ = Q_in;
}

void KalmanFilter::Predict() {
  x_ = F_ * x_;
  MatrixXd Ft = F_.transpose();
  P_ = F_ * P_ * Ft + Q_;
}

/**
 * The function that actually performs the (E)KF Update step matrix computations. This is to avoid
 * duplicating the same code for EKF and basic KF where the only difference is that different matrices are used.
 *
 * @param H - the measurement matrix / Jacobian matrix, depending on whether standard or extended KF
 * @param y - the measurement difference vector
 * @param R - the noise covariance matrix related to the measurements, either Laser or Radar
 */
void KalmanFilter::DoUpdate(const MatrixXd &H, const VectorXd &y, const MatrixXd &R){
  // Prepare the different matrices, their transposes and inverses to avoid computing multiple times
  MatrixXd Ht = H.transpose();
  MatrixXd S = H * P_ * Ht + R;
  MatrixXd Si = S.inverse();
  MatrixXd PHt = P_ * Ht;
  MatrixXd K = PHt * Si;

  // Perform the actual updates
  x_ = x_ + (K * y);
  long x_size = x_.size();
  MatrixXd I = MatrixXd::Identity(x_size, x_size);
  P_ = (I - K * H) * P_;
}

/**
 * Standard Kalman filter Update.
 * @param z - the measurement vector
 */
void KalmanFilter::Update(const VectorXd &z) {
  // Just prepare the difference vector
  VectorXd z_pred = H_ * x_;
  VectorXd y = z - z_pred;

  // Then call the actual updater
  DoUpdate(H_, y, R_laser_);
}

/**
 * The Extended Kalman filter Update function
 * @param z - the measurement vector
 */
void KalmanFilter::UpdateEKF(const VectorXd &z) {
  // Prepare the Jacobian
  MatrixXd Hj;
  Hj = Tools::CalculateJacobian(x_);

  /* Prepare the difference vector, correcting for round-the-circle weirdness in angle differences close to the
   * wrap-around value (Pi).
   * To be specific, e.g. in the case measured --> phi -3.13, predicted --> phi 0.01 we have made a very
   * small conceptual error that looks big numerically. Also conceptually, a difference angle greater
   * (in absolute terms) than Pi makes no sense.
   * Also, the source data contains at least one
   * weird value (not in range -Pi .. Pi) so accounting for that too (although I ended up also adding a filter
   * that ignores such weird values).
   */

  VectorXd meas = Tools::ConvertToPolar(x_);
  VectorXd y = z - meas;
  while(y(1) < M_PI)
    y(1) += 2 * M_PI;
  while(y(1) > M_PI)
    y(1) -= 2 * M_PI;

  // DEBUG:
  // std::cout << "x: " << x_ << "\nz: " << z << "\nmeas: " << meas << "\ny: " << y << "\n";


  // Call the actual updater
  DoUpdate(Hj, y, R_radar_);
}


