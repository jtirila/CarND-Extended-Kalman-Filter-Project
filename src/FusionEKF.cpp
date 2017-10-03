#include "FusionEKF.h"
#include "tools.h"
#include "Eigen/Dense"
#include <iostream>
#include <cmath>
#include "kalman_filter.h"

#include <assert.h>

// Some debug flags to easily switch between processing both, or just one type of measurement
#define IGNORE_LIDAR false
#define IGNORE_RADAR false

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

/*
 * Constructor.
 */
FusionEKF::FusionEKF() {
  is_initialized_ = false;

  // initializing matrices
  MatrixXd P_init(4, 4);
  MatrixXd R_laser(2, 2);
  MatrixXd R_radar(3, 3);
  MatrixXd H_laser(2, 4);

  MatrixXd Hj_(3, 4);

  //measurement covariance matrix - laser
  R_laser << 0.0225, 0,
             0,      0.0225;

  //measurement covariance matrix - radar
  R_radar << 0.09, 0,      0,
             0,    0.0009, 0,
             0,    0,      0.09;


  noise_ax_ = 9;
  noise_ay_ = 9;

  P_init << noise_ax_,   0,          0,    0,
            0,           noise_ay_,  0,    0,
            0,           0,          1000, 0,
            0,           0,          0,    1000;


  // Set up an initial state transition matrix; we do not know about elapsed time yet so let us just use a value of 0
  // as the matrix will be updated anyway before the next prediction
  MatrixXd F(4, 4);
  F << 1, 0, 0, 0,
       0, 1, 0, 0,
       0, 0, 1, 0,
       0, 0, 0, 1;

  MatrixXd Q(4, 4);
  Q << 0, 0, 0, 0,
       0, 0, 0, 0,
       0, 0, 0, 0,
       0, 0, 0, 0;

  H_laser << 1, 0, 0, 0,
             0, 1, 0, 0;

  ekf_ = KalmanFilter();
  ekf_.Init(ekf_.x_, P_init, F, H_laser, R_laser, R_radar, Q);

  // Some debug prints
  // cout << "KalmanFilter F matrix After fusionekf initialization: \n" << ekf_.F_ << "\n";
  // cout << "previous_timestamp_: " << previous_timestamp_ << "\n";
}

/**
* Destructor
*/
FusionEKF::~FusionEKF() = default;

void FusionEKF::ProcessMeasurement(const MeasurementPackage &measurement_pack) {

  // For this project, just use a guard against weird rows in the measurement data. If these did really occur,
  // something more sophisticated would be needed.
  assert(measurement_pack.sensor_type_ == MeasurementPackage::RADAR
      || measurement_pack.sensor_type_ == MeasurementPackage::LASER);

  if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR &&
      (M_PI < measurement_pack.raw_measurements_(1) || -M_PI > measurement_pack.raw_measurements_(1))) {

      // Do nothing with radar measurements where the measured angle is not between -PI and PI. This is probably
      // not necessary as one spurious measurement would do little harm in the big picture. However, I guess
      // it is one feasible strategy to deal with such values to just ignore them.
      cout << "Skipping a weird measurement, radar data with values " << measurement_pack.raw_measurements_ << " \n";
      return;
    }

  /*****************************************************************************
   *  Initialization
   ****************************************************************************/
  if (!is_initialized_) {

    ekf_.x_ = VectorXd(4);
    ekf_.x_ << 0.0, 0.0, 0.0, 0.0;

    if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR) {
      /**
      Convert radar from polar to cartesian coordinates and initialize state.
      */

      VectorXd cart_coord = Tools::ConvertToCartesian(measurement_pack.raw_measurements_);

      ekf_.x_(0) = cart_coord(0);
      ekf_.x_(1) = cart_coord(1);
      ekf_.x_(2) = 0;
      ekf_.x_(3) = 0;
    }
    else if (measurement_pack.sensor_type_ == MeasurementPackage::LASER) {
      ekf_.x_ << measurement_pack.raw_measurements_(0), measurement_pack.raw_measurements_(1), 0, 0;
    }

    previous_timestamp_ = measurement_pack.timestamp_;

    // done initializing, no need to predict or update
    is_initialized_ = true;

    // Debug:
    // cout << "Initialization done\n";

    return;
  }

  // Start the processing if initialization had been performed previously already

  if (IGNORE_LIDAR && measurement_pack.sensor_type_ == MeasurementPackage::LASER) {
    cout << "Ignoring laser measurements to investigate radar processing performance.\n";
    return;
  }

  if (IGNORE_RADAR && measurement_pack.sensor_type_ == MeasurementPackage::RADAR) {
    cout << "Ignoring radar measurements to investigate radar processing performance.\n";
    return;
  }


  /*****************************************************************************
   *  Prediction
   ****************************************************************************/


  double dt = (measurement_pack.timestamp_ - previous_timestamp_) / 1000000.0;	//dt - expressed in seconds
  previous_timestamp_ = measurement_pack.timestamp_;

  // Modify the F and Q matrices so that the time is integrated; these modifications will be in effect when entering the
  // Prediction and Update steps.

  ekf_.F_(0, 2) = dt;
  ekf_.F_(1, 3) = dt;


  float dt_square = dt * dt;
  float dt_cube = dt_square * dt;
  float dt_fourth = dt_cube * dt;


  ekf_.Q_(0, 0) = dt_fourth / 4.0 * noise_ax_;
  ekf_.Q_(0, 2) = dt_cube / 2.0 * noise_ax_;

  ekf_.Q_(1, 1) = dt_fourth / 4.0 * noise_ay_;
  ekf_.Q_(1, 3) = dt_cube / 2.0 * noise_ay_;

  ekf_.Q_(2, 0) = dt_cube / 2.0 * noise_ax_;
  ekf_.Q_(2, 2) = dt_square * noise_ax_;

  ekf_.Q_(3, 1) = dt_cube / 2.0 * noise_ay_;
  ekf_.Q_(3, 3) = dt_square * noise_ay_;

  // Debug:
  // cout << "ekf_.F_: " << ekf_.F_ << "\n";

  ekf_.Predict();

  /*****************************************************************************
   *  Update
   ****************************************************************************/

  if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR) {
    // Call the function that will perform extended KF on the inputs
    ekf_.UpdateEKF(measurement_pack.raw_measurements_);
  } else if (measurement_pack.sensor_type_ == MeasurementPackage::LASER) {
    // Call the function that will perform basic KF on the inputs
    ekf_.Update(measurement_pack.raw_measurements_);
  }

  // print the output
  cout << "x_ = " << ekf_.x_ << endl;
  cout << "P_ = " << ekf_.P_ << endl;
}
