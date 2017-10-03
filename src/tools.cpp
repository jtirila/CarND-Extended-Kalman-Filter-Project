#include <iostream>
#include "tools.h"


using Eigen::VectorXd;
using Eigen::MatrixXd;
using std::vector;

/*
 * Convert a vector in polar coordinates into the cartesian coordinate system
 *
 * The input vector polar_coords is a three-element VectorXd containing rho (radius), phi (angle in radians) and
 * rhodot (range rate)
 *
 * Returns the vector with cartesian coordinates
 */
VectorXd Tools::ConvertToCartesian(const VectorXd &polar_coords){
  VectorXd cartesian_coords(4);
  double r = polar_coords(0);
  double phi = polar_coords(1);

  cartesian_coords << std::cos(phi) * r, std::sin(phi) * r, 0, 0;

  return cartesian_coords;
}

/**
 * Convert a vector in cartesian coordinates into the polar coordinate system
 *
 * @param cart_coords - is a four-element VectorXd containing px, py, vx and vy
 *
 * @return - a vector with polar coordinates
 */
VectorXd Tools::ConvertToPolar(const VectorXd &cart_coords) {
  VectorXd polar_coords(3);
  double px = cart_coords(0);
  double py = cart_coords(1);
  double vx = cart_coords(2);
  double vy = cart_coords(3);

  double sqrt_px_2_py_2 = sqrt(px * px + py * py);

  double rho = sqrt_px_2_py_2;

  double phi;
  if(-0.0001 < px && px < 0.0001)
    phi = M_PI / 2.0;
  else
    phi = atan(py / px);

  // Correction in the 2nd and 4th quarter or the circle
  if(py > 0 && phi < 0)
    phi += M_PI;

  if(py < 0 && phi > 0)
    phi -= M_PI;


  double rhodot = (px * vx + py * vy) / sqrt_px_2_py_2;
  polar_coords << rho, phi, rhodot;
  return polar_coords;
}

VectorXd Tools::CalculateRMSE(const vector<VectorXd> &estimations,
                       const vector<VectorXd> &ground_truth){

  VectorXd rmse(4);
  rmse << 0, 0, 0, 0;

  if(estimations.empty() || estimations.size() != ground_truth.size()){
    cout << "Error!";
    return rmse;
  }

  // Accumulate the residuals
  for(int i=0; i < estimations.size(); ++i){
    VectorXd residuals = estimations[i] - ground_truth[i];
    VectorXd sqr_res = residuals.array() * residuals.array();
    rmse += sqr_res;
  }

  rmse /= estimations.size();
  rmse = rmse.array().sqrt();
  return rmse;
}


/**
 * Compute Jacobian, one function per row and one variable per column.
 *
 * The Jacobian computation only applies to Radar measurements!
 */
MatrixXd Tools::CalculateJacobian(const VectorXd& x_state) {

  MatrixXd Hj(3,4);

  //recover state parameters
  double px = x_state(0);
  double py = x_state(1);
  double vx = x_state(2);
  double vy = x_state(3);

  // pre-compute a set of terms to avoid repeated calculation
  double c1 = px*px+py*py;

  // Check division by zero and quit as early as possible
  if(fabs(c1) < 0.0001){
    cout << "CalculateJacobian () - Error - Division by Zero" << endl;

    // FIXME: is this really what we want to do?
    return Hj;
  }

  double c2 = sqrt(c1);
  double c3 = (c1*c2);


  //compute the Jacobian matrix
  Hj <<  px/c2                 , py/c2                , 0    , 0,
          -py/c1               , px/c1                , 0    , 0,
          py*(vx*py - vy*px)/c3, px*(px*vy - py*vx)/c3, px/c2, py/c2;

  return Hj;
}
