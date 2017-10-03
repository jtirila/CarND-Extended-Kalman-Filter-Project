#ifndef TOOLS_H_
#define TOOLS_H_
#include <vector>
#include "Eigen/Dense"

using Eigen::MatrixXd;
using Eigen::VectorXd;
using namespace std;

// I refactored the Tools class into a namespace as it seems to me we only need to be able to call static functions
// and not maintain any state within the Tools class, hence making a simple namespace work as well without
// the need of creating an instance and passing it around.
namespace Tools {
  /**
  * A helper function to calculate RMSE.
  */
  VectorXd CalculateRMSE(const vector<VectorXd> &estimations, const vector<VectorXd> &ground_truth);

  /**
  * A helper function to convert from polar to cartesian coordinates.
  */
  VectorXd ConvertToCartesian(const VectorXd &polar_coords);

  /**
  * A helper function to convert from cartesian to polar coordinates.
  */
  VectorXd ConvertToPolar(const VectorXd &cart_coords);

  /**
  * A helper function to calculate Jacobians.
  */
  MatrixXd CalculateJacobian(const VectorXd& x_state);

};

#endif /* TOOLS_H_ */
