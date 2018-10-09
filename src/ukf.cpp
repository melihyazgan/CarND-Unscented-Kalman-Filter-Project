#include "ukf.h"
#include "Eigen/Dense"
#include <iostream>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

UKF::UKF() {
	// Not initialized until first process measurement
	is_initialized_ = false;

	// if this is false, laser measurements will be ignored (except during init)
	use_laser_ = true;

	// if this is false, radar measurements will be ignored (except during init)
	use_radar_ = true;

	// initial state vector
	x_ = VectorXd(5);

	// initial covariance matrix
	P_ = MatrixXd(5, 5);

	// Process noise standard deviation longitudinal acceleration in m/s^2
	// ***** Tuning Parameter *****
	std_a_ = 2.0;

	// Process noise standard deviation yaw acceleration in rad/s^2
	// ***** Tuning Parameter *****
	std_yawdd_ = 0.3;

	// Laser measurement noise standard deviation position1 in m
	std_laspx_ = 0.15;

	// Laser measurement noise standard deviation position2 in m
	std_laspy_ = 0.15;

	// Radar measurement noise standard deviation radius in m
	std_radr_ = 0.3;

	// Radar measurement noise standard deviation angle in rad
	std_radphi_ = 0.03;

	// Radar measurement noise standard deviation radius change in m/s
	std_radrd_ = 0.3;

	// Set state dimension
	n_x_ = 5;

	// Set augmented dimension
	n_aug_ = 7;

	// Define spreading parameter
	lambda_ = 0;

	// Matrix to hold sigma points
	Xsig_pred_ = MatrixXd(n_x_, 2 * n_aug_ + 1);

	// Vector for weights
	weights_ = VectorXd(2 * n_aug_ + 1);

	// Start time
	time_us_ = 0;

	// NIs
	NIS_laser_ = 0;
	NIS_radar_ = 0;
}

UKF::~UKF() {}

/**
* @param {MeasurementPackage} meas_package The latest measurement data of
* either radar or laser.
*/
void UKF::ProcessMeasurement(MeasurementPackage meas_package) {
	/**
	TODO:

	Complete this function! Make sure you switch between lidar and radar
	measurements.
	*/
	if (!is_initialized_) {

		if (meas_package.sensor_type_ == MeasurementPackage::RADAR) {
			// Convert polar to cartesian coordinates
			double rho = meas_package.raw_measurements_[0];
			double phi = meas_package.raw_measurements_[1];
			double rho_dot = meas_package.raw_measurements_[2];

			double px = rho * cos(phi);
			double py = rho * sin(phi);
			double vx = rho_dot * cos(phi);
			double vy = rho_dot * sin(phi);
			// **** 3 is the initial velocity, which I estimate and can be tuned
			x_ << px, py, 3.0, vx, vy;
			//state covariance matrix
			//***** values can be tuned *****
			P_ << std_radr_ * std_radr_, 0, 0, 0, 0,
				0, std_radr_*std_radr_, 0, 0, 0,
				0, 0, 1, 0, 0,
				0, 0, 0, std_radphi_, 0,
				0, 0, 0, 0, std_radphi_;
		}
		else if (meas_package.sensor_type_ == MeasurementPackage::LASER) {
			/**
			Initialize state.
			*/
			// Last three are the initial veloctiy, psi and psi_dot and can be tuned
			x_ << meas_package.raw_measurements_(0), meas_package.raw_measurements_(1), 3.0, 0, 0;

			//state covariance matrix
			//***** values can be tuned *****
			P_ << std_laspx_ * std_laspx_, 0, 0, 0, 0,
				0, std_laspy_*std_laspy_, 0, 0, 0,
				0, 0, 1, 0, 0,
				0, 0, 0, 1, 0,
				0, 0, 0, 0, 1;
		}
		time_us_ = meas_package.timestamp_;
		// done initializing, no need to predict or update
		is_initialized_ = true;
		return;
	}
	double delta_t = (meas_package.timestamp_ - time_us_) / 1000000.0;
	time_us_ = meas_package.timestamp_;
	Prediction(delta_t);


	double radar_nis = 0; double laser_nis = 0;
	//vector<double> rnis = { radar_nis };
	//vector<double> lnis = { laser_nis };
	//ofstream outfile("out.txt");
	if (meas_package.sensor_type_ == MeasurementPackage::RADAR) {
		UpdateRadar(meas_package);
	}
	else {
		UpdateLidar(meas_package);
	}
}

/**
* Predicts sigma points, the state, and the state covariance matrix.
* @param {double} delta_t the change in time (in seconds) between the last
* measurement and this one.
*/
void UKF::Prediction(double delta_t) {
	cout << "Prediction.." << endl;
	/*****************************************************************************
	*  Generate Sigma Points
	****************************************************************************/
	lambda_ = 3 - n_x_;

	//create sigma point matrix
	MatrixXd Xsig = MatrixXd(n_x_, 2 * n_x_ + 1);

	//calculate square root of P
	MatrixXd A = P_.llt().matrixL();

	//calculate sigma points ...
	//set sigma points as columns of matrix Xsig
	Xsig.col(0) = x_;
	for (int i = 0; i < n_x_; ++i) {
		Xsig.col(i + 1) = x_ + sqrt(lambda_ + n_x_) * A.col(i);
		Xsig.col(i + 1 + n_x_) = x_ - sqrt(lambda_ + n_x_) * A.col(i);
	}
	/*****************************************************************************
	*  Augment Sigma Points
	****************************************************************************/

	// New Lmabda for Augmentation
	lambda_ = 3 - n_aug_;

	//create augmented mean vector
	VectorXd x_aug = VectorXd(n_aug_);

	//create augmented state covariance
	MatrixXd P_aug = MatrixXd(n_aug_, n_aug_);

	//create sigma point matrix
	MatrixXd Xsig_aug = MatrixXd(n_aug_, 2 * n_aug_ + 1);

	//create augmented mean state
	x_aug.head(5) = x_;
	x_aug(5) = 0;
	x_aug(6) = 0;

	//create augmented covariance matrix
	P_aug.fill(0.0);
	P_aug.topLeftCorner(5, 5) = P_;
	P_aug(5, 5) = std_a_ * std_a_;
	P_aug(6, 6) = std_yawdd_ * std_yawdd_;

	//create square root matrix
	MatrixXd L = P_aug.llt().matrixL();

	//create augmented sigma points
	Xsig_aug.col(0) = x_aug;
	for (int i = 0; i < n_aug_; ++i) {
		Xsig_aug.col(i + 1) = x_aug + sqrt(lambda_ + n_aug_)*L.col(i);
		Xsig_aug.col(i + 1 + n_aug_) = x_aug - sqrt(lambda_ + n_aug_)*L.col(i);
	}

	/*****************************************************************************
	*  Predict Sigma Points
	****************************************************************************/

	//predict sigma points
	for (int i = 0; i< 2 * n_aug_ + 1; ++i) {
		double px = Xsig_aug(0, i);
		double py = Xsig_aug(1, i);
		double v = Xsig_aug(2, i);
		double yaw = Xsig_aug(3, i);
		double yawd = Xsig_aug(4, i);
		double nua = Xsig_aug(5, i);
		double nuy = Xsig_aug(6, i);


		// predicted locations pxp,pyp
		double pxp, pyp;
		if (fabs(yawd)> 0.001) {
			pxp = px + (sin(yaw + yawd * delta_t) - sin(yaw))*(v / yawd);
			pyp = py + (-cos(yaw + yawd * delta_t) + cos(yaw))*(v / yawd);
		}
		else {
			pxp = px + (cos(yaw) * delta_t * v);
			pyp = py + (sin(yaw) * delta_t * v);
		}
		double vp, yawp, yawdp;
		vp = v;
		yawp = yaw + yawd * delta_t;
		yawdp = yawd;

		// add Process noises
		pxp = pxp + (cos(yaw) * nua *pow(delta_t, 2) / 2);
		pyp = pyp + (sin(yaw) * nua *pow(delta_t, 2) / 2);
		vp = vp + nua * delta_t;
		yawp = yawp + (nuy * pow(delta_t, 2) / 2);
		yawdp = yawdp + (nuy * delta_t);

		//write predicted sigma points into right column
		Xsig_pred_(0, i) = pxp;
		Xsig_pred_(1, i) = pyp;
		Xsig_pred_(2, i) = vp;
		Xsig_pred_(3, i) = yawp;
		Xsig_pred_(4, i) = yawdp;
	}

	/*****************************************************************************
	*  Predicted Sigma Points to Mean/Covariance
	****************************************************************************/
	//create vector for predicted state
	VectorXd x_pred = VectorXd(n_x_);

	//create covariance matrix for prediction
	MatrixXd P_pred = MatrixXd(n_x_, n_x_);

	//set weights
	weights_(0) = lambda_ / (lambda_ + n_aug_);
	for (int i = 1; i < 2 * n_aug_ + 1; ++i) {
		double weight_value = 0.5 / (lambda_ + n_aug_);
		weights_(i) = weight_value;
	}
	//predict state mean
	x_pred.fill(0.0);
	for (int i = 0; i < 2 * n_aug_ + 1; ++i) {
		x_pred = x_pred + weights_(i) * Xsig_pred_.col(i);
	}
	//predict state covariance matrix
	P_pred.fill(0.0);
	for (int i = 0; i < (2 * n_aug_ + 1); ++i) {
		VectorXd x_diff = Xsig_pred_.col(i) - x_pred;

		//angle normalization
		while (x_diff(3)> M_PI) x_diff(3) -= 2.*M_PI;
		while (x_diff(3)<-M_PI) x_diff(3) += 2.*M_PI;

		P_pred = P_pred + weights_(i) * x_diff * x_diff.transpose();
	}
	x_ = x_pred;
	P_ = P_pred;
}

/**
* Updates the state and the state covariance matrix using a laser measurement.
* @param {MeasurementPackage} meas_package
*/
double UKF::UpdateLidar(MeasurementPackage meas_package) {
	/**
	TODO:

	Complete this function! Use lidar data to update the belief about the object's
	position. Modify the state vector, x_, and covariance, P_.

	You'll also need to calculate the lidar NIS.
	*/
	cout << "Update Lidar..." << endl;
	int n_z = 2;

	// matrix sigma points in measurement space
	MatrixXd Zsig = MatrixXd(n_z, 2 * n_aug_ + 1);

	//mean predicted measurement
	VectorXd z_pred = VectorXd(n_z);

	//measurement covariance matrix S
	MatrixXd S = MatrixXd(n_z, n_z);

	//transform sigma points into measurement space
	for (int i = 0; i < 2 * n_aug_ + 1; ++i) {
		double px = Xsig_pred_(0, i);
		double py = Xsig_pred_(1, i);
		Zsig(0, i) = px;
		Zsig(1, i) = py;
	}
	//weights

	weights_(0) = lambda_ / (lambda_ + n_aug_);
	for (int i = 1; i < 2 * n_aug_ + 1; ++i) {
		double weight_value = 0.5 / (lambda_ + n_aug_);
		weights_(i) = weight_value;
	}
	//calculate mean predicted measurement
	z_pred.fill(0.0);
	for (int i = 0; i < 2 * n_aug_ + 1; ++i) {
		z_pred = z_pred + weights_(i) * Zsig.col(i);
	}

	//calculate innovation covariance matrix S
	S.fill(0.0);
	for (int i = 0; i < 2 * n_aug_ + 1; ++i) {
		VectorXd z_diff = Zsig.col(i) - z_pred;
		//angle normalization
		while (z_diff(1)> M_PI) z_diff(1) -= 2.*M_PI;
		while (z_diff(1)<-M_PI) z_diff(1) += 2.*M_PI;

		S = S + weights_(i) * z_diff * z_diff.transpose();
	}
	MatrixXd R = MatrixXd(n_z, n_z);
	R << std_laspx_ * std_laspx_, 0,
		0, std_laspy_*std_laspy_;
	S = S + R;

	//create example vector for incoming radar measurement
	VectorXd z = VectorXd(n_z);

	double meas_px = meas_package.raw_measurements_(0);
	double meas_py = meas_package.raw_measurements_(1);

	z << meas_px,
		meas_py;

	//create matrix for cross correlation Tc
	MatrixXd Tc = MatrixXd(n_x_, n_z);


	//calculate cross correlation matrix
	Tc.fill(0.0);
	for (int i = 0; i < 2 * n_aug_ + 1; ++i) {
		VectorXd x_diff = Xsig_pred_.col(i) - x_;
		VectorXd z_diff = Zsig.col(i) - z_pred;

		//angle normalization
		while (x_diff(3)> M_PI) x_diff(3) -= 2.*M_PI;
		while (x_diff(3)<-M_PI) x_diff(3) += 2.*M_PI;
		//angle normalization
		while (z_diff(1)> M_PI) z_diff(1) -= 2.*M_PI;
		while (z_diff(1)<-M_PI) z_diff(1) += 2.*M_PI;

		Tc = Tc + weights_(i) * x_diff * z_diff.transpose();
	}
	//calculate Kalman gain K;
	MatrixXd K = Tc * S.inverse();
	VectorXd z_diff = z - z_pred;
	NIS_laser_ = z_diff.transpose()*S.inverse()*z_diff;
	//update state mean and covariance matrix

	x_ = x_ + K * (z - z_pred);
	P_ = P_ - K * S * K.transpose();
	return NIS_laser_;
}

/**
* Updates the state and the state covariance matrix using a radar measurement.
* @param {MeasurementPackage} meas_package
*/
double UKF::UpdateRadar(MeasurementPackage meas_package) {
	cout << "Update Radar.." << endl;
	int n_z = 3;

	// matrix sigma points in measurement space
	MatrixXd Zsig = MatrixXd(n_z, 2 * n_aug_ + 1);

	//mean predicted measurement
	VectorXd z_pred = VectorXd(n_z);

	//measurement covariance matrix S
	MatrixXd S = MatrixXd(n_z, n_z);

	//transform sigma points into measurement space
	for (int i = 0; i < 2 * n_aug_ + 1; i++) {
		double px = Xsig_pred_(0, i);
		double py = Xsig_pred_(1, i);
		double v = Xsig_pred_(2, i);
		double yaw = Xsig_pred_(3, i);
		double yawd = Xsig_pred_(4, i);
		double vx = v * cos(yaw);
		double vy = v * sin(yaw);

		double rho, theta, rho_dot;
		rho = sqrt(pow(px, 2) + pow(py, 2));
		theta = atan2(py, px);
		rho_dot = (px*vx + py * vy) / rho;
		Zsig(0, i) = rho;
		Zsig(1, i) = theta;
		Zsig(2, i) = rho_dot;
	}
	//set weights
	weights_(0) = lambda_ / (lambda_ + n_aug_);
	for (int i = 1; i < 2 * n_aug_ + 1; i++) {
		double weight_value = 0.5 / (lambda_ + n_aug_);
		weights_(i) = weight_value;
	}
	//calculate mean predicted measurement
	z_pred.fill(0.0);
	for (int i = 0; i < 2 * n_aug_ + 1; ++i) {
		z_pred = z_pred + weights_(i) * Zsig.col(i);
	}

	//calculate innovation covariance matrix S
	S.fill(0.0);
	for (int i = 0; i < 2 * n_aug_ + 1; i++) {
		VectorXd z_diff = Zsig.col(i) - z_pred;
		//angle normalization
		while (z_diff(1)> M_PI) z_diff(1) -= 2.*M_PI;
		while (z_diff(1)<-M_PI) z_diff(1) += 2.*M_PI;

		S = S + weights_(i) * z_diff * z_diff.transpose();
	}
	MatrixXd R = MatrixXd(n_z, n_z);
	R << pow(std_radr_, 2), 0, 0,
		0, pow(std_radphi_, 2), 0,
		0, 0, pow(std_radrd_, 2);
	S = S + R;

	//create example vector for incoming radar measurement
	VectorXd z = VectorXd(n_z);

	double meas_rho = meas_package.raw_measurements_(0);
	double meas_phi = meas_package.raw_measurements_(1);
	double meas_rhod = meas_package.raw_measurements_(2);

	z << meas_rho,
		 meas_phi,
		 meas_rhod;

	//create matrix for cross correlation Tc
	MatrixXd Tc = MatrixXd(n_x_, n_z);


	//calculate cross correlation matrix
	Tc.fill(0.0);
	for (int i = 0; i < 2 * n_aug_ + 1; ++i) {
		VectorXd x_diff = Xsig_pred_.col(i) - x_;
		VectorXd z_diff = Zsig.col(i) - z_pred;

		//angle normalization
		while (x_diff(3)> M_PI) x_diff(3) -= 2.*M_PI;
		while (x_diff(3)<-M_PI) x_diff(3) += 2.*M_PI;
		//angle normalization
		while (z_diff(1)> M_PI) z_diff(1) -= 2.*M_PI;
		while (z_diff(1)<-M_PI) z_diff(1) += 2.*M_PI;

		Tc = Tc + weights_(i) * x_diff * z_diff.transpose();
	}
	VectorXd z_diff = z - z_pred;
	NIS_radar_ = z_diff.transpose()*S.inverse()*z_diff;

	//calculate Kalman gain K;
	MatrixXd K = Tc * S.inverse();

	//update state mean and covariance matrix
	x_ = x_ + K * (z - z_pred);
	P_ = P_ - K * S * K.transpose();
	return NIS_radar_;
}