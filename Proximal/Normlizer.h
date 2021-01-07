#pragma once
#include "problem.h"
#include "Eigen/dense"

/*特例化的 Normalization*/

class Norm_12 :public Normlization<Eigen::MatrixXd> {
	
	/*矩阵1-2范数*/

public:
	virtual double operator()(const Eigen::MatrixXd& x) {
		return x.rowwise().norm().sum();
	}

	virtual Eigen::MatrixXd Prox(const Eigen::MatrixXd& x, double t) {
		auto agg = (1 - x.rowwise().norm().array().inverse() * t).max(0);
		return (x.array().colwise() * agg).matrix();
	}

}; 

class Norm_21 :public Normlization<Eigen::MatrixXd> {

	/*矩阵2-1范数*/

public:
	virtual double operator()(const Eigen::MatrixXd& x) {
		return x.colwise().norm().sum();
	}

	virtual Eigen::MatrixXd Prox(const Eigen::MatrixXd& x, double t) {
		auto agg = (1 - x.colwise().norm().array().inverse() * t).max(0);
		return (x.array().rowwise() * agg).matrix();
	}

};

class Norm_0 :public Normlization<Eigen::MatrixXd> {

	/* 矩阵0-范数，即非零元素个数。非凸！ */
public:
	virtual double operator()(const Eigen::MatrixXd& x) {
		return double((x.array().abs() < 1e-6).count());
	}

	virtual Eigen::MatrixXd Prox(const Eigen::MatrixXd& x, double t) {
		return (x.array() * (x.array().abs() < std::sqrt(2 * t)).cast<double>()).matrix();
	}
};

class Norm_1 :public Normlization<Eigen::MatrixXd> {

	/* 矩阵1-范数。 */
public:
	virtual double operator()(const Eigen::MatrixXd& x) {
		return x.lpNorm<1>();
	}

	virtual Eigen::MatrixXd Prox(const Eigen::MatrixXd& x, double t) {
		return (x.array().sign() * (x.array().abs() - t).max(0)).matrix();
	}
};

class Norm_2 :public Normlization<Eigen::MatrixXd> {

	/* 矩阵2-范数。 */
public:
	virtual double operator()(const Eigen::MatrixXd& x) {
		return x.lpNorm<2>();
	}

	virtual Eigen::MatrixXd Prox(const Eigen::MatrixXd& x, double t) {
		return std::max(0.0, 1 - t / x.norm()) * x;
	}
};

class Norm_Inf :public Normlization<Eigen::MatrixXd> {

	/* 矩阵无穷范数。 */
public:
	virtual double operator()(const Eigen::MatrixXd& x) {
		return x.array().abs().maxCoeff();
	}

	virtual Eigen::MatrixXd Prox(const Eigen::MatrixXd& x, double t) {
		auto tmp = x.array().abs();
		double maxValue = tmp.maxCoeff();
		auto rtn = (tmp == maxValue).select(maxValue - t, tmp);
		double secondMax = rtn.maxCoeff();
		if (secondMax > maxValue - t || secondMax == 0) {
			secondMax = std::max(0.0, secondMax);
			return ((tmp == maxValue).select(secondMax, tmp) * x.array().sign()).matrix();
		}
		return (rtn * x.array().sign()).matrix();
	}
};

class Indicator_L2Ball :public Normlization<Eigen::MatrixXd> {

	/* L2范数球的示性函数。 */
public:
	virtual double operator()(const Eigen::MatrixXd& x) {
		return 0; // 未考虑不在球上的情况，因为传入系数的限制。请使用重载。
	}

	double operator()(const Eigen::MatrixXd& x, double r) {
		return std::abs(x.norm() - r) < 1e-6 ? 0 : INFINITY;
	}

	virtual Eigen::MatrixXd Prox(const Eigen::MatrixXd& x, double t) {
		return (std::abs(t) / x.norm()) * x;
	}
};

template<class Matrix>
class Nothing :public Normlization<Matrix> {
	/*h = 0*/
public:

	virtual double operator()(const Matrix& x) {
		return 0;
	}

	virtual Matrix Prox(const Matrix& x, double t) {
		return x;
	}
};