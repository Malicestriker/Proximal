#pragma once
#include "problem.h"
#include "Eigen/dense"
#include "File.h"
#include <random>
template<class Matrix> 
class Least_Square : public Object<Matrix> {

	Eigen::MatrixXd A;
	Matrix b;
	Eigen::MatrixXd AA;
	
	
public:
	

	double obj(const Matrix x) {
		//std::cout << A.rows() << A.cols() << x.rows() << x.cols() << b.rows() << b.cols();
		return  0.5*(A * x - b).squaredNorm();
	}

	Matrix obj_g(const Matrix x) {
		return AA * x - A.transpose() * b;
	}

	void _Reset(const Eigen::MatrixXd & _A, const Matrix& _b) {
		using namespace std::placeholders;
		A = _A;
		b = _b;
		AA = A.transpose() * A;
		this->f = std::bind(&Least_Square::obj,this,_1);
		this->f_gradient = std::bind(&Least_Square::obj_g, this, _1);
	}

	Least_Square(const std::string & file_name){
		std::ifstream input(file_name);
		Eigen::MatrixXd A, b;
		Read_From_File(A, input);
		Read_From_File(b, input);
		_Reset(A, b);
	}
	
	virtual const std::tuple<int,int> Solution_Size() {
		return { A.cols(), b.cols() };
	}

	Matrix Residual(const Matrix& x) {
		return A * x - b;
	}

	virtual const double Lipschitz() {
		Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(AA);
		return 0.5*es.eigenvalues()(AA.rows() - 1);
	}
};


template<class Matrix>
class Logistic : public Object<Matrix> {

	Eigen::MatrixXd A;
	double m;
public:
	double obj(const Matrix& x) {
		return Eigen::log(Eigen::exp((A * x).array()) + 1).sum() / m;
	}

	Matrix obj_g(const Matrix& x) {
		auto r = (1 - (1 + Eigen::exp((A * x).array())).inverse()).matrix();
		return (1 / m) * A.transpose() * r;
	}

	void _Reset(const Eigen::MatrixXd& _a, Eigen::VectorXd& _b) {
		using namespace std::placeholders;
		//assert(_a.cols() = _b.rows());
		A = _a.transpose();
		A = (-1 * (A.array().colwise() * _b.array())).matrix();
		m = _a.cols();
		this->f = std::bind(&Logistic::obj, this, _1);
		this->f_gradient = std::bind(&Logistic::obj_g, this, _1);
	}

	virtual const std::tuple<int, int> Solution_Size() {
		return { A.cols(), 1 };
	}

	Logistic(const std::string& file_name) {
		std::ifstream input(file_name);
		Eigen::MatrixXd _a, _b;
		Read_From_File(_a, input);
		Read_From_File(_b, input);
		_Reset(_a, _b);
	}

	Logistic() {
	
	};
	
	void Test_Init(unsigned int n, unsigned int m, int seed = 97006855) {
		auto raw_generator = std::mt19937_64(seed);
		auto normal_distribution = std::normal_distribution<double>();
		auto generator = [&]() {return normal_distribution(raw_generator); };
		auto a = Eigen::MatrixXd::NullaryExpr(n, m, generator);
		Eigen::VectorXd b = Eigen::VectorXd::NullaryExpr(m, generator);
		_Reset(a, b);
	}

	virtual const double Lipschitz() {

		return 1.0 / m * A.rowwise().norm().sum();
	}
};