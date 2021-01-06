#pragma once
#include "problem.h"
#include "Eigen/dense"
#include "File.h"

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
};