#pragma once
#include "problem.h"
#include "Eigen/dense"

/*特例化的 Normalization*/

class Norm_12 :public Normlization<Eigen::MatrixXd> {
	
	/*矩阵1-2范数*/

public:
	virtual Eigen::MatrixXd Prox(const Eigen::MatrixXd& x, double t) {
		Eigen::MatrixXd ans = x;
		int m = x.rows();
		for (int i = 0; i < m; i++) {
			double r_12 = x.row(i).norm();
			double scaler = r_12 < t ? 0 : 1 - t / r_12;
			ans.row(i) *= scaler;
		}
		return ans;
	}

};

template<class Matrix>
class Nothing :public Normlization<Matrix> {
	/*h = 0*/
public:
	virtual Matrix Prox(const Matrix& x, double t) {
		return x;
	}
};