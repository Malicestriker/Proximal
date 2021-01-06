#include<Eigen/dense>
#include<iostream>
#include "Scenario.h"
#include "Normlizer.h"
#include "LineSearch.h"
int main() {

	// 创建 f(x)
	Least_Square<Eigen::MatrixXd>* LS = new Least_Square<Eigen::MatrixXd>("A.txt");
	const auto [m,n] = LS->Solution_Size();

	Eigen::MatrixXd x = Eigen::MatrixXd::Zero(m, n);

	// 创建 h(x)
	Normlization<Eigen::MatrixXd>* n_12 = new Norm_12();

	// 创建 Line Search
	LineSearch<Eigen::MatrixXd>* BT = new Back_Trace<Eigen::MatrixXd>(LS->f, LS->f_gradient);

	// 合成Problem
	Problem<Eigen::MatrixXd> problem(LS, n_12, 0, BT); // 暂时不测试h(x)

	//得到解
	Output<Eigen::MatrixXd> ans= problem.Solve(x);
	std::cout << LS->Residual(ans.solution);
	return 0;
}