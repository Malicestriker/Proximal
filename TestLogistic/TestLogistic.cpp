#include<Eigen/dense>
#include<iostream>
#include "Scenario.h"
#include "Normlizer.h"
#include "LineSearch.h"
#include "Evaluate.h"

int main() {
	// 创建 f(x)
	Logistic<Eigen::MatrixXd>* LS = new Logistic<Eigen::MatrixXd>("a9a.txt");

	const auto [m, n] = LS->Solution_Size();

	Eigen::MatrixXd x = Eigen::MatrixXd::Zero(m, n);

	// 创建 h(x)
	Normlization<Eigen::MatrixXd>* n_12 = new Norm_12();

	// 创建 Line Search
	LineSearch<Eigen::MatrixXd>* BT = new Back_Trace<Eigen::MatrixXd>(LS->f, LS->f_gradient);

	// 创建 BB 步长
	BB<Eigen::MatrixXd>* bb = new No_BB<Eigen::MatrixXd>();

	int height = 5;

	// 合成Problem
	Problem<Eigen::MatrixXd> problem(LS, n_12, 0.01, BT, bb); // 暂时不测试h(x)
	problem.Set_Tol(1e-6);
	Output<Eigen::MatrixXd> ans = problem.Basic_Solve(x);
	x = ans.solution;
	//std::cout << LS->Residual(x);
	std::cout << x << std::endl;
	std::cout << Sparsity(x);

	
	return 0;
}