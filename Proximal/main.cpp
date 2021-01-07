#include<Eigen/dense>
#include<iostream>
#include "Scenario.h"
#include "Normlizer.h"
#include "LineSearch.h"
int main() {

	// ���� f(x)
	Least_Square<Eigen::MatrixXd>* LS = new Least_Square<Eigen::MatrixXd>("A.txt");
	const auto [m,n] = LS->Solution_Size();

	Eigen::MatrixXd x = Eigen::MatrixXd::Zero(m, n);

	// ���� h(x)
	Normlization<Eigen::MatrixXd>* n_12 = new Norm_12();

	// ���� Line Search
	LineSearch<Eigen::MatrixXd>* BT = new Back_Trace<Eigen::MatrixXd>(LS->f, LS->f_gradient);

	// ���� BB ����
	BB<Eigen::MatrixXd>* bb = new Alter_BB<Eigen::MatrixXd>();

	// �ϳ�Problem
	Problem<Eigen::MatrixXd> problem(LS, n_12, 0, BT, bb); // ��ʱ������h(x)

	//�õ���
	Output<Eigen::MatrixXd> ans = problem.Basic_Solve(x);
	std::cout << LS->Residual(ans.solution);
	return 0;
}