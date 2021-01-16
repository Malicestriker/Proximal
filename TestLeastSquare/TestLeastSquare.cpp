#include<Eigen/dense>
#include<iostream>
//#define VERBOSE
#include "Scenario.h"
#include "Normlizer.h"
#include "LineSearch.h"
#include "Evaluate.h"
#include <ctime>
int main() {

	// 创建 f(x)
	Least_Square<Eigen::MatrixXd>* LS = new Least_Square<Eigen::MatrixXd>("data/cpusmall.txt");

	// 创建 h(x)
	Norm_12* n_12 = new Norm_12();

	// 设置测试数据
	//LS->Test_Init(256, 512, 2);

	// 得到解的大小
	const auto [m, n] = LS->Solution_Size();
	Eigen::MatrixXd x = Eigen::MatrixXd::Zero(m, n);

	// 创建 Line Search
	LineSearch<Eigen::MatrixXd>* BT = new Back_Trace<Eigen::MatrixXd>(LS->f, LS->f_gradient);

	// 创建 BB 步长
	BB<Eigen::MatrixXd>* bb = new Alter_BB<Eigen::MatrixXd>();



	// 合成Problem
	Problem<Eigen::MatrixXd> problem(LS, n_12, 0, BT, bb); // f(x),h(x),mu,Backtrack,BB

	int height = 5;
	std::vector<double> opt_hist;
	int total_iter = 0;
	auto time = std::clock();
	double fin = 0;
	for (int i = 0; i < height; ++i) {
		double mu = 0.01  * pow(10, height - i - 1);
		problem.Reset_mu(mu);
		problem.Set_Tol(1e-5 * pow(100, height - i - 1));
		Output<Eigen::MatrixXd> ans = problem.FISTA_Solve(x);
		total_iter += ans.iter + 1;
		opt_hist.insert(opt_hist.end(), ans.f_val.begin(), ans.f_val.end());
		x = ans.solution;
		fin = ans.obj_fval;
	}
	time = std::clock() - time;
	//std::cout << LS->Residual(x);
	std::cout << Sparsity(x) << std::endl;
	std::cout << "total iterations: " << total_iter << std::endl;
	std::cout << "time: " << 1.0 * time / CLOCKS_PER_SEC << 's' << std::endl;
	std::cout << "final_value: " << fin << std::endl;
	Write_Trajectory(opt_hist, "output/cpusmall_fista.txt");

	
	std::ifstream test("data/cpusmall_test.txt");
	Eigen::MatrixXd A, b;
	Read_From_File(A,test);
	Read_From_File(b,test);
	/*
	auto t = (A*x).cwiseSign();
	b = b.cwiseSign();
	int ans = 0;
	for (int i = 0; i < A.rows(); i++) {
		if (t(i,0) == b(i,0)) {
			ans++;
		}
	}
	std::cout << ans * 1.0 / A.rows();*/

	std::cout << (A * x - b).squaredNorm()  / b.squaredNorm();
	return 0;
}