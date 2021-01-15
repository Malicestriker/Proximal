#include<Eigen/dense>
#include<iostream>
#include<ctime>
#include "Scenario.h"
#include "Normlizer.h"
#include "LineSearch.h"
#include "Evaluate.h"

int main() {
	// 创建 f(x)
	Logistic<Eigen::MatrixXd>* LS = new Logistic<Eigen::MatrixXd>("data/a1a.txt");

	const auto [m, n] = LS->Solution_Size();


	// 创建 h(x)
	Normlization<Eigen::MatrixXd>* n_12 = new Norm_12();

	// 创建 Line Search
	LineSearch<Eigen::MatrixXd>* BT = new Back_Trace<Eigen::MatrixXd>(LS->f, LS->f_gradient);

	// 创建 BB 步长
	BB<Eigen::MatrixXd>* bb = new Alter_BB<Eigen::MatrixXd>();

	int height = 5;

	// 合成Problem
	Problem<Eigen::MatrixXd> problem(LS, n_12, 0.01 / m, BT, bb); // 暂时不测试h(x)


	Eigen::MatrixXd x = Eigen::MatrixXd::Zero(m, n);
	std::vector<double> opt_hist;
	int total_iter = 0;
	auto time = std::clock();
	double fin = 0;
	for (int i = 0; i < height; ++i) {
		double mu = 0.01 / m * pow(10, height - i - 1);
		problem.Reset_mu(mu);
		problem.Set_Tol(1e-8 * pow(100, height - i - 1));
		Output<Eigen::MatrixXd> ans = problem.Basic_Solve(x);
		total_iter += ans.iter + 1;
		opt_hist.insert(opt_hist.end(), ans.f_val.begin(), ans.f_val.end());
		x = ans.solution;
		fin = ans.obj_fval;
	}
	time = std::clock() - time;
	//std::cout << LS->Residual(x);
	std::cout << x << std::endl;
	std::cout << "total iterations: " << total_iter << std::endl;
	std::cout << "time: " << 1.0 * time / CLOCKS_PER_SEC << 's' << std::endl;
	std::cout << "final_value: " << fin << std::endl;
	std::cout << "train_accuracy: " << LS->check_accuracy(x) << std::endl;
	std::cout << "test_accuracy: " << Logistic<Eigen::MatrixXd>("data/a1a_test.txt").check_accuracy(x) << std::endl;
	Write_Trajectory(opt_hist, "output/a1a_Basic.txt");
	return 0;
}