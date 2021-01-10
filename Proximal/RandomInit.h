#pragma once

#include <random>
#include <algorithm>
#include <numeric>
#include <Eigen/Dense>

template<class Matrix>
class Init {
	Eigen::MatrixXd A;
	Matrix b;
	Matrix u;
	Matrix x0;

	void generate_LS(unsigned int m, unsigned int n, unsigned int l, double sparsity, int seed = 97006855) {
		auto raw_generator = std::mt19937_64(seed);
		auto normal_distribution = std::normal_distribution<double>();
		auto generator = [&]() {return normal_distribution(raw_generator); };
		
		A = Eigen::MatrixXd::NullaryExpr(m, n, generator);
		unsigned long k = std::round(n * sparsity);
		std::vector<size_t> p(n);
		std::iota(p.begin(), p.end(), 0);
		std::shuffle(p.begin(), p.end, raw_generator);
		u = Matrix::NullaryExpr(n, l, generator);
		for (size_t i = k; i < n; ++i) u.row(p[i]).setZero();
		b = A * u;
		x0 = Matrix::NullaryExpr(n, l, generator);
	}

	void generate_Logistic(unsigned int m, unsigned int n, int seed = 97006855) {
		auto raw_generator = std::mt19937_64(seed);
		auto normal_distribution = std::normal_distribution<double>();
		auto generator = [&]() { return normal_distribution(raw_generator); };
		A = Eigen::MatrixXd::NullaryExpr(m, n, generator);
		b = Matrix::NullaryExpr(n, l, generator);
		x0 = Matrix::NullaryExpr(n, l, generator);
		u = Matrix::Zero(0, 0);
	}
};