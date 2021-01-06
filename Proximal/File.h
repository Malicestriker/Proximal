#pragma once
#include<Eigen/Dense>
#include<fstream>

void Read_From_File(Eigen::MatrixXd& A, std::ifstream & input) {

	int m, n;
	input >> m >> n;
	A.resize(m, n);
	for (int i = 0; i < m; i++) {
		for (int j = 0; j < n; j++) {
			input >> A(i,j);
		}
	}
}

void Read_From_File(Eigen::MatrixXd& A, const std::string file_name) {

	std::ifstream input(file_name);
	Read_From_File(A, input);
}