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

void Write_To_File_Verbose(Eigen::MatrixXd& A, const std::string file_name) {
	std::ofstream output(file_name);
	output << A.rows() << " " << A.cols()<<std::endl;
	for (int i = 0; i < A.rows(); i++) {
		for (int j = 0; j < A.cols(); j++) {
			output << A(i, j) << " ";
		}
		output << std::endl;
	}
}

void Write_To_File(Eigen::MatrixXd& A, const std::string file_name) {
	std::ofstream output(file_name);
	for (int i = 0; i < A.rows(); i++) {
		for (int j = 0; j < A.cols(); j++) {
			output << A(i, j) << " ";
		}
		output << std::endl;
	}
}