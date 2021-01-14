#pragma once
#include<Eigen/Dense>
#include<fstream>
#include<vector>
#include<string>

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

void Read_From_File_Vec(Eigen::VectorXd& A, std::ifstream& input) {

	int m, n;
	input >> m >> n;
	A.resize(m);
	for (int i = 0; i < m; i++) {
		input >> A(i);
	}
}

void Read_From_File_Vec(Eigen::VectorXd& A, const std::string file_name) {

	std::ifstream input(file_name);
	Read_From_File_Vec(A, input);
}

void Write_To_File_Verbose(Eigen::MatrixXd& A, std::ofstream& output) {

	output << A.rows() << " " << A.cols() << std::endl;
	for (int i = 0; i < A.rows(); i++) {
		for (int j = 0; j < A.cols(); j++) {
			output << A(i, j) << " ";
		}
		output << std::endl;
	}
}

void Write_To_File_Verbose(Eigen::MatrixXd& A, const std::string file_name) {
	std::ofstream output(file_name);
	Write_To_File_Verbose(A, output);
}

void Write_To_File(Eigen::MatrixXd& A, std::ofstream& output) {
	
	for (int i = 0; i < A.rows(); i++) {
		for (int j = 0; j < A.cols(); j++) {
			output << A(i, j) << " ";
		}
		output << std::endl;
	}
}

void Write_To_File(Eigen::MatrixXd& A, const std::string& file_name) {
	std::ofstream output(file_name);
	Write_To_File(A, output);
}

void a9aConvert(const std::string& src, const std::string& dst) {
	std::ifstream input(src);
	std::string tmp;
	int i = -1, maxdim = 0;
	int k;
	std::vector<int> b;
	std::vector<std::vector<int>> a;
	while (true) {
		input >> tmp;
		if (!input) break;
		size_t pos = tmp.find(':');
		if (pos == std::string::npos) {
			++i;
			b.push_back(std::stoi(tmp));
			a.emplace_back();
		}
		else {
			k = std::stoi(tmp.substr(0, pos));
			a[i].push_back(k);
			if (k > maxdim) maxdim = k;
		}
	}
	size_t m = b.size(), n = maxdim;
	Eigen::MatrixXi am = Eigen::MatrixXi::Zero(n, m);
	Eigen::VectorXi bm = Eigen::VectorXi::Zero(m);
	for (i = 0; i < m; ++i) {
		bm(i) = b[i];
		for (auto it : a[i]) am(it, i) = 1;
	}
	std::ofstream output(dst);
	output << n << " " << m << std::endl;
	for (i = 0; i < n; ++i) {
		for (int j = 0; j < m; ++j) {
			output << am(i, j) << " ";
		}
		output << std::endl;
	}
	output << std::endl << m << " " << 1 << std::endl;
	for (i = 0; i < m; ++i) output << bm(i) << std::endl;
	output.close();
}