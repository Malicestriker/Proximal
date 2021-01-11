#pragma once
template<class Matrix>
auto Sparsity(const Matrix& x) -> double {
	return  double((x.array().abs() > 1e-5).count()) / double(x.size());
}
template<class Matrix>
auto Errfun(const Matrix& x, const Matrix& u) -> double {
	return (x - u).norm() / (1 + u.norm());
}