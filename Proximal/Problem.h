#pragma once
#include<iostream>
#include<functional>

/*
	f(x)
*/
template<class Matrix>
class Object {

public:
	using MMfunction_p = std::function<Matrix(Matrix)>; // ����ָ������
	using Mdfunction_p = std::function<double(Matrix)>; // Ŀ�꺯��ָ��

	Mdfunction_p f;// f(x)
	MMfunction_p f_gradient; // f(x)�ĵ���
	virtual const std::tuple<int, int> Solution_Size() = 0;
};

/*
	h(x)
*/
template<class Matrix>
class Normlization {

public:
	virtual Matrix Prox(const Matrix &x, double t) = 0;

};


template<class Matrix>
class LineSearch {
	
public:
	virtual double Line_Search_Result(Matrix x, Matrix Direction) = 0;
};


template<class Matrix>
class Output {

public:
	Matrix solution;
	double obj_fval;
	int iter;
	
};

/*
	f(x) + mu * h(x)
*/
template<class Matrix>
class Problem {

	Object<Matrix>* f;// f(x)
	Normlization<Matrix>* h; // h(x)������
	LineSearch<Matrix>* Ls; // Linesearch ����
	double mu;

	double tol;  //����
	int max_iter; // ����������
public:

	Problem() {};
	Problem(Object<Matrix>* obj, Normlization<Matrix>* nor, const double _mu, LineSearch<Matrix>* ls) { _Reset(obj, nor,_mu, ls); };
	void _Reset(Object<Matrix>* obj, Normlization<Matrix>* nor,const double _mu, LineSearch<Matrix>* ls); 
	Output<Matrix> Solve(const Matrix &x0);
	
	/*���ò���*/
	void Set_Tol(const double _tol) { tol = std::max(_tol,0); }
	void Set_Max_Iter(const int iter) { max_iter = std::max(iter, 0); }
	void Reset_mu(const double _mu) { mu = _mu; }
};
 



/*������ Problem������*/
// ��Ϊtemplate���ܷ���ʵ�֡�����
template<class Matrix> void Problem<Matrix>::_Reset(Object<Matrix>* obj, Normlization<Matrix>* nor, const double _mu, LineSearch<Matrix>* ls) {
	f = obj;
	h = nor;
	Ls = ls;
	mu = _mu;
	tol = 1e-9;
	max_iter = 10000;
}

template<class Matrix> Output<Matrix> Problem<Matrix>::Solve(const Matrix& x0) {

	int iter = 0;
	bool End_Flag = false; // ����������֮ǰ�ɹ�ֹͣ

	Matrix x = x0;
	Matrix x_pre = x;
	double obj_pre = 2147483647;
	double obj = f->f(x);

	/*���Ƶ��㷨*/
	for (int iter = 0; iter < max_iter; iter++) {

		Matrix gd = f->f_gradient(x); // �ݶ�
		double t = Ls->Line_Search_Result(x, -gd); // ������
		x_pre = x;
		obj_pre = obj;

		x = h->Prox(x - t * gd, mu*t);
		obj = f->f(x);

		if ((x - x_pre).norm() < tol * x.norm() && std::abs(obj_pre - obj) < tol * abs(obj)) {
			End_Flag = true;
			break;
		}

		if ((x - x_pre).norm() < tol && std::abs(obj_pre - obj) < tol) {
			End_Flag = true;
			break;
		}

		std::cout << "iter " << iter << ": obj " << obj << " t" << t<< "norm "<< (x - x_pre).norm()<<std::endl;
		//system("pause");
	}

	/*δ��������*/
	if (!End_Flag) {
		std::cerr << "[Problem::Solve] didn't converge before max iter " << max_iter << std::endl;
	}

	/*���������Ϣ*/
	Output<Matrix> ans;
	ans.iter = iter;
	ans.obj_fval = obj;
	ans.solution = x;

	return ans;
}
