#pragma once
#include<iostream>
#include<functional>
#include<cmath>
#include<cstdlib>
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
	virtual const double Lipschitz() = 0;
};

/*
	h(x)
*/
template<class Matrix>
class Normlization {

public:
	virtual double operator()(const Matrix& x) { return 0; }
	virtual Matrix Prox(const Matrix &x, double t) = 0;

};


template<class Matrix>
class LineSearch {
	
public:
	virtual double Line_Search_Result(Matrix x, Matrix Direction, double t0) = 0;
};


template<class Matrix>
class BB {

public:
	virtual double BB_Step_Length(double t_pre, const Matrix& x, const Matrix& x_pre, const Matrix& g, const Matrix& g_pre, int iter) = 0;
};


template<class Matrix>
class Output {

public:
	Matrix solution;
	std::vector<double> f_val;
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
	BB<Matrix>* bb; //BB��������
	double mu;

	double tol;  //����
	double t0;	//��ʼ����
	int max_iter; // ����������
public:

	Problem() {};
	Problem(Object<Matrix>* obj, Normlization<Matrix>* nor, const double _mu, LineSearch<Matrix>* ls, BB<Matrix>* _bb) { _Reset(obj, nor,_mu, ls, _bb); };
	void _Reset(Object<Matrix>* obj, Normlization<Matrix>* nor,const double _mu, LineSearch<Matrix>* ls, BB<Matrix>* _bb);
	Output<Matrix> Basic_Solve(const Matrix& x0);
	Output<Matrix> FISTA_Solve(const Matrix& x0);
	Output<Matrix> Nesterov_Solve(const Matrix& x0);
	Output<Matrix> Inertial_Solve(const Matrix& x0);
	inline bool stopCriteria(const Matrix& x, const Matrix& x_pre, const std::vector<double>& f_val, double tol, int iter);
	
	/*���ò���*/
	void Set_Tol(const double _tol) { tol = std::max(_tol,0.0); }
	void Set_Max_Iter(const int iter) { max_iter = std::max(iter, 0); }
	void Set_T0(const double _t0) { t0 = std::max(_t0, 0.0); }
	void Reset_mu(const double _mu) { mu = _mu; }
};
 



/*������ Problem������*/
// ��Ϊtemplate���ܷ���ʵ�֡�����
template<class Matrix> void Problem<Matrix>::_Reset(Object<Matrix>* obj, Normlization<Matrix>* nor, const double _mu, LineSearch<Matrix>* ls, BB<Matrix>* _bb) {
	f = obj;
	h = nor;
	Ls = ls;
	mu = _mu;
	tol = 1e-9;
	max_iter = 10000;
	t0 = 1/obj->Lipschitz(); //TODO:Ӧ��Ϊ1/L������L��f(x)������ϣ�ĳ�����Ӧ��f(x)�Ľӿ��ṩ��
	bb = _bb;
}

template<class Matrix> Output<Matrix> Problem<Matrix>::Basic_Solve(const Matrix& x0) {

	int iter = 0;
	bool End_Flag = false; // ����������֮ǰ�ɹ�ֹͣ

	Matrix x = x0;
	Matrix x_pre = x;
	Matrix gd = x;
	Matrix gd_pre = gd;
	double obj_pre = 2147483647;
	double obj = f->f(x) + mu * (*h)(x);
	double t = t0;
	Output<Matrix> ans;
	ans.f_val.reserve(max_iter);

	/*���Ƶ��㷨*/
	for (; iter < max_iter; iter++) {
		
		gd_pre = gd;
		x_pre = x;
		obj_pre = obj;
		gd = f->f_gradient(x); // �ݶ�
		t = Ls->Line_Search_Result(x, -gd, t); // ������

		x = h->Prox(x - t * gd, mu*t);
		obj = f->f(x) + mu * (*h)(x);
		t = bb->BB_Step_Length(t, x, x_pre, gd, gd_pre, iter); //BB����
		t = std::min(1e5, std::max(t, t0)); //BB�����ض��ڹ̶���������

		ans.f_val.push_back(obj);
#ifdef VERBOSE
		std::cout << "iter " << iter << ", obj " << obj << ", t" << t << ", norm " << (x - x_pre).norm() << std::endl;
#endif // VERBOSE
		if (stopCriteria(x, x_pre, ans.f_val, tol, iter)) {
			End_Flag = true;
			break;
		}
		//system("pause");
	}

	/*δ��������*/
	if (!End_Flag) {
		std::cerr << "[Problem::Solve] didn't converge before max iter " << max_iter << std::endl;
	}

	/*���������Ϣ*/
	ans.iter = iter;
	ans.obj_fval = obj;
	ans.solution = x;

	return ans;
}

template<class Matrix> Output<Matrix> Problem<Matrix>::FISTA_Solve(const Matrix& x0) {
	int iter = 0;
	bool End_Flag = false; // ����������֮ǰ�ɹ�ֹͣ

	Matrix x = x0, x_pre = x;
	Matrix gd = x, gd_pre = gd;
	Matrix y = x, y_pre = y;
	Matrix v = x;
	double theta = 1.0, theta_pre = theta;
	double t = t0, t_pre = t;
	double eta = 0.2;
	double tmp;
	
	double obj_pre = 2147483647;
	double obj = f->f(x) + mu * (*h)(x);
	Output<Matrix> ans;
	ans.f_val.reserve(max_iter);
	std::cout << "FISTA�㷨Ĭ�ϲ�����������BB������\n";

	for (; iter < max_iter; iter++) {

		gd_pre = gd, x_pre = x, obj_pre = obj, theta_pre = theta, y_pre = y;
		theta = 2.0 / double(iter + 1);
		y = (1 - theta) * x_pre + theta * v;
		gd = f->f_gradient(y); // �ݶ�
		x = h->Prox(y - t * gd, mu * t);
		tmp = f->f(y);

		for (int ls = 0; ls < 5; ++ls) {
			if (f->f(x) <= tmp + (gd * (x - y)).array().sum() + 0.5 / t * (x - y).squaredNorm()) break;
			t = t * eta;
			x = h->Prox(y - t * gd, mu * t);
		}
		t_pre = t;


		obj = f->f(x) + mu * (*h)(x);
		t = bb->BB_Step_Length(t, y, y_pre, gd, gd_pre, iter); //BB����
		t = std::min(1e5, std::max(t, t0)); //BB�����ض��ڹ̶���������
		v = x_pre + 1.0 / theta * (x - x_pre);

		ans.f_val.push_back(obj);
#ifdef VERBOSE
		std::cout << "iter " << iter << ", obj " << obj << ", t" << t << ", norm " << (x - x_pre).norm() << std::endl;
#endif // VERBOSE
		if (stopCriteria(x, x_pre, ans.f_val, tol, iter)) {
			End_Flag = true;
			break;
		}
		//system("pause");
	}

	/*δ��������*/
	if (!End_Flag) {
		std::cerr << "[Problem::Solve] didn't converge before max iter " << max_iter << std::endl;
	}

	/*���������Ϣ*/
	ans.iter = iter;
	ans.obj_fval = obj;
	ans.solution = x;

	return ans;
}

template<class Matrix> Output<Matrix> Problem<Matrix>::Nesterov_Solve(const Matrix& x0) {
	int iter = 0;
	bool End_Flag = false; // ����������֮ǰ�ɹ�ֹͣ

	Matrix x = x0, x_pre = x;
	Matrix gd = x, gd_pre = gd;
	Matrix y = x;
	Matrix v = x, v_pre = v;
	double theta = 1.0;
	double t = t0;

	double obj_pre = 2147483647;
	double obj = f->f(x) + mu * (*h)(x);
	Output<Matrix> ans;
	ans.f_val.reserve(max_iter);
	std::cout << "�ڶ���Nesterov�㷨Ĭ�ϲ�������������BB������\n";

	for (; iter < max_iter; iter++) {

		gd_pre = gd, x_pre = x, obj_pre = obj, v_pre = v;
		theta = 2.0 / double(iter + 1);

		y = (1 - theta) * x + theta * v;
		gd = f->f_gradient(y); // �ݶ�
		v = h->Prox(v_pre - t / theta * gd, t / theta * mu);
		x = (1 - theta) * x_pre + theta * v;

		obj = f->f(x) + mu * (*h)(x);
		//t = bb->BB_Step_Length(t, x, y, gd, gd_pre, iter); //BB����
		//t = std::min(1e5, std::max(t, t0)); //BB�����ض��ڹ̶���������

		ans.f_val.push_back(obj);
#ifdef VERBOSE
		std::cout << "iter " << iter << ", obj " << obj << ", t" << t << ", norm " << (x - x_pre).norm() << std::endl;
#endif // VERBOSE
		if (stopCriteria(x, x_pre, ans.f_val, tol * 300, iter)) {
			End_Flag = true;
			break;
		}
		//system("pause");
	}

	/*δ��������*/
	if (!End_Flag) {
		std::cerr << "[Problem::Solve] didn't converge before max iter " << max_iter << std::endl;
	}

	/*���������Ϣ*/
	ans.iter = iter;
	ans.obj_fval = obj;
	ans.solution = x;

	return ans;
}

template<class Matrix> Output<Matrix> Problem<Matrix>::Inertial_Solve(const Matrix& x0) {
	int iter = 0;
	bool End_Flag = false; // ����������֮ǰ�ɹ�ֹͣ

	Matrix x = x0, x_pre = x, x_prepre = x;
	Matrix gd = x, gd_pre = gd;
	double t = t0;
	double beta = 0.1;

	double obj_pre = 2147483647;
	double obj = f->f(x) + mu * (*h)(x);
	Output<Matrix> ans;
	ans.f_val.reserve(max_iter);

	for (; iter < max_iter; iter++) {

		gd_pre = gd,x_prepre = x_pre, x_pre = x,  obj_pre = obj;

		gd = f->f_gradient(x); // �ݶ�
		t = Ls->Line_Search_Result(x, -gd, t); // ������
		x = h->Prox(x_pre - t * gd + beta * (x_pre - x_prepre), mu * t);

		obj = f->f(x) + mu * (*h)(x);
		t = bb->BB_Step_Length(t, x, x_pre, gd, gd_pre, iter); //BB����
		t = std::min(1e5, std::max(t, t0)); //BB�����ض��ڹ̶���������

		ans.f_val.push_back(obj);

#ifdef VERBOSE
		std::cout << "iter " << iter << ", obj " << obj << ", t" << t << ", norm " << (x - x_pre).norm() << std::endl;
#endif // VERBOSE

		if (stopCriteria(x, x_pre, ans.f_val, tol, iter)) {
			End_Flag = true;
			break;
		}

		//system("pause");
	}

	/*δ��������*/
	if (!End_Flag) {
		std::cerr << "[Problem::Solve] didn't converge before max iter " << max_iter << std::endl;
	}

	/*���������Ϣ*/
	ans.iter = iter;
	ans.obj_fval = obj;
	ans.solution = x;

	return ans;
}


template<class Matrix>
inline bool Problem<Matrix>::stopCriteria(const Matrix& x, const Matrix& x_pre, const std::vector<double>& f_val, double tol, int iter) {
	if ((x - x_pre).norm() < tol * x.norm() && std::abs(f_val[iter] - f_val[iter - 1]) < tol * std::abs(f_val[iter])) return true;
	if ((x - x_pre).norm() < tol && std::abs(f_val[iter] - f_val[iter - 1]) < tol) return true;
	if (iter < 20) return false;
	double min = f_val[iter];
	for (int i = iter - 1; i >= iter - 19; --i)
		if (f_val[i] < min)
			min = f_val[i];
	if (f_val[iter - 20] < min) return true;
	return false;
}