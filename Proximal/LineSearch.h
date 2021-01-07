#pragma once
#include "Problem.h"
#include<functional>

template<class Matrix>
class Back_Trace:public LineSearch<Matrix> {

	using MMfunction_p = std::function<Matrix(Matrix)>; // 函数指针类型
	using Mdfunction_p = std::function<double(Matrix)>; // 目标函数指针

	Mdfunction_p object;// f(x)
	MMfunction_p gradient; // f(x)的导数

	double c1;
	double c2;
public:

	Back_Trace(Mdfunction_p f, MMfunction_p f_g) {
		object = f;
		gradient = f_g;
		c1 = 1e-3;
		c2 = 0.9;
	}

	virtual double Line_Search_Result(Matrix x, Matrix Direction) {
		
		double t = 10;
		double ref = object(x);
		Matrix Gradient = gradient(x);
		double lin = Gradient.cwiseProduct(Direction).sum();

		bool continue_flag = true;
		while (continue_flag) {

			t = t * 0.8;
			continue_flag = false;
			if (object(x + t * Direction) > ref - (1e-4) * ref + c1 * t * lin) continue_flag = true; // Armijo
			else if( gradient(x + t * Direction).cwiseProduct(Direction).sum() < c2 *  lin ) continue_flag = true; // Wolfe
			
			std::cout << "[LineSearch] "<<t<<" " << (object(x + t * Direction) > ref - (1e-4) * ref + c1 * t * lin)
				<< " " << (gradient(x + t * Direction).cwiseProduct(Direction).sum() < c2 * lin) << std::endl;
			std::cout << "[detail] " << ref << " " << gradient(x + t * Direction).cwiseProduct(Direction).sum() << " "<< c2 * lin <<std::endl;
			if (t < 1e-9) break;
		}

		return t;
	}


};