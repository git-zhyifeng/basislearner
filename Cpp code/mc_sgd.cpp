#include <iostream>
#include <vector>
#include <cstdlib>

#include "armadillo"

using namespace std;
using namespace arma;

int randomInterval(int begin, int end)
{
	return begin + rand() % (end - begin);
}

vector<int> randperm(int len)
{
	vector<int> pem(len, -1);
	for (int i = 0; i != len; i++){
		pem[i] = i;
	}
	for (int i = 0; i != len; i++){
		swap(pem[i], pem[randomInterval(i, len)]);
	}
	return pem;
}

/*vector<int> randperm(int len)
{
	irowvec inds;
	inds.load("inds.txt");
	vector<int> v(inds.n_elem, -1);
	for (int i = 0; i < inds.n_elem; i++){
		v[i] = inds(i) - 1;
	}
	return v;
}*/

//solve multiclass SVM using SGD
//Assume that the instances are columns of X
mat mc_sgd(mat X, ivec Y, double lam)
{
	int k = ((ivec)unique(Y)).n_elem;
	Y = Y - Y.min() + 1;

	int num_epochs = 1; //how many times to go over dataset
	int m = X.n_cols;
	int n = X.n_rows;
	mat w = zeros<mat>(k, n);

	int t  = 100;
	int ind;
	uword j;
	double val;
	vec pred;
	vector<int> inds;
	ivec classvec = linspace<ivec>(1,k,k);
	ivec Yelem;
	
	cout << "Runing SGD epoch" << endl;
	for (int epoch = 0; epoch < num_epochs; epoch++){
		inds = randperm(m);
		for (int i = 0; i < inds.size(); i++){
			ind = inds[i];
			pred = w * X.col(ind);
			Yelem = linspace<ivec>(Y(ind),Y(ind),k);
			val = ((vec)((classvec != Yelem) + pred - pred(Y(ind) - 1))).max(j);
			w = w * (1 - 1.0 / t);
			if (val > 0){
				w.row(Y(ind) - 1) = w.row(Y(ind) - 1) + X.col(ind).t() / (lam * t);
				w.row(j) = w.row(j) - X.col(ind).t() / (lam * t);
			}
			t++;
		}
	}

	return w.t();
}

int main()
{
	mat X;
	X.load("mc_x.txt");
	ivec Y;
	Y.load("mc_y.txt");
	double lam = 0.1;
	mat w = mc_sgd(X, Y, lam);
	cout << w << endl;
	return 0;
}
