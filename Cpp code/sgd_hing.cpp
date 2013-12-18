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

/*vector<int> randperm(int len)
{
	vector<int> pem(len, -1);
	for (int i = 0; i != len; i++){
		pem[i] = i;
	}
	for (int i = 0; i != len; i++){
		swap(pem[i], pem[randomInterval(i, len)]);
	}
	return pem;
}*/

vector<int> randperm(int len)
{
	irowvec inds;
	inds.load("inds.txt");
	vector<int> v(inds.n_elem, -1);
	for (int i = 0; i < inds.n_elem; i++){
		v[i] = inds(i) - 1;
	}
	return v;
}

//function performs stochastic gradient descent to solve regularized
//hinge loss over data X,Y (with L2 regularization parameter lam).
mat sgd_hinge(mat X, ivec Y, double lam)
{
	X = X.t(); //store examples column-wise to speed training
	double factor = sqrt(2.0 / lam);
	double flag = 1.0;

	int num_epochs = 50; //how many times to go over dataset

	int m = X.n_cols;
	int n = X.n_rows;
	mat w = zeros<mat>(1, n);
	
	int t = 1;
	int ind;
	vector<int> inds = randperm(m);

	cout << "Running SGD, epoch" << endl;
	for (int epoch = 0; epoch < num_epochs; epoch++){
		for (int i = 0; i < inds.size(); i++){
			ind = inds[i];
			double temp = as_scalar(Y(ind) * w * X.col(ind));
			if (flag > temp){
				w = (1 - 1.0 / t) * w + Y(ind) * X.col(ind).t() / (lam * t);
			}
			else{
				w = (1 - 1.0 / t) * w;
			}
			w = std::min(flag, factor / norm(w, 2)) * w;
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
	mat w = sgd_hinge(X, Y, lam);
	cout << w << endl;
	return 0;
}
