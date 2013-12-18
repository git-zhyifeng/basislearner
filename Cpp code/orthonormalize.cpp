#include <iostream>
#include <algorithm>
#include <cmath>

#include "armadillo"

using namespace std;
using namespace arma;

mat orthonormalize(mat G, int width)
{
	mat L;
	vec D;
	mat W;
	svd_econ(L, D, W, G);
	W = W.cols(find(D > 0.0001));
	W = W.cols(0, std::min(width, (int)W.n_cols) - 1);
	mat B = G * W;
	W = W / repmat(sqrt(sum(pow(B,2) , 0)), W.n_rows, 1);
	return W;
}

int main()
{
    mat A;
	A << 1 << 2 << 3 << endr
	  << 4 << 5 << 6 << endr
	  << 7 << 8 << 9 << endr;
	cout << orthonormalize(A, 2) << endl;
	return 0;
}
