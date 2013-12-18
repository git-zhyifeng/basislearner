#include <iostream>

#include "armadillo"

using namespace std;
using namespace arma;

//function orthonormalizes G and return weight matrix W such that
//G*W is orthonormal, corresponding to largest 'width' singular values
//uses approximate SVD: G*W is still orthonormal, but only approximately
//corresponds to the 'width' largest singular values
mat orthonormalize_approx(mat G, int width)
{
	//internal parameters
	int p = width;
	double tol = 1e-9;

	//perform approximate SVD that with a single quality improvement iteration
	//(multiplying by G*G' once)
	mat Y = G * (G.t() * (G * randn(G.n_cols, width+p)));
	mat L;
	vec D;
	mat W;
	//svd_econ(L, D, W, orth(Y).t() * G);
	W = W.cols(find(D > tol));
	W = W.cols(0, std::min(width, (int)W.n_cols) - 1);

	//correct so that G*W is indeed orthonomal (G*W not exactly orthonormal,
	//since U is only approximate singular vectors of G)
	mat B = G * W;
	mat W2;
	svd_econ(L, D, W2, B);
	B = B * W2;
	W = W * W2;
	mat norms = sqrt(sum(pow(B,2), 0));
	W = W / repmat(norms, W.n_rows, 1);
}

int main()
{
	return 0;
}
