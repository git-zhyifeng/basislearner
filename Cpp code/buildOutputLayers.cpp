#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>

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

ivec maxValueIndEachRow(mat X)
{
	ivec pred = zeros<ivec>(X.n_rows);
	uword ind;
	for (int i = 0; i < X.n_rows; i++){
		((rowvec)X.row(i)).max(ind);
		pred(i) = (int)ind;
	}
	return pred;
}

//solve multiclass SVM using SGD
//Assume that the instances are columns of X
mat mc_sgd(mat X, ivec Y, double lam)
{
	int k = ((ivec)unique(Y)).n_elem;
	Y = Y - Y.min() + 1;

	int num_epochs = 100; //how many times to go over dataset
	int m = X.n_cols;
	int n = X.n_rows;
	mat w = zeros<mat>(k, n);

	int t  = 1;
	int ind;
	uword j;
	double val;
	vec pred;
	vector<int> inds;
	ivec classvec = linspace<ivec>(1,k,k);
	ivec Yelem;
	
	//cout << "Runing SGD epoch" << endl;
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

	//cout << "Running SGD, epoch" << endl;
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

ivec transformResult(mat X)
{
	ivec preds = zeros<ivec>(X.n_rows);
	for (int i = 0; i < X.n_rows; i++){
		preds(i) = (int)X(i,0);
	}
	return preds;
}

cube buildOutputLayers(mat F, ivec Y, int trainend, urowvec widths, rowvec lambdaRange)
{
	cube result = zeros<cube>(widths.n_elem, lambdaRange.n_elem, 2);

	cout << "Building Output Layer\n trainend=" << trainend 
		 << " widths=[" << widths << "], lambdaRange=[" << lambdaRange << "]\n";
	
	string lossType;
	if (((ivec)unique(Y)).n_elem > 2){
		lossType = "multiclass_hinge";
	}
	else{
		lossType = "hinge";
	}

	int szCounter = 1;
	int lambdaCounter = 1;
    urowvec sumwidths = cumsum(widths);
	urowvec::const_iterator iterBegin = sumwidths.begin();
	urowvec::const_iterator iterEnd = sumwidths.end();
	rowvec::const_iterator iterBegin_ = lambdaRange.begin();
	rowvec::const_iterator iterEnd_ = lambdaRange.end();
	mat w; 
	ivec preds;

	//srand((unsigned)time(NULL));
	for (urowvec::const_iterator sz = iterBegin; sz != iterEnd; sz++){
		lambdaCounter = 1;
		for (rowvec::const_iterator lambda = iterBegin_; lambda != iterEnd_; lambda++){
			cout << "Training depth " << szCounter + 1 << " lambda " << *lambda << endl;
			if (lossType == "hinge"){
				w = sgd_hinge(F.submat(0, 0, trainend - 1, *sz - 1), Y.subvec(0, trainend - 1), *lambda);
				mat tempResult = sign(F.cols(0, *sz -1) * w);
				//preds = transformResult(tempResult);
				preds = conv_to<ivec>::from(tempResult);
			}
			else{
				int minY = Y.min();
				Y = Y - minY + 1;
				w = mc_sgd(F.submat(0, 0, trainend - 1, *sz - 1).t(), Y.subvec(0, trainend - 1), *lambda);
				preds = maxValueIndEachRow(F.cols(0, *sz -1) * w);
			}
			result(szCounter - 1, lambdaCounter - 1, 0) = mean(conv_to<vec>::from(preds.subvec(0, trainend - 1) != Y.subvec(0, trainend - 1)));
			result(szCounter - 1, lambdaCounter - 1, 1) = mean(conv_to<vec>::from(preds.subvec(trainend, preds.n_elem - 1) != Y.subvec(trainend, Y.n_elem - 1)));
			cout << "Train Error: " << result(szCounter - 1, lambdaCounter - 1, 0) << ", "
				 << "Test Error: " << result(szCounter - 1, lambdaCounter - 1, 1) << endl;
			lambdaCounter++;
		}
		szCounter++;
	}

	cout << result << endl;
	double bestTestError = result.slice(1).min();
	uvec ind = find(result.slice(1) == bestTestError);
	int i = (int)ind(0) % widths.n_elem;
	int j = (int)ind(0) / widths.n_elem;
	cout << "Best Test Error Result:" << bestTestError << endl;
	cout << "- Architecture [" << widths.subvec(0, i) << "]" << endl;
	cout << "- lambda:" << lambdaRange(j) << " (no." << j + 1 << "in lambdaRange)" << endl;
	
	return result;
}

int main()
{
	mat F;
	F.load("F.txt");
	ivec Y;
	Y.load("Y.txt");
	urowvec widths = zeros<urowvec>(2);
	widths.fill(10);
	int trainend = 250;
	rowvec lambdaRange = zeros<rowvec>(5);
	//rowvec lambdaRange = zeros<rowvec>(1);
	lambdaRange << 0.001 << 0.01 << 0.1 << 1.0 << 10;
	//lambdaRange << 0.001;
	cube result = buildOutputLayers(F, Y, trainend, widths, lambdaRange);
	return 0;
}
