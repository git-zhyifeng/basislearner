#include <iostream>

#include "armadillo"

using namespace std;
using namespace arma;

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

mat mc_sgd(mat X, ivec Y, double lam)
{
	return NULL;
}

mat sgd_hinge(mat X, ivec Y, double lam)
{
	return NULL;
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

	for (urowvec::const_iterator sz = iterBegin; sz != iterEnd; sz++){
		lambdaCounter = 1;
		for (rowvec::const_iterator lambda = iterBegin_; lambda != iterEnd_; lambda++){
			cout << "Training depth " << szCounter + 1 << "lambda " << *lambda << endl;
			if (lossType == "hinge"){
				w = sgd_hinge(F.submat(0, 0, trainend - 1, *sz - 1), Y.subvec(0, trainend - 1), *lambda);
				mat tempResult = sign(F.cols(0, *sz -1) * w);
				preds = transformResult(tempResult);
			}
			else{
				int minY = Y.min();
				Y = Y - minY + 1;
				w = mc_sgd(F.submat(0, 0, trainend - 1, *sz - 1).t(), Y.subvec(0, trainend - 1), *lambda);
				preds = maxValueIndEachRow(F.cols(0, *sz -1) * w);
			}
			result(szCounter - 1, lambdaCounter - 1, 1) = mean(preds.subvec(0, trainend - 1) != Y.subvec(0, trainend - 1));
			result(szCounter - 1, lambdaCounter - 1, 2) = mean(preds.subvec(trainend, preds.n_elem - 1) != Y.subvec(trainend, Y.n_elem - 1));
			cout << "Train Error: " << result(szCounter - 1, lambdaCounter - 1, 1) << ", "
				 << "Test Error: " << result(szCounter - 1, lambdaCounter - 1, 2) << endl;
			lambdaCounter++;
		}
		szCounter++;
	}

	double bestTestError = result.slice(1).min();
	uvec ind = find(result.slice(1) == bestTestError);
	int i = (int)ind(0) % widths.n_elem;
	int j = (int)ind(0) / widths.n_elem;
	cout << "Best Test Error Result:" << bestTestError << endl;
	cout << "- Architecture [" << widths.subvec(0, i) << "]" << endl;
	cout << "- lambda:" << lambdaRange(j) << " (no." << j << "in lambdaRange" << endl;
}

int main()
{
	return 0;
}
