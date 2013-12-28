#include <iostream>
#include <string>

#include "armadillo"

using namespace std;
using namespace arma;


void ind2sub(urowvec &I, urowvec &J, int row_size, int col_size, uvec inds)
{
	int size = row_size * col_size;
	for (int i = 0; i < size; i++)
	{
		I(i) = inds(i) % row_size;
		J(i) = inds(i) / row_size;
	}
}

mat orth(mat A)
{
	mat U;
	vec s;
	mat V;
	svd(U, s, V, A);
	double tol  = std::max(A.n_rows, A.n_cols) * s(0) * datum::eps;
	int rank = ((uvec)find(s > tol)).n_elem;
	mat retval;
	if (rank > 0){
		retval = -U.cols(0, rank - 1);
	}
	else{
		retval = zeros<mat>(A.n_rows, 0);
	}
	return retval;
}


// function [W] = orthonormalize_unsup(G,width)
// Orthonormalizes G and return weight matrix W such that 
// G*W is is orthonormal, corresponding to largest 'width' singular values
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

//function orthonormalizes G and return weight matrix W such that
//G*W is orthonormal, corresponding to largest 'width' singular values
//uses approximate SVD: G*W is still orthonormal, but only approximately
//corresponds to the 'width' largest singular values
mat orthonomalize_approx(mat G, int width)
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
	svd_econ(L, D, W, orth(Y).t() * G);
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
	return W;
}


// function F = BuildNetwork(X,Y,widths,trainend)
// Build network for classification.
// Input:
// - Instances X (each row is an instance);
// - Target values vector Y (either binary, e.g. {-1,+1}, or multiclass, e.g. {1,2,3,..})
// - widths vector - width(t) is desired width of layer (t)
// - trainend - X(1:trainend,:) is assumed to be training instances, and
// X(trainend+1:end,:) is assumed to be test instances
// Output: A matrix F, so that F(i,j) is the response of element j on
// instance X(i,:)
mat buildNetwork(mat X, ivec Y, urowvec widths, int trainend)
{
	//Internal parameters
	//{'exact', 'approx'}:Can be either using exact SVD or approximate SVD
	//(less precise but faster for large problems)
	string buildMethodFirstLayer = "exact";
	int batchSize = 50;
	double tol = 1e-9; //Coarse tolernace bound to avoid rounding errors

	cout << "Building F for widths:[" << widths << "]" << endl;

	X = X.cols(find(sum(abs(X), 0) > 0)); //eliminate zero coordinates
	ivec Ytrain = Y.subvec(0, trainend - 1); //just need the training coordinates
	ivec classLabels = unique(Ytrain);
	mat F = zeros<mat>(X.n_rows, sum(widths));
	mat OF = zeros<mat>(trainend, F.n_cols); //maintain orthonormal basis of F

	//creat input layer
	mat W;
	if (buildMethodFirstLayer == "exact"){
		W = orthonormalize(join_rows(ones<mat>(trainend, 1), X.rows(0, trainend - 1)), (int)widths(0));
	}
	else if (buildMethodFirstLayer == "approx"){
		W = orthonomalize_approx(join_rows(ones<mat>(trainend, 1), X.rows(0, trainend - 1)), (int)widths(0));
	}
	else{
		cout << "Unknow BuildMethodFirstLayer!" << endl;
	}
	
	F.cols(0, (int)widths(0) - 1) = join_rows(ones<mat>(X.n_rows, 1), X) * W;
	OF.cols(0, (int)widths(0) - 1) = F.submat(0, 0, trainend - 1, (int)widths(0) - 1);
	F.cols(0, (int)widths(0) - 1) = F.cols(0, (int)widths(0) - 1) / repmat(sqrt(mean(pow(F.submat(0, 0, trainend - 1, (int)widths(0) - 1), 2))), F.n_rows, 1); //Normalize

	//create intermediate layers
	for (int t = 1; t < widths.n_elem; t++){
		int beginThis = sum(widths.subvec(0, t - 1)); //column of F where this layer should go to.
		int beginLast;
		if (t - 2 >= 0){
	    	beginLast = sum(widths.subvec(0, t - 2));
		}
		else{
			beginLast = 0;
		}
		mat V;
		if (classLabels.n_elem > 2){ //multiclass create indicator matrix v for class types
			V = zeros<mat>(trainend, classLabels.n_elem);
			uvec index = zeros<uvec>(1);
			for (int i = 1; i <= classLabels.n_elem; i++){
				V(find(Ytrain == i), index.fill(i - 1)).fill(1);
			}
		}
		else{ //binary: create sign vector V representing class
			umat tempResult = Ytrain == classLabels(0);
			V = sign(conv_to<mat>::from(tempResult) - 0.1);
		}
		V = V - OF.cols(0, beginThis - 1) * (OF.cols(0, beginThis - 1).t() * V);
		int r = beginThis;
		mat OV;
		while (r < beginThis + widths(t) - 1){ //begin mini-batch constructions
		    //cout << "the 1th log in while(r <= beginThis + widths(t))" << endl;
			mat scores = zeros<mat>(widths(0), widths(t - 1));
			OV = orth(V);
			mat Ci, normCi;
			uvec rowinds = zeros<uvec>(1);
			//cout << "the 2th log in while(r <= beginThis + widths(t))" << endl;
			for(int i = 0; i < widths(0); i++){ //compute scores - one row of the scores matrix at a time
				Ci = repmat(F.rows(0, trainend - 1).col(i), 1, widths(t - 1)) % F.submat(0, beginLast , trainend - 1, beginThis - 1); //注意下标
				//cout << "the 1th log in for(int i =0; i < widths(0); i++)" << endl;
				Ci = Ci - OF.cols(0, r - 1) * (OF.cols(0, r -1).t() * Ci);
				//cout << "the 2th log in for(int i =0; i < widths(0); i++)" << endl;
				normCi = sqrt(sum(pow(Ci, 2), 0));
				//cout << "the 3th log in for(int i =0; i < widths(0); i++)" << endl;
				Ci = Ci / repmat(normCi, trainend, 1);
				//cout << "the 4th log in for(int i =0; i < widths(0); i++)" << endl;
				scores.row(i) = sum(pow(OV.t() * Ci, 2), 0);
				//cout << "the 5th log in for(int i =0; i < widths(0); i++)" << endl;
				scores.submat(rowinds.fill(i), find(normCi < tol)).fill(-datum::inf);  //exclude vectors in (or almost in) the span of OF
			}
			//cout << "the 3th log in while(r <= beginThis + widths(t))" << endl;
			vec scoresVec = vectorise(scores,0);
			uvec index = sort_index(scoresVec, "descend");
			urowvec I = zeros<urowvec>(scores.n_rows * scores.n_cols);
			urowvec J = zeros<urowvec>(scores.n_rows * scores.n_cols);
			ind2sub(I, J, scores.n_rows, scores.n_cols, index);
            int numNewColumns = std::min(batchSize, beginThis + (int)widths(t) - r);
			mat Cchosen = zeros<mat>(trainend, numNewColumns);
			mat OC = zeros<mat>(Cchosen.n_rows, Cchosen.n_cols);
			int l = 0;
			int ind = 0; // 注意下标
			double normOCl;
			//cout << "the 4th log in while(r <= beginThis + widths(t))" << endl;
			while (l < numNewColumns){
				//cout << "the 1th log in while(l < numNewColumns)" << endl;
				F.col(r + l) = F.col(I(ind)) % F.col(beginLast + J(ind));    // change r - 1 + l  to r + l and change beginLast + J(ind) - 1 to beginLast + J(ind)
				//cout << "the 2th log in while(l < numNewColumns)" << endl;
				Cchosen.col(l) = F(span(0, trainend - 1), r + l);  // change r - 1 + l  to r + l
				//cout << "the 3th log in while(l < numNewColumns)" << endl;
				OC.col(l) = Cchosen.col(l) - OF.cols(0, r - 1) * (OF.cols(0, r - 1).t() * Cchosen.col(l));
				//cout << "the 4th log in while(l < numNewColumns)" << endl;
				if (l - 1 >= 0){
					OC.col(l) = OC.col(l) - OC.cols(0, l - 1) * (OC.cols(0, l - 1).t() * OC.col(l));
				}			
				//cout << "the 5th log in while(l < numNewColumns)" << endl;
				normOCl = norm(OC.col(l), 2);
				//cout << "the 6th log in while(l < numNewColumns)" << endl;
			    if (normOCl > tol){ //accept new vector if it's linearly independent from previous ones
					OC.col(l) = OC.col(l) / normOCl;
					F.col(r + l) = F.col(r + l) / as_scalar(sqrt(mean(pow(F(span(0, trainend - 1), r + l),2)))); // change r - 1 + l to  r + l
					l++;
			    }
				ind++;
				//cout << "the 7th log in while(l < numNewColumns)" << endl;
				if (ind > I.n_elem){ //exhausted all candidate vectors
					if (l == 1){ // no candidates at all found
						cout << "No linearly-independent candidates were found in mini-batch" << endl;
					}
					else{
						cout << "Warning: Only " << l - 1 << " (out of mini-batch size " << batchSize 
							 << ") linearly-independent candidate vectors were found and added." << endl; 
					}
					break;
				}
				//cout << "the 8th log in while(l < numNewColumns)" << endl;
			}
			//cout << "the 5th log in while(r <= beginThis + widths(t))" << endl;
			if (normOCl > tol){ //if l was again incremented in last loop previously
				l--;
			}
			//cout << "l = " << l << endl;
			OF.cols(r, r + l) = OC.cols(0, l);  //change l - 1 to l
			//cout << "the 6th log in while(r <= beginThis + widths(t))" << endl;
			V = V - OC.cols(0, l) * (OC.cols(0, l).t() * V);
			//cout << "the 7th log in while(r <= beginThis + widths(t))" << endl;
			r = r + l;
			//OF(:,1:r - 1) = orth(OF.cols(0, r - 2)) // possibly re-orthogonalize for numerical stability
			cout << "Build " << std::min((int)widths(t), r - beginThis) << " out of " << widths(t)
				 << " elements in layer " << t + 1 << endl;
		}
	}
	return F;
}

int main()
{
	/*mat F;
	F << 1 << 2 << 3 << 4 << endr
	  << 2 << 3 << 4 << 5 << endr
	  << 3 << 4 << 5 << 6 << endr
	  << 4 << 5 << 6 << 7 << endr;
	F.cols(0,1) = randn<mat>(4,2);
	cout << F << endl;*/
	mat X;
	X.load("X.txt");
	ivec Y;
	Y.load("Y.txt");
	urowvec widths = zeros<urowvec>(3);
	widths.fill(10);
	int trainend = 250;
	mat F = buildNetwork(X, Y, widths, trainend);
	//F.save("F.txt",arma_ascii);
	cout.precision(16);
	F.raw_print(cout,"F = ");
	return 0;
}
