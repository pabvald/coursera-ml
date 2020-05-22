function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
all_C = [0.01; 0.03; 0.1; 0.3; 1; 3; 10; 30];
all_sigma = [0.01; 0.03; 0.1; 0.3; 1; 3; 10; 30];

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%
i = 1;
error_val = zeros(numel(all_C)*numel(all_sigma),1);

for c = 1:numel(all_C)
  for s = 1:numel(all_sigma)
    model = svmTrain(X, y, all_C(c),  @(x1, x2) gaussianKernel(x1, x2, all_sigma(s)));
    predictions = svmPredict(model, Xval);
    error_val(i) = mean(double(predictions ~= yval));
    i = i +1;
  endfor
endfor

[mn, mindex] = min(error_val);
C = all_C(floor(mindex / numel(all_sigma)) + 1);
sigma = all_sigma(mod(mindex, numel(all_sigma)));

fprintf("\nMínimo C = %d,  sigma = %d,  error = %d\n",  C, sigma, mn);





% =========================================================================

end
