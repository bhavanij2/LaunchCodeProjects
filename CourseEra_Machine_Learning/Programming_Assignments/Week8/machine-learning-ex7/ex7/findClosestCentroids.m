function idx = findClosestCentroids(X, centroids)
%FINDCLOSESTCENTROIDS computes the centroid memberships for every example
%   idx = FINDCLOSESTCENTROIDS (X, centroids) returns the closest centroids
%   in idx for a dataset X where each row is a single example. idx = m x 1 
%   vector of centroid assignments (i.e. each entry in range [1..K])
%

% Set K
K = size(centroids, 1);

% You need to return the following variables correctly.
idx = zeros(size(X,1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Go over every example, find its closest centroid, and store
%               the index inside idx at the appropriate location.
%               Concretely, idx(i) should contain the index of the centroid
%               closest to example i. Hence, it should be a value in the 
%               range 1..K
%
% Note: You can use a for-loop over the examples to compute this.
%

for sample_index = 1: size(X, 1)
  sample = X(sample_index, :);
  
  centroid_distance = zeros(size(centroids, 1), size(centroids, 2));
  
  for cent_indx = 1: size(centroids, 1)
    centroid  = centroids(cent_indx, :);  
    centroid_distance(cent_indx, :) = abs(sample - centroid);
  endfor

  sq_distance = 0;
  for i = 1: size(centroids, 2)
    sq_distance = sq_distance + centroid_distance(: , i) .^ 2;
  endfor
  sqrt_distance = sqrt(sq_distance);
  
  [val, min_idx] = min(sqrt_distance);  
  idx(sample_index) = min_idx(1);
    
endfor







% =============================================================

end

