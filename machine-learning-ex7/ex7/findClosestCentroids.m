function idx = findClosestCentroids(X, centroids)
%FINDCLOSESTCENTROIDS computes the centroid memberships for every example
%   idx = FINDCLOSESTCENTROIDS (X, centroids) returns the closest centroids
%   in idx for a dataset X where each row is a single example. idx = m x 1 
%   vector of centroid assignments (i.e. each entry in range [1..K])
%

%
[n d] = size(X);  % # points and # dimensions
% Set K
K = size(centroids, 1);
assert(size(centroids, 2) == d);
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
norm2 = zeros(n, K);  % squared distance

for i = 1:d
  norm2 = norm2 + (X(:, i) - centroids(:, i)').^2;
end

[_, idx] = min(norm2, [], 2);

% =============================================================

end

