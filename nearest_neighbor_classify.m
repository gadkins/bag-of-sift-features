%This function will predict the category for every test image by finding
%the training image with most similar features. Instead of 1 nearest
%neighbor, you are requested to vote based on k nearest neighbors which will increase
%performance (although you need to pick a reasonable value for k).

function predicted_categories = nearest_neighbor_classify(train_image_feats, train_labels, test_image_feats)
% image_feats is an N x d matrix, where d is the dimensionality of the
%  feature representation.
% train_labels is an N x 1 cell array, where each entry is a string
%  indicating the ground truth category for each training image.
% test_image_feats is an M x d matrix, where d is the dimensionality of the
%  feature representation. You can assume M = N unless you've modified the
%  starter code.
% predicted_categories is an M x 1 cell array, where each entry is a string
%  indicating the predicted category for each test image.

%{
Useful functions:
 matching_indices = strcmp(string, cell_array_of_strings)
   This can tell you which indices in train_labels match a particular
   category. Not necessary for simple one nearest neighbor classifier.

 D = vl_alldist2(X,Y) 
    http://www.vlfeat.org/matlab/vl_alldist2.html
    returns the pairwise distance matrix D of the columns of X and Y. 
    D(i,j) = sum (X(:,i) - Y(:,j)).^2
    Note that vl_feat represents points as columns vs this code (and Matlab
    in general) represents points as rows. So you probably want to use the
    transpose operator ' 
   vl_alldist2 supports different distance metrics which can influence
   performance significantly. The default distance, L2, is fine for images.
   CHI2 tends to work well for histograms.
 
  [Y,I] = MIN(X) if you're only doing 1 nearest neighbor, or
  [Y,I] = SORT(X) if you're going to be reasoning about many nearest
  neighbors 

%}

label_strings = unique(train_labels);
N_feats = size(test_image_feats, 1);
k = 10;

D = vl_alldist2(train_image_feats', test_image_feats');
[~, idx] = sort(D, 1);

votes = zeros(size(label_strings, 1), N_feats);
for i = 1:N_feats
    for j = 1:size(label_strings, 1)
        nearest_votes = train_labels(idx(1:k, i));
        votes(j, i) = sum(strcmp(label_strings(j), nearest_votes));
    end
end
[~, max_idx] = max(votes, [], 1);
predicted_categories = label_strings(max_idx);


end    

