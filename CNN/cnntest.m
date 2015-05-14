function [er, bad] = cnntest(net, x, y)
%CNNTEST Evaluates a trained CNN over the provided test set.
%
%  Parameters
%    x  - Matrix of test images, with dimensions 
%         imgHeight x imgWidth x numImages
%    y  - Correct labels, represented as a binary matrix with 1 row per
%         category and one column per example.
%
%  Returns
%    er - Percentage (0 - 1) of test samples which were misclassified.
%    bad - Indeces of misclassified test samples.

    % Evaluate the CNN over the test samples.
    net = cnnff(net, x);
    
    % For each example, assign the category with the maximum output score.
    [~, h] = max(net.o);
    
    % Convert the test labels from binary vectors into integers.
    [~, a] = max(y);
    
    % Find the indeces of all of the incorrectly classified examples.
    bad = find(h ~= a);

    % Divide the number of incorrect classifications by the number of 
    % test examples to calculate the error.
    er = numel(bad) / size(y, 2);
end
