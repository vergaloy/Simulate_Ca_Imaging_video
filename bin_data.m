function M = bin_data(A, sf, binsize)
% BIN_DATA Bins the input data matrix by summing values over specified bins.
%
% INPUTS:
%   A: Input data matrix to be binned.
%
%   sf: Sampling frequency of the data.
%
%   binsize: Size of each bin in seconds.
%
% OUTPUT:
%   M: Binned data matrix.

% Calculate the number of data points in each bin
k = binsize * sf;

% Define the block size for block processing
blockSize = [size(A, 1), k];

% Define the function to apply within each block (summing along the columns)
sumFilterFunction = @(theBlockStructure) sum(theBlockStructure.data, 2);

% Apply block processing to the input data matrix, summing values within each block
M = blockproc(full(A), blockSize, sumFilterFunction);
