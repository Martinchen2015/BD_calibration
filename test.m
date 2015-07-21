clear; clc; clf;
run('../manopt/importmanopt');      % make sure this is set properly!
addpath('./helpers');

%% Load data:
load('nafeas_26k.mat');

k = [40 40];
m = size(nafeas_26k);   m = m(1:2);

sliceind = [1 19 22 29 38];        n = numel(sliceind);
slicemin = zeros(n,1);
slicerng = zeros(n,1);
slicebias = zeros(n,1);

Y = zeros([m n]);
for i = 1:n
    tempvar = nafeas_26k(:,:,sliceind(i));
    slicemin(i) = min(tempvar(:));
    slicerng(i) = range(tempvar(:)); 
    tempvar = (tempvar - slicemin(i)) / slicerng(i);
    slicebias(i) = median(tempvar(:));
    tempvar = tempvar - slicebias(i);
    
    Y(:,:,i) = tempvar;
end

%% Set up options for the algorithm:
lamstruct.lambda1 = 9e-2;
lamstruct.lam2dec = 2;          % decreasing factor for lambda cont.
lamstruct.lambda2_end = ...     % decrease 3 times
    lamstruct.lambda1/ lamstruct.lam2dec ^ (3-0.1);  
lamstruct.lambda3 = [];         % skip Phase III

% dispfun is a function handle, given to BD_main, that displays updates:
options.dispfun = ...           % the interface is a little wonky at the moment
    @( Y, a, X, k, kplus, idx ) showimsrec( Y, reshape(a, [k n]), X, k, kplus, 3 );

%% Run the algorithm:
[Aout, Xout, stats] = BD2_main( Y, k, lamstruct, options );
