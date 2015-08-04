function [Aout, Xout, stats] = BD2_Cali_main(Y0, k, lamstruct, varargin)
% main fucntion of calculate the A and X with a biased Y
% Defaults for the options:
mu = 1e-6;              % Approximation quality of the sparsity promoter.
kplus = ceil(0.5*k);    % For Phase II/III: k2 = k + 2*kplus.
method = 'TR';          % Solver for optimizing over the sphere.
maxit = 20;             % Maximum number of iterations for the solver.
dispfun = @(Y, a, X, k, kplus, idx) 0;   % Display/plotting function.

% Application-specific options:
zeropad = true;         % Zero-pad the observation Y0 to avoid border effects.
center = true;          % Return the center of the activations rather than the corners.
signflip = true;        % Attempt to sign-flip the activation map so signal mass is +ve.

%% process input arguments
addpath('./helpers');
starttime = tic;
nvarargin = numel(varargin);

% Process the lambda structure:
flag2 = [isfield(lamstruct, 'lam2dec') && ~isempty(lamstruct.lam2dec) ;
	isfield(lamstruct, 'lambda2_end') && ~isempty(lamstruct.lambda2_end) ];

if xor(sum(flag2), prod(flag2))
    warning('Phase II ignored as either lam2dec or lambda2_end was not properly specified.');
end

flag2 = prod(flag2);
flag3 = isfield(lamstruct, 'lambda3') && ~isempty(lamstruct.lambda3);

% Accept user-specified options:
if (nvarargin == 1)
    if isfield(varargin{1}, 'mu')
        mu = varargin{1}.mu;
    end
    if isfield(varargin{1}, 'kplus')
        kplus = varargin{1}.kplus;
    end
    if isfield(varargin{1}, 'dispfun')
        dispfun = varargin{1}.dispfun;
    end
    if isfield(varargin{1}, 'method')
        method = varargin{1}.method;
    end
    if isfield(varargin{1}, 'maxit')
        maxit = varargin{1}.maxit;
    end
    
    if isfield(varargin{1}, 'zeropad')
        zeropad = varargin{1}.zeropad;
    end
    if isfield(varargin{1}, 'center')
        center = varargin{1}.center;
    end
    if isfield(varargin{1}, 'signflip')
        signflip = varargin{1}.signflip;
    end
else
    error('Too many input arguments.');
end

% Get sizes from Y and zero-pad:
m0 = size(Y0);
if (numel(m0) > 2)
    n = m0(3); m0 = m0(1:2);
else
    n = 1;
end

if zeropad
    m = m0 + k - 1;
    Y = zeros([m n]);
    Y(1:m0(1), 1:m0(2), :) = Y0;
else
    Y = Y0;
end
clear Y0;

% Set up display functions for each phase:
k2 = k + 2*kplus;
dispfun1 = @(a, X) dispfun(Y, a, X, k, [], 1);
dispfun23 = @(a, X) dispfun(Y, a, X, k2, kplus, 1);

end