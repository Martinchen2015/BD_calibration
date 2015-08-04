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

%% Phase I: First pass at BD
Ain = randn([k n]); Ain = Ain/norm(Ain(:));

fprintf('PHASE I: \n=========\n');
[A, Wsol, stats] = BD2_Cali_Manopt(Y, Ain,lamstruct.lambda1, mu, [], dispfun1, method, maxit);
fprintf('\n');

%% Phase II: Lift the sphere and do lambda continuation
if flag2
    A2 = zeros([k2 n]);
    A2(kplus(1)+(1:k(1)),kplus(2)+(1:k(2)),:) = A;
    Wsol2.X = circshift(Wsol.X, -kplus);
    Wsol2.X_dual = circshift(Wsol.X_dual, -kplus);
    Wsol2.beta = Wsol.beta;
    
    lambda2 = lamdtruct.lambda1;
    score = zeros(2*kplus+1);
    fprintf('PHASE II: \n=========\n');
    while lambda2 >= lamstruct.lambda2_end
        fprintf('lambda = %.1e: \n', lambda2);
        [A2, Wsol2, stats] = BD2_Cali_Manopt(Y, Ain, lambda2, mu, Wsol2, dispfun23, method, maxit);
        fprintf('\n');
        %Attempt to 'unshift" the a and x by taking the l1-norm over all k-contiguous elements:
        for tau1 = -kplus(1):kplus(1)
            ind1 = tau1+kplus(1)+1;
            for tau2 = -kplus(2):kplus(2)
                ind2 = tau2+kplus(2)+1;
                temp = A2(ind1:(ind1+k(1)-1), ind2:(ind2+k(2))-1,:);
                score(ind1,ind2) = norm(temp(:), 1);
            end
        end
        [temp,ind1] = max(score); [~,ind2] = max(temp);
        tau = [ind1(ind2) ind2]-kplus-1;
        A2 = circshift(A2,-tau);
        Wsol2.X = circshift(Wsol2.X, tau);
        Wsol2.X_dual = circshift(Wsol2.X_dual, tau);
        Wsol2.beta = Wsol2.beta;
        
        dispfun23(A2,Xsol2.X);
        lambda2 = lambda2/lamstruct.lam2dec;
    end  
end

%% Phase III
if flag3
    fprintf('PHASE III: \n=========\n');
    Xsol3 = wsolve2_pdNCG(Y, A2, lamstruct.lambda3, mu, Xsol2, 1e-6, 2e2);
    dispfun23(A2, Xsol3.X);
else
    Xsol3 = Xsol2;
end
clear Xsol2;

%% final result
Aout = A2(kplus(1)+(1:k(1)), kplus(2)+(1:k(2)), :);
Xout = circshift(Xsol3.X,kplus);
stats.A = A;
stats.A2 = A2;

if signflip
    thresh = 0.2*max(abs(Xout(:)));
    sgn = sign(sum(Xout(abs(Xout) >= thresh)));
    Aout = sgn*Aout;
    Xout = sgn*Xout;
    stats.A = sgn*A;
    stats.A2 = sgn*A2;
end

if center
    Xout = circshift(Xout, ceil(k/2));
end

if zeropad
    Xout = Xout(1:m0(1), 1:m0(2));
end

runtime = toc(starttime);
fprintf('\nDone! Runtime = %.2fs. \n\n', runtime);
end