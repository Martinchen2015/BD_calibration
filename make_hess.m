%% make hessian
m = size(Y);
if (numel(m) > 2)
    slice = m(3); m = m(1:2);
else
    slice = 1;
end
RA_hat = zeros([m slice]);
for i = 1:slice         % cconvfft2(A,A,m,'left') in freq. dom.
    tmp = fft2(A(:,:,i),m(1),m(2));     
    RA_hat(:,:,i) = conj(tmp).*tmp;
end
RA_t = zeros(size(RA_hat));
for k = 1:slice
RA_t(:,:,k) = ifft2(RA_hat(:,:,k));
cache = zeros(size(RA_hat,1),size(RA_hat,1),size(RA_hat,2));
%hess_ul = zeros(size(RA_gen,1));
for i = 1:size(cache,3)
    tmp = flip(RA_t(:,i,k));
    length = size(tmp,1);
    for j = 1:length
        cache(j,j+1:length,i) = (tmp(1:length-j))';
        cache(j,1:j,i) = (tmp(length-j+1:length))';
        %pause();
    end
end
hess_ul_tmp = [];
hess_ul = [];
length = size(RA_t,2);
for i = 1:size(RA_t,2)
    tmp = [];
    for j = 1:size(RA_t,2)
        j_p = length + 1 - j;
        index = mod((j_p + i),length);
        if index == 0
            index = length;
        end
        tmp = [tmp cache(:,:,index)];
    end
    hess_ul_tmp = [hess_ul_tmp;tmp];
end
if k == 1
    hess_ul = hess_ul_tmp;
else
    hess_ul = hess_ul + hess_ul_tmp;
end
end

hess_ur = zeros(size(hess_ul_tmp,1),slice);
A_sum = zeros(slice,1);
for i = 1:slice
    A_sum(i) = sum(sum(A(:,:,i)));
end
for i = 1:slice
    hess_ur(:,i) = A_sum(i);
end

hess_dl = zeros(slice,size(hess_ul_tmp,1));
for i = 1:slice
    hess_dl(i,:) = A_sum(i);
end

hess_dr = eye(slice)*prod(m);

hess = [hess_ul,hess_ur;hess_dl,hess_dr];

%% compare with Hww_function