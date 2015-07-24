function [H_v] = Haa_function(v, Y, A, X, beta, lambda, mu, INVTOL, INVIT)
    % apply the hessain on vector v
    m = size(Y);
    k = size(A);
    if numel(m > 2)
        n = m(3);
        m = m(1:2);
        k = k(1:2);
    else
        n = 1;
    end
    
    
end