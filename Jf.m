function out = Jf(x,k,c,m)
%x = [x(2); -2*c/m*(x(1)^2-1)*x(2)-k/m*x(1)];

out = [0, 1; -4*c/m*x(1)*x(2)-k/m, -2*c/m*(x(1)^2-1)];

end