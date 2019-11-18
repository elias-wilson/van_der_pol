function x = vdp_(x,k,c,m)

x = [x(2); -2*c/m*(x(1)^2-1)*x(2)-k/m*x(1)];
end