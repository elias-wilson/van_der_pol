function x = vdp_c(x,k,c,m,u)

x = [x(2); -2*c/m*(x(1)^2-1)*x(2)-k/m*x(1)+u];
end