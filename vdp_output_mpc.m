clc;
clear;
close all;


dt = 0.01;
t = 0:dt:10;


x0 = [0 1]';
K = [1 1]'.*ones(2,49);
v_ = zeros(1,49);

x = zeros(2,length(t));
x(:,1) = x0;

z = x;

u = zeros(1,length(t)-1);
v = u;
for ii = 1:length(t)-1
    
    K = fmincon(@(K)J(x(:,ii),K,dt),K);
    
    u(ii) = -K(:,1)'*x(:,ii);
    
    x(:,ii+1) = x(:,ii) + f(@(x,u)vdp__(x,u),x(:,ii),u(ii),dt);
    
%     v_ = fmincon(@(v)L(z(:,ii),v,dt),v_);
%     v(ii) = v_(1);
%     
%     z(:,ii+1) = z(:,ii) + f(@(z,v)vdp__(z,v),z(:,ii),v(ii),dt);
    
    clc;
    disp(ii/length(t));
end

figure;
subplot(211);
hold on;
plot(t,x);
plot(t,z);
hold off;
subplot(212);
hold on;
plot(t(1:end-1),u);
plot(t(1:end-1),v);
hold off;


function dx = f(g,x,u,dt)
f1 = dt*g(x,u);
f2 = dt*g(x+0.5*f1,u);
f3 = dt*g(x+0.5*f2,u);
f4 = dt*g(x+f3,u);

dx = 1/6*(f1+2*f2+2*f3+f4);
end

function xd = vdp_(x,u)
mu = 1;

xd = [x(2); mu*(1-x(1)^2)*x(2) - x(1) + u];

end

function xd = vdp__(x,u)
mu = 1;

xd = [x(2); mu*(1-x(1)^2)*x(2) - x(1) + u] + [0; 10*sin(x(1))];

end

function out = J(x0,K,dt)
n = 50;
x = zeros(2,n);
x(:,1) = x0;
u = zeros(1,n-1);
for ii = 1:n-1
    
    u(ii) = -K(:,ii)'*x(:,ii);
    
    x(:,ii+1) = x(:,ii) + f(@(x,u)vdp_(x,u),x(:,ii),u(ii),dt);
    
end
 out = sum(sum(x.^2)) + 0.01*sum(u.^2);
end

function out = L(x0,u,dt)
n = 50;
x = zeros(2,n);
x(:,1) = x0;

for ii = 1:n-1
    
    x(:,ii+1) = x(:,ii) + f(@(x,u)vdp_(x,u),x(:,ii),u(ii),dt);
    
end
 out = sum(sum(x.^2)) + 0.01*sum(u.^2);
end