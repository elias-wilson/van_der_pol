%%%%%
% Currently Developing
%%%%%

clc;
clear;
close all;

%% Setup

dt = 0.01;
t = 0:dt:10;

mu = 1;

Q = [1 0; 0 1];
R = 0.005;

x0 = [0; 1]';

x = zeros(2,length(t));
x(:,1) = x0;
u = zeros(1,length(t)-1);

%% Simulation

for ii = 1:length(t)-1
%     clc;
%     disp(ii);
%     disp(length(t));
    
    del = 1;
    V = 0;
    kk = 0;
    while del > 0.001
        kk = kk + 1;
        xp = int(x(:,ii),u(ii),mu,dt);
        V0 = V;
        V = cst(xp,0,Q,R) + cst(x(:,ii),u(ii),Q,R);
        del = abs(V-V0);
        u(ii) = 0.5*R\[0 1]*fx(xp,mu)*x(:,ii);
        
        clc;
        disp(kk);
        disp(del);
    end
    
    x(:,ii+1) = int(x(:,ii),u(:,ii),mu,dt);
end


%% Plotting
figure;
subplot(211);
plot(t,x);
subplot(212);
plot(t(1:end-1),u);

%% Functions

function x = int(x,u,mu,dt)
% RK4
f1 = dt*vp(x,mu,u);
f2 = dt*vp(x+0.5*f1,mu,u);
f3 = dt*vp(x+0.5*f2,mu,u);
f4 = dt*vp(x+f3,mu,u);

x = x + 1/6*(f1+2*f2+2*f3+f4);
end

function xd = vp(x,mu,u)
% Van der Pol Dynamics
xd = [x(2); -mu*(x(1)^2-1)*x(2)-x(1)] + [0; 1]*u;
end

function A = fx(x,mu)
% Jacobian
A = [0 1; -2*mu*x(2)*x(1)-1 -mu*(x(1)^2-1)];
end

function J = cst(x,u,Q,R)
% Cost Function
J = x'*Q*x + R*u^2;
end

function id = ind(v,x)
id = find(max(0,v-x),1);
end