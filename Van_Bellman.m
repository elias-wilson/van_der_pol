%%%%%
% Currently Developing
%%%%%

clc;
clear;
close all;

%% Setup

dt = 0.01;
t = 0:dt:100;

mu = 1;

x0 = [0; 1]';

x = zeros(2,length(t));
x(:,1) = x0;
u = zeros(1,length(t)-1);

%% Dynamic Programming
xlim = 4;
co = linspace(-8,8,20);
st = linspace(-xlim,xlim,20);
c = zeros(length(st), length(st), length(co));

for x1 = 1:length(st)
    clc;
    disp(x1);
    for x2 = 1:length(st)
        for uu = 1:length(co)
            xp = int([st(x1); st(x2)],co(uu),mu,dt);
            if any(abs(xp) > xlim)
                c(x1,x2,uu) = inf;
            else
                c(x1,x2,uu) = cst(xp,0) + cst([st(x1); st(x2)],co(uu));
            end
        end
    end
end

[~,us] = min(c,[],3);

%% Simulation

for ii = 1:length(t)-1
    clc;
    disp(ii);
    disp(length(t));
    
    id1 = ind(st,x(1,ii));
    id2 = ind(st,x(2,ii));
    id3 = interp2(1:length(st),1:length(st),us,id2,id1);
    
    u(:,ii) = interp1(1:length(co),co,id3);
    
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

f1 = dt*vp(x,mu,u);
f2 = dt*vp(x+0.5*f1,mu,u);
f3 = dt*vp(x+0.5*f2,mu,u);
f4 = dt*vp(x+f3,mu,u);

x = x + 1/6*(f1+2*f2+2*f3+f4);

% x = x + dt*vp(x,mu,u);
end

function xd = vp(x,mu,u)
xd = [x(2); -mu*(x(1)^2-1)*x(2)-x(1)+u];
end

function J = cst(x,u)
Q = [1 0; 0 1];
R = 0.005;
J = x'*Q*x + R*u^2;
end

function id = ind(v,x)
 id = find(max(0,v-x),1);
end