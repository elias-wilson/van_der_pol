clc;
clear;
close all;

%% Inputs

% Simulation Time
dt = 0.01;
t = 0:dt:10;

% System Parameters
k = 1;
c = 1;
m = 1;

% Initial Conditions
x0 = [1 0]'; % States
xe0 = [1 0 0 0]'; % Estimates

% Output Matrix
C = [1 0];

% Kalman Parameters
r = 0.01^2; % Measurment Noise
q = 0.001; % Proccess Noise
W = [0 0 1 1]'; % Proccess Noise States
P = [1000 0 0 0; 0 1000 0 0; 0 0 1000 0; 0 0 0 1000]; % Inital Covariance
H = [1 0 0 0]; % Measurement

%% Simulation

n = length(t); % Number of Time Steps
z = length(C(:,1)); % Number of Outputs

% Allocate Memory
x = zeros(2,n);
xe = zeros(4,n);
y = zeros(z,n);
xecov = zeros(4,n);

% Fill Initial Conditions
x(:,1) = x0;
xe(:,1) = xe0;
xecov(:,1) = 3*sqrt(diag(P));

for ii = 1:n-1
    
    % System Integration
    f1 = dt*vdp_(x(:,ii),k,c,m);
    f2 = dt*vdp_(x(:,ii)+0.5*f1,k,c,m);
    f3 = dt*vdp_(x(:,ii)+0.5*f2,k,c,m);
    f4 = dt*vdp_(x(:,ii)+f3,k,c,m);
    
    x(:,ii+1) = x(:,ii) + 1/6*(f1+2*f2+2*f3+f4) + [0 1]'*sqrt(q)*randn*dt;
    y(1:z,ii) = C*x(:,ii) + sqrt(r)*randn*ones(z,1);
    
    % Kalman Filter
    K = P*H'*(H*P*H' + r*eye(z))^-1;
    P = (eye(4) - K*H)*P;
    xe(:,ii) = xe(:,ii) + K*(y(:,ii)-H*xe(:,ii));
   
    % Estimation Propagation
    f1 = dt*vdp_(xe(1:2,ii),xe(4,ii),xe(3,ii),m);
    f2 = dt*vdp_(xe(1:2,ii)+0.5*f1,xe(4,ii),xe(3,ii),m);
    f3 = dt*vdp_(xe(1:2,ii)+0.5*f2,xe(4,ii),xe(3,ii),m);
    f4 = dt*vdp_(xe(1:2,ii)+f3,xe(4,ii),xe(3,ii),m);
    
    xe(:,ii+1) = [xe(1:2,ii)+1/6*(f1+2*f2+2*f3+f4);xe(3,ii);xe(4,ii)];
    A = [0 1 0 0; -4*xe(3,ii)/m*xe(1,ii)*xe(2,ii)-xe(4,ii)/m -2*xe(3,ii)/m*(xe(1,ii)^2-1) -2/m*(x(1,ii)^2-1)*x(2,ii) -xe(1,ii)/m; 0 0 0 0; 0 0 0 0]; % Jacobian
    phi = c2d(A,W,dt);
    P = phi*P*phi' + W*q*W'*dt;
    xecov(:,ii+1) = 3*sqrt(diag(P));
end


%% Plotting
figure(1);
set(gcf,'Color','w');
subplot(221);
plot(t,x(1,:),t,xe(1,:),'Linewidth',1.5);
xlabel('Time (s)');
ylabel('Position');
ylim([-5 5]);
grid minor;
subplot(222);
plot(t,x(2,:),t,xe(2,:),'Linewidth',1.5);
xlabel('Time (s)');
ylabel('Velocity');
ylim([-4 4]);
grid minor;
subplot(223);
plot(t,c*ones(1,n),t,xe(3,:),'Linewidth',1.5);
xlabel('Time (s)');
ylabel('Damping');
ylim([-4 4]);
grid minor;
subplot(224);
plot(t,k*ones(1,n),t,xe(4,:),'Linewidth',1.5);
xlabel('Time (s)');
ylabel('Spring');
ylim([-4 4]);
grid minor;

figure(2);
set(gcf,'Color','w');
subplot(221);
plot(t,xe(1,:)-x(1,:),t,xecov(1,:),'r--',t,-xecov(1,:),'r--','Linewidth',1.5);
xlabel('Time (s)');
ylabel('Position Residual');
ylim([-xecov(1,end)-0.5*xecov(1,end) xecov(1,end)+0.5*xecov(1,end)]);
grid minor;
subplot(222);
plot(t,xe(2,:)-x(2,:),t,xecov(2,:),'r--',t,-xecov(2,:),'r--','Linewidth',1.5);
xlabel('Time (s)');
ylabel('Velocity Residual');
ylim([-xecov(2,end)-0.5*xecov(2,end) xecov(2,end)+0.5*xecov(2,end)]);
grid minor;
subplot(223);
plot(t,xe(3,:)-c,t,xecov(3,:),'r--',t,-xecov(3,:),'r--','Linewidth',1.5);
xlabel('Time (s)');
ylabel('Damping Residual');
ylim([-xecov(3,end)-0.5*xecov(3,end) xecov(3,end)+0.5*xecov(3,end)]);
grid minor;
subplot(224);
plot(t,xe(4,:)-k,t,xecov(4,:),'r--',t,-xecov(4,:),'r--','Linewidth',1.5);
xlabel('Time (s)');
ylabel('Damping Residual');
ylim([-xecov(4,end)-0.5*xecov(4,end) xecov(4,end)+0.5*xecov(4,end)]);
grid minor;


figure(3);
set(gcf,'Color','w');
plot(t,y,'Linewidth',1.5);
xlabel('Time (s)');
ylabel('Measurement');
grid minor;
