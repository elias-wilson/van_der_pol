clc;
clear;
close all;

%% Inputs

% Simulation Time
dt = 0.01;
t = 0:dt:10;

% System Parameters (Van der Pol)
k = 1;
c = 1;
m = 1;
B = [0; 1];

% Initial Conditions
x0 = [-1 -3]'; % States
xr0 = [0 0]'; % Reference States

% Output Matrix
C = eye(2);

% MPC Parameters
Ns = 100; % Horizon (Steps)
gam = 0.5; % Learning Rate
Q = [10 0; 0 1];
R = 0.0001;

% Control Limits
cont_high = 8*ones(1,Ns);
cont_low = -8*ones(1,Ns);

% Reference Parameters (Damped Oscillator)
kr = 1;
cr = 1;
% ur = ones(1,length(t)+Ns);
% ur = 3*sin(pi*[t,t(end):dt:t(end)+dt*Ns]+pi/4);
ur = 3*sin(pi*[t,t(end):dt:t(end)+dt*Ns]+pi/4) + 2*sin(2*pi*[t,t(end):dt:t(end)+dt*Ns]);

%% Simulation Setup
n = length(t); % Number of Time Steps
z = length(C(:,1)); % Number of Outputs

% Create Reference System
Ar = [0 1; -kr -cr];
Br = [0; 1];
Cr = eye(2);

% Allocate Memory
x = zeros(2,n);
y = zeros(z,n);
u = zeros(1,n-1);
xr = zeros(2,n+Ns);
yr = zeros(2,n+Ns);
u_ = zeros(1,Ns);
x_ = zeros(2,Ns);

% Fill Initial Conditions
x(:,1) = x0;
xr(:,1) = xr0;
y(:,1) = C*x0;
yr(:,1) = Cr*xr(:,1);

%% Reference Trajectory

for ii = 1:n+Ns
    f1 = dt*dyn(xr(:,ii),ur(ii),Ar,Br);
    f2 = dt*dyn(xr(:,ii)+0.5*f1,ur(ii),Ar,Br);
    f3 = dt*dyn(xr(:,ii)+0.5*f2,ur(ii),Ar,Br);
    f4 = dt*dyn(xr(:,ii)+f3,ur(ii),Ar,Br);
    
    xr(:,ii+1) = xr(:,ii) + 1/6*(f1+2*f2+2*f3+f4);
    yr(:,ii+1) = Cr*xr(:,ii+1);
end

%% Simulation

for ii = 1:n-1
    
    %%% Model Predictive Controller  
    
    ij = 0;
    del = 1;
    L = 10000;
    while del > 0.001
        ij = ij+1;
        L0 = L;
        L = 0;
        A = zeros(2,2,Ns);
        
        % Initiate Cost Dervivative with Initial Control Guess
        dL_dU = R*u_;
        
        % Start Model Prediciton at Current State
        x_(:,1) = x(:,ii);
        
        % Get Part of Reference
        xr_ = xr(:,ii:ii+Ns-1);
        
        % Linearize the Initial Condition
        A(:,:,1) = Jf(x_(:,1),k,c,m);
        
        for jj = 1:Ns-1
            % Integrate the Prediciton
            f1 = dt*vdp_c(x_(:,jj),k,c,m,u_(jj));
            f2 = dt*vdp_c(x_(:,jj)+0.5*f1,k,c,m,u_(jj));
            f3 = dt*vdp_c(x_(:,jj)+0.5*f2,k,c,m,u_(jj));
            f4 = dt*vdp_c(x_(:,jj)+f3,k,c,m,u_(jj));
            
            x_(:,jj+1) = x_(:,jj) + 1/6*(f1+2*f2+2*f3+f4);
            
            % Linearize System at Each Point
            A(:,:,jj+1) = Jf(x_(:,jj+1),k,c,m);
        end
        
        % Cost Derivative
        for jj = 1:Ns
            dL_dx = 0;
            dx_du = zeros(2,Ns);
            for kj = jj:Ns-1
                % Evaluate Derviative of States
                if kj == jj
                    dx_du(:,kj+1) = dt*B;
                else
                    dx_du(:,kj+1) = (eye(2) + dt*A(:,:,kj))*dx_du(:,kj);
                end
                
                % Running Total of Cost State Derivative
                dL_dx = dL_dx + dx_du(:,kj)'*Q*(x_(:,kj)-xr_(:,kj));
            end
            if jj < Ns
                dL_dx = dL_dx + dx_du(:,kj+1)'*Q*(x_(:,kj+1)-xr_(:,kj+1));
            end
            
            % Cost Derivative
            dL_dU(jj) = dL_dU(jj) + dL_dx;
            
            % Cost
            L = L + (x_(:,jj)-xr_(:,jj))'*Q*(x_(:,jj)-xr_(:,jj)) + u_(:,jj)'*R*u_(:,jj);
        end
        
        % Change in Cost
        del = abs(L-L0);
        
        % Neural Network (saturate limits the control inputs)
        u_shift = u_ - cont_low;
        u_shift = u_shift + gam*(-u_shift + saturate(u_shift - dL_dU,2*cont_high,zeros(1,Ns)));
        u_ = u_shift + cont_low;
        
        clc;
        disp(ii/length(t)); % Progress
        disp(ij); % Iteration
        disp(del); % Convergence
        disp(L); % Cost
    end
    
    % Implement First Control Input
    u(ii) = u_(1);
    
    % Set Next Initial Guess
    u_ = [u_(2:end),u_(end)];
    
    %%% Plant Integration
    f1 = dt*vdp_c(x(:,ii),k,c,m,u(ii));
    f2 = dt*vdp_c(x(:,ii)+0.5*f1,k,c,m,u(ii));
    f3 = dt*vdp_c(x(:,ii)+0.5*f2,k,c,m,u(ii));
    f4 = dt*vdp_c(x(:,ii)+f3,k,c,m,u(ii));
    
    x(:,ii+1) = x(:,ii) + 1/6*(f1+2*f2+2*f3+f4);
    y(:,ii+1) = C*x(:,ii+1);
    
end


%% Plotting
figure(1);
set(gcf,'Color','w');
subplot(311);
plot(t,x(1,:),t,xr(1,(1:n)),'Linewidth',1.5);
xlabel('Time (s)');
ylabel('Position');
ylim([-5 5]);
grid minor;
legend('Plant','Reference');
subplot(312);
plot(t,x(2,:),t,xr(2,(1:n)),'Linewidth',1.5);
xlabel('Time (s)');
ylabel('Velocity');
ylim([-4 4]);
grid minor;
legend('Plant','Reference');
subplot(313);
plot(t(1:end-1),u,'Linewidth',1.5);
xlabel('Time (s)');
ylabel('Control');
ylim([-cont_low(1) cont_high(1)]);
grid minor;

