clc;
clear;
close all;
% delete(findall(0));

dt = 0.01;
t = 0:dt:11.5;

Nb = 30;
Nf = 50;

x0 = [0 1]';

x = zeros(2, length(t));
dx = zeros(2, length(t)-1);
u = zeros(1,length(t)-1);
x(:,1) = x0;
z = x;

% T = randn(12,8);
T = [1.33295101142013,2.48682132029734,-1.56502465785673,-0.233953540330954,-0.0656148757390404,-1.46144181507780,-0.163880256756727,-1.22401897480879;0.935251207390719,-1.66564311788150,-0.0788010280958669,-1.04673406988337,-0.255683033134042,-1.24429309468994,-0.0672404702443348,0.118509213098193;0.663517921831603,-0.415927942768154,-0.941394415849255,1.58331888600714,0.404724197904154,-0.154701822122984,-1.00159245598463,0.959475384996167;-0.350232666735205,-0.0842246064708947,-0.652861547259777,-0.249413421524592,0.351756727708850,-1.61491385765397,-0.555661447134641,-0.340930438594071;1.61987743935497,0.0892519475074891,0.282529239916988,1.29401900835506,0.807511060208285,2.24606514978483,-1.35155634252200,-0.173318330984157;-0.0508386867375274,1.45638981240098,-0.819076950277555,0.361672089874990,-1.14878052664809,1.04526987869494,0.364211322734809,0.579713613873998;-0.812697070766603,0.219453518638082,-0.988974087143185,-0.263052508703654,0.715113278785686,-2.06764536263329,-0.452207604978530,0.863919369587695;-0.438419934696927,-0.114877251919321,-1.51585878878068,1.08214445191656,1.04859729800334,-1.15455061354921,0.783365969886060,-1.41691416697625;0.858606672918231,0.0615105350763639,-2.22851047687315,0.984112268352391,0.857492493944586,-0.107192054400332,1.23723664787976,0.396256272467988;0.195211780411692,0.747063933303562,-0.151521678303304,-0.109185445343789,-0.583636190608236,-1.71287908652886,1.07719818143346,0.725554878380130;0.888862153882707,-0.689425378466651,1.16273157463377,1.29595078548944,0.0846375729475371,-0.125944914097829,-0.0348411236037387,-0.151132402938184;0.0692218210392502,0.450818291961002,-0.182239733973500,1.78129512142970,-0.547192940203608,1.24173052459136,-0.503987333962698,0.892784166409411];
% T = zeros(12,8);

 xr = [sin(t); cos(t)];
 u_ = zeros(1,Nf);
 
 for ii = 1:length(t)-1
     
%     if ii >= Nb
%         n = ii-Nb+1:ii;
%         T = fmincon(@(T)cst(xd(:,n),x(:,n),u(n),T,dt),T);
%     end
%      
%     if ii >= Nb
%       n = ii:ii+Nf;
%       u_ = fmincon(@(u_)cst2(x(:,ii),xr(:,n),u_,T,dt),u_);
%       clc;
%     end
    
    u(ii) = u_(1);
    
    dx(:,ii) = rk4(@(x,u)vdp_(x,u),x(:,ii),u(ii),dt);
    
    x(:,ii+1) = x(:,ii) + dt*dx(:,ii);
%     z(:,ii+1) = z(:,ii) + rk4(@(z,u)est(z,u,T),z(:,ii),u(ii),dt);
    
    disp(ii/length(t));
end

%     if ii >= Nb
%         n = ii-Nb+1:ii;
%         T = fmincon(@(T)cst(xd(:,n),x(:,n),u(n),T,dt),T);
%     end

layers = [ ...
    sequenceInputLayer(2)
    fullyConnectedLayer(128)
    tanhLayer
    fullyConnectedLayer(128)
    tanhLayer
    fullyConnectedLayer(2)
    regressionLayer];

options = trainingOptions('adam', ...
    'InitialLearnRate',0.001, ...
    'Verbose',false, ...
    'MaxEpochs',1500, ...
    'Plots','training-progress',...
    'Shuffle','on');

xtrain = 1/8*(x-4);
dxtrain = dx - vdp(x(:,1:end-1),u);
% net1 = trainNetwork(xtrain(:,1:end-1),dxtrain(1,:),layers,options);
% net2 = trainNetwork(xtrain(:,1:end-1),dxtrain(2,:),layers,options);
net = trainNetwork(xtrain(:,1:end-1),dxtrain,layers,options);

for ii = 1:length(t)-1
%     z(:,ii+1) = z(:,ii) + predict(net,1/8*(z(:,ii)-4));
    z(:,ii+1) = z(:,ii)  + dt*rk4(@(z,u)vdp(z,u),z(:,ii),u(ii),dt) + dt*predict(net,1/8*(z(:,ii)-4)); %+ dt*rk4(@(z,u)vdp(z,u),z(:,ii),u(ii),dt)
end

figure();
subplot(211);
hold on;
plot(t,x(1,:));
plot(t,z(1,:));
hold off;
subplot(212);
hold on;
plot(t,x(2,:));
plot(t,z(2,:));
hold off;


%% Functions 

% Van der Pol 
function xd = vdp(x,u)
mu = 1;
xd = [x(2,:); mu*(1-x(1,:).^2).*x(2,:) - x(1,:)] + [0; 1].*u;
end

% Van der Pol with Bonus Nonlinearity
function xd = vdp_(x,u)
mu = 1;
xd = [x(2); mu*(1-x(1)^2)*x(2) - x(1)] + [0; cos(x(1))] + [0; 1]*u;
end

% Runge-Kutta 4
function dx = rk4(f,x,u,dt)
    
f1 = f(x,u);
f2 = f(x+0.5*f1*dt,u);
f3 = f(x+0.5*f2*dt,u);
f4 = f(x+f3*dt,u);

dx = 1/6*(f1 + 2*f2 + 2*f3 + f4);
end

% Rectified Linear Unit
function out = relu(in)
out = max(in,zeros(size(in)));
end

% Sigmoid
function out = sigmoid(in)
a = 10;
out = 1./(1+exp(-a*in));
end

% Estimation System
function xd = est(x,u,T)
xd = vdp(x,u) + nn(x,T);
end

% Neural Network
function xd = nn(x,T)
x = 2/8*(x-4)-1;
h = sigmoid(T(9:10,:)'*x);
h = sigmoid(T(1:8,:)*h);
xd = T(11:12,:)*h;
end

% NN Cost Function
function J = cst(Xd,X,U,T,dt)

n = length(X(1,:));
xd = zeros(2,n);

for ii = 1:n
    xd(:,ii) = rk4(@(x,u)est(x,u,T),X(:,ii),U(ii),dt);
end

J = sum((xd-Xd).^2,'all');
end

% Cost Function
function J = cst2(x0,X,U,T,dt)

n = length(X(1,:));
x = zeros(2,n);
x(:,1) = x0;

for ii = 1:n-1
    x(:,ii+1) = x(:,ii) + rk4(@(x,u)est(x,u,T),x(:,ii),U(ii),dt);
end

J = sum((x-X).^2,'all');
end