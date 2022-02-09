%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Function: Test on CIFAR-10 dataset. %%%
%%% Author: TSF                         %%%
%%% Time: 2022.01.10                    %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear;
clc;


%% Parameter preparation (Federated learning: Multi-LR)
K = 10;  % Number of edge devices
para = load('processeddataset.mat');
% para1 = load('parameter.mat');
train_x_loc = para.train_x_loc;
train_y_loc = para.train_y_loc;
test_x = para.test_x;
test_y = para.test_y;
train_x = [];
train_y = [];
for j = 1:K
    train_x = [train_x;train_x_loc(:,:,j)];
    train_y = [train_y;train_y_loc(:,:,j)];
end
batchsize = 5000;  % Learning batch size
learningRate = 0.01;
[m,n] = size(train_x);  % Date volume
Dloc = m/K;
d = (n+1)*10;
theta_ori = zeros(n+1,10);  % n is model dimension, 10 is the classes.
grad_loc = zeros(n+1,10,K);  % Local gradient for each edge device.
numepochs = 60;
eblock = 5;

epsilon = 20;     % DP guidelines
delta = 0.03;
N = 30;
u = 0.3;
L = 2.5;
SNR_d = 3;
SNR = SNR_d;
SNR = 10^(SNR/10);
N0 = 1;
P=SNR*d/eblock;
syms x               % DP guidelines
f(x) = sqrt(pi)*x*exp(x^2);
tmp = vpasolve(f(x) == 1/delta);
Rdp = power(sqrt(epsilon+power(double(tmp),2))-double(tmp),2);
Gk = 8*ones(1,K);     % Parameter Gk
gamma = 112;
ak = zeros(1,K);
for j = 1:K
    ak(j) = power(Dloc*Gk(j),2)/P;
end


%% Channel preparation
kappa_IB = 5;  % IRS to BS
kappa_DI = 0;  % Decives to IRS
kappa_DB = 0;  % Devices to BS
r_mat_DB = normrnd(0,0.5,1,K,eblock)+1j*normrnd(0,0.5,1,K,eblock);
r_mat_DI = normrnd(0,0.5,N,K,eblock)+1j*normrnd(0,0.5,N,K,eblock);
r_mat_IB = normrnd(0,0.5,N,1,eblock)+1j*normrnd(0,0.5,N,1,eblock);
G = sqrt(kappa_IB/(1+kappa_IB))+sqrt(1/(1+kappa_IB))*r_mat_IB;
hr = sqrt(kappa_DI/(1+kappa_DI))+sqrt(1/(1+kappa_DI))*r_mat_DI;
hd = sqrt(kappa_DB/(1+kappa_DB))+sqrt(1/(1+kappa_DB))*r_mat_DB;
Random_max = 10000;
g_mat = zeros(1,K,eblock);     % Without IRS
for j = 1:K
    for e = 1:eblock
        g_mat(:,j,e) = roundn(hd(:,j,e),-8);
    end
end
find_v = zeros(N+1,Random_max,eblock);
R = zeros(N+1,N+1,K,eblock);
for e = 1:eblock
    for j = 1:K
        R(1:N,1:N,j,e) = roundn((diag(hr(:,j,e))')*G(:,:,e)*(G(:,:,e)')*diag(hr(:,j,e)),-8);
        R(N+1,1:N,j,e) = roundn((hd(:,j,e)')*(G(:,:,e)')*diag(hr(:,j,e)),-8);
        R(1:N,N+1,j,e) = roundn(hd(:,j,e)*(diag(hr(:,j,e))')*G(:,:,e),-8);
    end
end
noise_noma = 0.001*normrnd(0,N0,numepochs,d); % para1.noise_noma;


%% Naive mean with SignSGD (No attack)
zeta_tem = zeros(1,numepochs);
test_accuracy = zeros(1,numepochs);

% Learning process
theta = theta_ori;
tem_noma = zeros(1,K,eblock);
temp2 = zeros(1,eblock);
for e = 1:eblock    % The channel won't change, so we can select condition firstly
    for t = 1:K
        tem_noma(:,t,e) = abs(g_mat(:,t,e))/(Gk(:,t)*Dloc);
    end
    temp2(e) = min(tem_noma(:,:,e));
end
temp1 = max(temp2);  % Maximum in one learning round
temm = 4*SNR*power(gamma*temp1,2)*d/eblock;


% Design of c_vector
c_vector = zeros(1,eblock,numepochs);
if numepochs*temm < Rdp  % We first justify condition (a)
    for i = 1:numepochs
        for e = 1:eblock
            c_vector(1,e,i) = P*temp2(e)^2;
        end
    end
else  % The others are (b) and (c)   
    % Enum searching
    block_round = zeros(eblock+1,numepochs);  % Compute the number of block with iota=0

    % We say iota~=0, then we have iota=(1-u/L)^(-t)
    % We say iota=0, then we have psi~=0, i.e., meet full power

    % Firstly, we set every tau=1 for initialization
    for k = 1:numepochs
        val = 0;
        for j = 1:k
            val = val + 2*gamma*(1-u/L)^(-j/2);
        end    
        zeta_sq = val/(Rdp-temm*(numepochs-k));
        if zeta_sq < 0
            continue
        elseif 2*gamma*(1-u/L)^(-k/2)/zeta_sq > temm
            continue
        else
            zeta_tem(k) = zeta_sq;  % This value is tau^0.5
        end 
    end
    zeta_tem((zeta_tem==0))=NaN;
    zeta = min(zeta_tem);

    % Initialize block_round
    for i=1:numepochs
        for e=1:eblock
            tmp1 = 0.5*N0*(1-u/L)^(-i/2)/gamma/zeta;
            tmp2 = P*temp2(e)^2;
            if tmp1 < tmp2  % Privacy is dominent
                block_round(e,i) = 1;
            end
        end
        if block_round(:,i) == 0
            block_round(eblock+1,i) = 1;
        end
    end

    % Find the best parameter
    obj_fin = [1e60,1e50];
    while obj_fin(1)-obj_fin(2) ~= 0
        zeta_tem = zeros(1,numepochs);
        for k = 1:numepochs
            val = 0;
            for j = 1:k
                val = val + 2*sqrt(sum(block_round(:,j)))*gamma*(1-u/L)^(-j/2); 
            end    
            zeta_sq = val/(Rdp-temm*(numepochs-k));
            if zeta_sq < 0
                continue
            elseif 2*sqrt(sum(block_round(:,k)))*gamma*(1-u/L)^(-k/2)/zeta_sq > temm 
                continue
            else
                zeta_tem(k) = zeta_sq;  % This value is beta^0.5
            end 
        end
        zeta_tem((zeta_tem==0))=NaN;
        zeta = min(zeta_tem);

        for i=1:numepochs
            if block_round(:,i) == 0
                block_round(eblock+1,i) = 1;
            end
            tmp1 = 0.5*sqrt(sum(block_round(:,i)))*N0*(1-u/L)^(-i/2)/gamma/zeta;
            for e=1:eblock
                tmp2 = P*temp2(e)^2;
                if tmp1 == 0
                    disp(tmp1)
                end
                c_vector(1,e,i) = min([tmp1,tmp2]);
                if tmp1 < tmp2  % Privacy is dominent
                    block_round(e,i) = 1;
                else
                    block_round(e,i) = 0;
                end
            end
            if block_round(:,i) == 0
                block_round(eblock+1,i) = 1;
            else
                block_round(eblock+1,i) = 0;
            end
        end

        obj_fin(1) = obj_fin(2);
        obj_fin(2) = 0;
        for i=1:numepochs
            tmm = 0;
            for e=1:eblock
                tmm=tmm+1/c_vector(1,e,i);
            end
            obj_fin(2)=obj_fin(2)+(1-u/L)^(-i)*tmm;
        end
    end
end

for i = 1:numepochs  % Learning round

    for ee = 1:eblock
        c = sqrt(c_vector(1,ee,i));
        noise_tem = noise_noma(i,1+(ee-1)*d/eblock : ee*d/eblock);
        noise6(1+(ee-1)*d/eblock : ee*d/eblock) = (1/c)*noise_tem;
    end
    noise6 = reshape(noise6,n+1,10);

    for j = 1:K        % Local update
        kk = randperm(Dloc);
        batch_x = train_x_loc(kk(1 : batchsize), :, j);
        batch_y = train_y_loc(kk(1 : batchsize), :, j);
        [~,grad] = lrCostFunction(theta, batch_x, batch_y, 0.1);
        grad_loc(:,:,j) = grad;
    end
    gradtot = (1/K)*sum(grad_loc,3)+noise6*(1/K);
    theta = theta-learningRate*gradtot;

    % Test set accuracy
    pred_t = predict(theta,test_x);
    test_accuracy(1,i) = mean(double(pred_t==test_y));
end


%% NOMA condition with IRS continuous phase (Adaptive PA)
zeta_tem = zeros(1,numepochs);
noise6 = zeros(1,d);
accuracy_cIRS = zeros(1,numepochs);
g_mat_cIRS = zeros(1,K,eblock);
threshould = 1e-8;
theta_tem = zeros(N,N,eblock);
theta = theta_ori;
for e = 1:eblock
    theta_tem(:,:,e) = diag(exp(1j*unifrnd(0,2*pi,1,N)));
end

temp_value = [1e10,1e8];

% We will use CVX to compute θ given c and σ
while temp_value(1)-temp_value(2) > threshould

    % Firstly, we compute c and σ given θ
    for e = 1:eblock
        for j = 1:K
            g_mat_cIRS(:,j,e) = (G(:,:,e)')*theta_tem(:,:,e)*hr(:,j,e)+hd(:,j,e);
        end
    end

    tem_noma = zeros(1,K,eblock);
    temp2 = zeros(1,eblock);
    for e = 1:eblock    % The channel won't change, so we can select condition firstly
        for t = 1:K
            tem_noma(:,t,e) = abs(g_mat_cIRS(:,t,e))/(Gk(:,t)*Dloc);
        end
        temp2(e) = min(tem_noma(:,:,e));
    end
    temp1 = max(temp2);  % Maximum in one learning round
    temm = 4*SNR*power(gamma*temp1,2)*d/eblock;


    % Design of c_vector
    c_vector = zeros(1,eblock,numepochs);
    if numepochs*temm < Rdp  % We first justify condition (a)
        for i = 1:numepochs
            for e = 1:eblock
                c_vector(1,e,i) = P*temp2(e)^2;
            end
        end
    else  % The others are (b) and (c)   
        % Enum searching
        block_round = zeros(eblock+1,numepochs);  % Compute the number of block with iota=0

        % We say iota~=0, then we have iota=(1-u/L)^(-t)
        % We say iota=0, then we have psi~=0, i.e., meet full power

        % Firstly, we set every tau=1 for initialization
        for k = 1:numepochs
            val = 0;
            for j = 1:k
                val = val + 2*gamma*(1-u/L)^(-j/2);
            end    
            zeta_sq = val/(Rdp-temm*(numepochs-k));
            if zeta_sq < 0
                continue
            elseif 2*gamma*(1-u/L)^(-k/2)/zeta_sq > temm
                continue
            else
                zeta_tem(k) = zeta_sq;  % This value is tau^0.5
            end 
        end
        zeta_tem((zeta_tem==0))=NaN;
        zeta = min(zeta_tem);

        % Initialize block_round
        for i=1:numepochs
            for e=1:eblock
                tmp1 = 0.5*N0*(1-u/L)^(-i/2)/gamma/zeta;
                tmp2 = P*temp2(e)^2;
                if tmp1 < tmp2  % Privacy is dominent
                    block_round(e,i) = 1;
                end
            end
            if block_round(:,i) == 0
                block_round(eblock+1,i) = 1;
            end
        end

        % Find the best parameter
        obj_fin = [1e60,1e50];
        while obj_fin(1)-obj_fin(2) ~= 0
            zeta_tem = zeros(1,numepochs);
            for k = 1:numepochs
                val = 0;
                for j = 1:k
                    val = val + 2*sqrt(sum(block_round(:,j)))*gamma*(1-u/L)^(-j/2); 
                end    
                zeta_sq = val/(Rdp-temm*(numepochs-k));
                if zeta_sq < 0
                    continue
                elseif 2*sqrt(sum(block_round(:,k)))*gamma*(1-u/L)^(-k/2)/zeta_sq > temm 
                    continue
                else
                    zeta_tem(k) = zeta_sq;  % This value is beta^0.5
                end 
            end
            zeta_tem((zeta_tem==0))=NaN;
            zeta = min(zeta_tem);

            for i=1:numepochs
                if block_round(:,i) == 0
                    block_round(eblock+1,i) = 1;
                end
                tmp1 = 0.5*sqrt(sum(block_round(:,i)))*N0*(1-u/L)^(-i/2)/gamma/zeta;
                for e=1:eblock
                    tmp2 = P*temp2(e)^2;
                    if tmp1 == 0
                        disp(tmp1)
                    end
                    c_vector(1,e,i) = min([tmp1,tmp2]);
                    if tmp1 < tmp2  % Privacy is dominent
                        block_round(e,i) = 1;
                    else
                        block_round(e,i) = 0;
                    end
                end
                if block_round(:,i) == 0
                    block_round(eblock+1,i) = 1;
                else
                    block_round(eblock+1,i) = 0;
                end
            end

            obj_fin(1) = obj_fin(2);
            obj_fin(2) = 0;
            for i=1:numepochs
                tmm = 0;
                for e=1:eblock
                    tmm=tmm+1/c_vector(1,e,i);
                end
                obj_fin(2)=obj_fin(2)+(1-u/L)^(-i);
            end
        end
    end


    % Secondly, we will optimaize θ given c and σ
    % We think that the θ will not change in each iteration
    ak_all = zeros(K,eblock,numepochs);
    for t = 1:numepochs
        for e = 1:eblock
            for j = 1:K
                ak_all(j,e,t) = c_vector(1,e,t)^2*ak(j)-abs(hd(:,j,e))^2;
            end
        end
    end

    % We use CVX to obtain a sequence of theta
    cvx_begin quiet
        variable VV(N+1,N+1,eblock) complex semidefinite;
        minimize(0);
        subject to
            for t = 1:numepochs
                for j = 1:K
                    for e = 1:eblock
                        trace(R(:,:,j,e)*VV(:,:,e))>=ak_all(j,e,t);
                    end
                end
            end
            for e = 1:eblock
                for nn = 1:N
                    VV(nn,nn,e) == 1;
                end
                VV(:,:,e) == hermitian_semidefinite(N+1);
            end
    cvx_end   

    % SDR Gaussian randomiz method
    for e = 1:eblock
        for l = 1:Random_max
            find_v(:,l,e) = mvnrnd(zeros(N+1,1),VV(:,:,e)).';
        end
        find_feasible = zeros(1,Random_max);
        h_ctemp = zeros(1,K);
        h_cmax = 0;
        for l = 1:Random_max
            v_res = find_v(1:N,l,e)/find_v(N+1,l,e);
            for i = 1:N
                v_res(i) = v_res(i)/abs(v_res(i));
            end
            theta_c = diag(v_res);
            for j = 1:K
                h_ctemp(j) = abs((G(:,:,e)')*theta_c*hr(:,j,e)+hd(:,j,e))/Gk(:,j);
            end
            tepc1 = min(h_ctemp);
            if h_cmax < tepc1
                h_cmax = tepc1;
                theta_tem(:,:,e) = theta_c;
            end
        end
    end
    temp_value(1) = temp_value(2);
    temp_value(2) = 0;
    for t = 1:numepochs
        tmm = 0;
        for e = 1:eblock
            tmm = tmm + 1/c_vector(1,e,t);
        end
        temp_value(2) = temp_value(2)+tmm;
    end
end

theta_opt = theta_tem;

for e = 1:eblock
    for j = 1:K
        g_mat_cIRS(:,j,e) = (G(:,:,e)')*theta_tem(:,:,e)*hr(:,j,e)+hd(:,j,e);
    end
end

tem_noma = zeros(1,K,eblock);
temp2 = zeros(1,eblock);
for e = 1:eblock    % The channel won't change, so we can select condition firstly
    for t = 1:K
        tem_noma(:,t,e) = abs(g_mat_cIRS(:,t,e))/(Gk(:,t)*Dloc);
    end
    temp2(e) = min(tem_noma(:,:,e));
end
temp1 = max(temp2);  % Maximum in one learning round
temm = 4*SNR*power(gamma*temp1,2)*d/eblock;


% Design of c_vector
c_vector = zeros(1,eblock,numepochs);
if numepochs*temm < Rdp  % We first justify condition (a)
    for i = 1:numepochs
        for e = 1:eblock
            c_vector(1,e,i) = P*temp2(e)^2;
        end
    end
else  % The others are (b) and (c)   
    % Enum searching
    block_round = zeros(eblock+1,numepochs);  % Compute the number of block with iota=0

    % We say iota~=0, then we have iota=(1-u/L)^(-t)
    % We say iota=0, then we have psi~=0, i.e., meet full power

    % Firstly, we set every tau=1 for initialization
    for k = 1:numepochs
        val = 0;
        for j = 1:k
            val = val + 2*gamma*(1-u/L)^(-j/2);
        end    
        zeta_sq = val/(Rdp-temm*(numepochs-k));
        if zeta_sq < 0
            continue
        elseif 2*gamma*(1-u/L)^(-k/2)/zeta_sq > temm
            continue
        else
            zeta_tem(k) = zeta_sq;  % This value is tau^0.5
        end 
    end
    zeta_tem((zeta_tem==0))=NaN;
    zeta = min(zeta_tem);

    % Initialize block_round
    for i=1:numepochs
        for e=1:eblock
            tmp1 = 0.5*N0*(1-u/L)^(-i/2)/gamma/zeta;
            tmp2 = P*temp2(e)^2;
            if tmp1 < tmp2  % Privacy is dominent
                block_round(e,i) = 1;
            end
        end
        if block_round(:,i) == 0
            block_round(eblock+1,i) = 1;
        end
    end

    % Find the best parameter
    obj_fin = [1e60,1e50];
    while obj_fin(1)-obj_fin(2) ~= 0
        zeta_tem = zeros(1,numepochs);
        for k = 1:numepochs
            val = 0;
            for j = 1:k
                val = val + 2*sqrt(sum(block_round(:,j)))*gamma*(1-u/L)^(-j/2); 
            end    
            zeta_sq = val/(Rdp-temm*(numepochs-k));
            if zeta_sq < 0
                continue
            elseif 2*sqrt(sum(block_round(:,k)))*gamma*(1-u/L)^(-k/2)/zeta_sq > temm 
                continue
            else
                zeta_tem(k) = zeta_sq;  % This value is beta^0.5
            end 
        end
        zeta_tem((zeta_tem==0))=NaN;
        zeta = min(zeta_tem);

        for i=1:numepochs
            if block_round(:,i) == 0
                block_round(eblock+1,i) = 1;
            end
            tmp1 = 0.5*sqrt(sum(block_round(:,i)))*N0*(1-u/L)^(-i/2)/gamma/zeta;
            for e=1:eblock
                tmp2 = P*temp2(e)^2;
                if tmp1 == 0
                    disp(tmp1)
                end
                c_vector(1,e,i) = min([tmp1,tmp2]);
                if tmp1 < tmp2  % Privacy is dominent
                    block_round(e,i) = 1;
                else
                    block_round(e,i) = 0;
                end
            end
            if block_round(:,i) == 0
                block_round(eblock+1,i) = 1;
            else
                block_round(eblock+1,i) = 0;
            end
        end

        obj_fin(1) = obj_fin(2);
        obj_fin(2) = 0;
        for i=1:numepochs
            tmm = 0;
            for e=1:eblock
                tmm=tmm+1/c_vector(1,e,i);
            end
            obj_fin(2)=obj_fin(2)+(1-u/L)^(-i);
        end
    end
end

% Get the vector c 
for i = 1:numepochs  % Learning round

    for ee = 1:eblock
        c = sqrt(c_vector(1,ee,i));
        noise_tem = noise_noma(i,1+(ee-1)*d/eblock : ee*d/eblock);
        noise6(1+(ee-1)*d/eblock : ee*d/eblock) = (1/c)*noise_tem;
    end
    noise6 = reshape(noise6,n+1,10);

    for j = 1:K        % Local update
        kk = randperm(Dloc);
        batch_x = train_x_loc(kk(1 : batchsize), :, j);
        batch_y = train_y_loc(kk(1 : batchsize), :, j);
        [~,grad] = lrCostFunction(theta, batch_x, batch_y, 0.1);
        grad_loc(:,:,j) = grad;
    end
    gradtot = (1/K)*sum(grad_loc,3)+noise6*(1/K);
    theta = theta-learningRate*gradtot;

    % Test set accuracy
    pred_t = predict(theta,test_x);
    accuracy_cIRS(1,i) = mean(double(pred_t==test_y));
end


%% NOMA condition with IRS
zeta_tem = zeros(1,100);
noise6_IRS = zeros(1,d);
accuracy_IRS = zeros(1,numepochs);
g_mat_IRS = zeros(1,K,eblock);    % With IRS
theta = theta_ori;
theta_result = theta_opt;

for e = 1:eblock
    for j = 1:K
        g_mat_IRS(1,j,e) = roundn((G(:,:,e)')*theta_result(:,:,e)*hr(:,j,e)+hd(:,j,e),-8);
    end
end

tem_noma = zeros(1,K,eblock);
temp2 = zeros(1,eblock);
for e = 1:eblock    % The channel won't change, so we can select condition firstly
    for t = 1:K
        tem_noma(:,t,e) = abs(g_mat_IRS(:,t,e))/(Gk(:,t)*Dloc);
    end
    temp2(e) = min(tem_noma(:,:,e));
end

% Design of c_vector
c_vector = zeros(1,eblock,numepochs);
for i = 1:numepochs
    for e = 1:eblock
        c_vector(1,e,i) = P*temp2(e)^2;
    end
end

% Get the vector c 
for i = 1:numepochs  % Learning round

    for ee = 1:eblock
        c = sqrt(c_vector(1,ee,i));
        noise_tem = noise_noma(i,1+(ee-1)*d/eblock : ee*d/eblock);
        noise6(1+(ee-1)*d/eblock : ee*d/eblock) = (1/c)*noise_tem;
    end
    noise6 = reshape(noise6,n+1,10);

    for j = 1:K        % Local update
        kk = randperm(Dloc);
        batch_x = train_x_loc(kk(1 : batchsize), :, j);
        batch_y = train_y_loc(kk(1 : batchsize), :, j);
        [~,grad] = lrCostFunction(theta, batch_x, batch_y, 0.1);
        grad_loc(:,:,j) = grad;
    end
    gradtot = (1/K)*sum(grad_loc,3)+noise6*(1/K);
    theta = theta-learningRate*gradtot;

    % Test set accuracy
    pred_t = predict(theta,test_x);
    accuracy_IRS(1,i) = mean(double(pred_t==test_y));
end


%% Analysis
plot(test_accuracy);
hold on
plot(accuracy_cIRS);
hold on
plot(accuracy_IRS);
hold on 
legend('benchmark','RIS with privacy','RIS without privacy')