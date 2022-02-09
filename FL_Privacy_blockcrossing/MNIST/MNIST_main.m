%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Function: Offline-block via MNIST. %%%
%%% Author: TSF                        %%%
%%% Time: 2021.05.28                   %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear all
clc


%% Note
% This model is for offline condition with different number of learning round.


%% Parameter preparation (Federated learning)
load mnist_uint8;
para = load('parameter.mat');
train_x = double(train_x) / 255;
test_x  = double(test_x) / 255;
train_y = double(train_y);
test_y  = double(test_y);
K = 10;
eblock = 10;
d = 79510;
mu=mean(train_x);    
sigma=max(std(train_x),eps);
train_x=bsxfun(@minus,train_x,mu);  % Noramlized mean value
train_x=bsxfun(@rdivide,train_x,sigma);  % Normalized variance
train_x_loc = zeros(60000/K,784,K);
train_y_loc = zeros(60000/K,10,K);
for j = 1:K
    train_x_loc(:,:,j) = train_x(6000*(j-1)+1 : j*6000,:);
    train_y_loc(:,:,j) = train_y(6000*(j-1)+1 : j*6000,:);
end
test_x=bsxfun(@minus,test_x,mu);
test_x=bsxfun(@rdivide,test_x,sigma);

arc = [784 100 10];  % Input:784, inherent:100, output:10
n=numel(arc);  % The number of layers

W_ori = para.W_ori;  % Weight cell 100*785 and 10*101

learningRate = 1.5;
batchsize = 6000;  
m = size(train_x, 1);  % Date volume
Dloc = m / K;
numbatches = Dloc / batchsize;   % Number of batches
lambda = 0.01;

epsilon = 10;     % DP guidelines
delta = 0.03;
N = 30;
u = 0.3;
L = 2.5;
% P = 1000;
SNR_d = -2;
SNR = SNR_d;
SNR = 10^(SNR/10);
N0 = 1;
P=SNR*d/eblock;
syms x               % DP guidelines
f(x) = sqrt(pi)*x*exp(x^2);
tmp = vpasolve(f(x) == 1/delta);
Rdp = power(sqrt(epsilon+power(double(tmp),2))-double(tmp),2);
Gk = [15,5,7,7,17,2,19,5,11,7];     % Parameter Gk
gamma = 66;
ak = zeros(1,K);
for j = 1:K
    ak(j) = power(Dloc*Gk(j),2)/P;
end


%% Channel preparation
kappa_IB = 5;  % IRS to BS
kappa_DI = 0;  % Decives to IRS
kappa_DB = 0;  % Devices to BS
r_mat_DB = para.r_mat_DB;
r_mat_DI = para.r_mat_DI;%normrnd(0,0.5,N,K,eblock)+1j*normrnd(0,0.5,N,K,eblock);
r_mat_IB = para.r_mat_IB;%normrnd(0,0.5,N,1,eblock)+1j*normrnd(0,0.5,N,1,eblock);
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
noise_noma = normrnd(0,N0,100,d);%para.noise_noma;


%% Training-test process
zeta_tem = zeros(1,100);
accuracy_noma = zeros(1,10);
noise6 = zeros(1,d);
for numepochs = 10:10:70
    W = W_ori;
    
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
                obj_fin(2)=obj_fin(2)+(1-u/L)^(-i);
            end
        end
    end

    for i = 1 : numepochs
        
        % noise
        for ee = 1:eblock
            c = sqrt(c_vector(1,ee,i));
            noise_tem = noise_noma(i,1+(ee-1)*d/eblock : ee*d/eblock);
            noise6(1+(ee-1)*d/eblock : ee*d/eblock) = (1/c)*noise_tem/60000;
        end
        noise_lay1 = noise6(1,1:78500);
        noise_lay2 = noise6(1,78501:79510);
        noise{1,1} = reshape(noise_lay1,100,785);
        noise{1,2} = reshape(noise_lay2,10,101);        
        
        %kk = randperm(Dloc);
        dW_tot{1,1} = zeros(100,785);
        dW_tot{1,2} = zeros(10,101);
        dW{1,1} = zeros(100,785);
        dW{1,2} = zeros(10,101);
        for j = 1:K
            for l = 1 : numbatches
                batch_x = train_x_loc(:,:,j);%(kk((l - 1) * batchsize + 1 : l * batchsize), :, j);
                batch_y = train_y_loc(:,:,j);%(kk((l - 1) * batchsize + 1 : l * batchsize), :, j);

                % Forward transmission
                mm = size(batch_x,1);
                x = [ones(mm,1) batch_x];
                a{1} = x;
                for ii = 2 : n-1
                    a{ii} = 1.7159*tanh(2/3.*(a{ii - 1} * W{ii - 1}'));   
                    a{ii} = [ones(mm,1) a{ii}];
                end

                a{n} = 1./(1+exp(-(a{n - 1} * W{n - 1}')));
                e = batch_y - a{n};

                % Backpropagation
                dd{n} = -e.*(a{n}.*(1 - a{n}));
                for ii = (n - 1) : -1 : 2
                    d_act = 1.7159 * 2/3 * (1 - 1/(1.7159)^2 * a{ii}.^2);

                    if ii+1==n    
                        dd{ii} = (dd{ii + 1} * W{ii}) .* d_act; 
                    else 
                        dd{ii} = (dd{ii + 1}(:,2:end) * W{ii}).* d_act;
                    end          
                end

                for ii = 1 : n-1
                    if ii + 1 == n
                        dW{ii} = dW{ii} + (dd{ii + 1}' * a{ii}) / size(dd{ii + 1}, 1) + lambda*W{ii};
                    else
                        dW{ii} = dW{ii} + (dd{ii + 1}(:,2:end)' * a{ii}) / size(dd{ii + 1}, 1) + lambda*W{ii};      
                    end
                end
            end
        end
        for ii = 1:n-1
            dW_tot{ii} = 0.1*dW{ii} + noise{ii};
        end

        % Parameter update
        for ii = 1 : n - 1       
            W{ii} = W{ii} - learningRate*dW_tot{ii};
        end
    end

    % Test process
    mm = size(test_x,1);
    x = [ones(mm,1) test_x];
    a{1} = x;
    for ii = 2 : n-1    
        a{ii} = 1.7159 * tanh( 2/3 .* (a{ii - 1} * W{ii - 1}'));  
        a{ii} = [ones(mm,1) a{ii}];
    end
    a{n} = 1./(1+exp(-(a{n - 1} * W{n - 1}')));

    [~, i] = max(a{end},[],2);
    labels = i;                         % Prediction variable
    [~, expected] = max(test_y,[],2);
    bad = find(labels ~= expected);     % Error
    er = numel(bad) / size(x, 1);       % Accuracy or error rate
    accuracy_noma(numepochs/10) = 1-er;
    zeta_tem = zeros(1,100);
    zeta_sq = 0;
end
semilogy(10:10:100,accuracy_noma,'-r','LineWidth',2,'Marker','o')
hold on


%% NOMA condition with IRS continuous phase (Adaptive PA)
zeta_tem = zeros(1,100);
noise6_cIRS = zeros(1,d);
accuracy_cIRS = zeros(1,10);
g_mat_cIRS = zeros(1,K,eblock);
threshould = 1e-8;
theta_opt = zeros(N,N,eblock,7);
theta_tem = zeros(N,N,eblock);
for e = 1:eblock
    theta_tem(:,:,e) = diag(exp(1j*unifrnd(0,2*pi,1,N)));
end
for numepochs = 10:10:70
    W = W_ori;
    temp_value = [1e10,1e8];
    
    % We will use CVX to compute ¦È given c and ¦Ò
    while temp_value(1)-temp_value(2) > threshould
        
        % Firstly, we compute c and ¦Ò given ¦È
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

        
        % Secondly, we will optimaize ¦È given c and ¦Ò
        % We think that the ¦È will not change in each iteration
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
    
    theta_opt(:,:,:,numepochs/10) = theta_tem;

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
    for i = 1 : numepochs
        
        % noise
        for ee = 1:eblock
            c = sqrt(c_vector(1,ee,i));
            noise_tem = noise_noma(i,1+(ee-1)*d/eblock : ee*d/eblock);
            noise6_cIRS(1+(ee-1)*d/eblock : ee*d/eblock) = (1/c)*noise_tem/60000;
        end
        noise_lay1 = noise6_cIRS(1,1:78500);
        noise_lay2 = noise6_cIRS(1,78501:79510);
        noise{1,1} = reshape(noise_lay1,100,785);
        noise{1,2} = reshape(noise_lay2,10,101);        
        
        kk = randperm(Dloc);
        dW_tot{1,1} = zeros(100,785);
        dW_tot{1,2} = zeros(10,101);
        dW{1,1} = zeros(100,785);
        dW{1,2} = zeros(10,101);
        for l = 1 : numbatches
            for j = 1:K
                batch_x = train_x_loc(kk((l - 1) * batchsize + 1 : l * batchsize), :, j);
                batch_y = train_y_loc(kk((l - 1) * batchsize + 1 : l * batchsize), :, j);

                % Forward transmission
                mm = size(batch_x,1);
                x = [ones(mm,1) batch_x];
                a{1} = x;
                for ii = 2 : n-1
                    a{ii} = 1.7159*tanh(2/3.*(a{ii - 1} * W{ii - 1}'));   
                    a{ii} = [ones(mm,1) a{ii}];
                end

                a{n} = 1./(1+exp(-(a{n - 1} * W{n - 1}')));
                e = batch_y - a{n};

                % Backpropagation
                dd{n} = -e.*(a{n}.*(1 - a{n}));
                for ii = (n - 1) : -1 : 2
                    d_act = 1.7159 * 2/3 * (1 - 1/(1.7159)^2 * a{ii}.^2);

                    if ii+1==n    
                        dd{ii} = (dd{ii + 1} * W{ii}) .* d_act; 
                    else 
                        dd{ii} = (dd{ii + 1}(:,2:end) * W{ii}).* d_act;
                    end          
                end

                for ii = 1 : n-1
                    if ii + 1 == n
                        dW{ii} = dW{ii} + (dd{ii + 1}' * a{ii}) / size(dd{ii + 1}, 1) + lambda*W{ii};
                    else
                        dW{ii} = dW{ii} + (dd{ii + 1}(:,2:end)' * a{ii}) / size(dd{ii + 1}, 1) + lambda*W{ii};      
                    end
                end
            end
        end
        for ii = 1:n-1
            dW_tot{ii} = 0.1*dW{ii} + noise{ii};
        end

        % Parameter update
        for ii = 1 : n - 1       
            W{ii} = W{ii} - learningRate*dW_tot{ii};
        end
    end

    % Test process
    mm = size(test_x,1);
    x = [ones(mm,1) test_x];
    a{1} = x;
    for ii = 2 : n-1    
        a{ii} = 1.7159 * tanh( 2/3 .* (a{ii - 1} * W{ii - 1}'));  
        a{ii} = [ones(mm,1) a{ii}];
    end
    a{n} = 1./(1+exp(-(a{n - 1} * W{n - 1}')));

    [~, i] = max(a{end},[],2);
    labels = i;                         % Prediction variable
    [~, expected] = max(test_y,[],2);
    bad = find(labels ~= expected);     % Error
    er = numel(bad) / size(x, 1);       % Accuracy or error rate
    accuracy_cIRS(numepochs/10) = 1-er;
    zeta_tem = zeros(1,100);
    zeta_sq = 0;
end
semilogy(10:10:100,accuracy_cIRS,'-m','LineWidth',2,'Marker','p')
hold on


%% NOMA condition with IRS
zeta_tem = zeros(1,100);
noise6_IRS = zeros(1,d);
accuracy_IRS = zeros(1,10);
g_mat_IRS = zeros(1,K,eblock);    % With IRS
for numepochs = 10:10:70
    W = W_ori;
    
    theta_result = theta_opt(:,:,:,numepochs/10);
    
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

    
    for i = 1 : numepochs
        
        % noise
        for ee = 1:eblock
            c = sqrt(c_vector(1,ee,i));
            noise_tem = noise_noma(i,1+(ee-1)*d/eblock : ee*d/eblock);
            noise6_IRS(1+(ee-1)*d/eblock : ee*d/eblock) = (1/c)*noise_tem/60000;
        end
        noise_lay1 = noise6_IRS(1,1:78500);
        noise_lay2 = noise6_IRS(1,78501:79510);
        noise{1,1} = reshape(noise_lay1,100,785);
        noise{1,2} = reshape(noise_lay2,10,101);        
        
        kk = randperm(Dloc);
        dW_tot{1,1} = zeros(100,785);
        dW_tot{1,2} = zeros(10,101);
        dW{1,1} = zeros(100,785);
        dW{1,2} = zeros(10,101);
        for l = 1 : numbatches
            for j = 1:K
                batch_x = train_x_loc(kk((l - 1) * batchsize + 1 : l * batchsize), :, j);
                batch_y = train_y_loc(kk((l - 1) * batchsize + 1 : l * batchsize), :, j);

                % Forward transmission
                mm = size(batch_x,1);
                x = [ones(mm,1) batch_x];
                a{1} = x;
                for ii = 2 : n-1
                    a{ii} = 1.7159*tanh(2/3.*(a{ii - 1} * W{ii - 1}'));   
                    a{ii} = [ones(mm,1) a{ii}];
                end

                a{n} = 1./(1+exp(-(a{n - 1} * W{n - 1}')));
                e = batch_y - a{n};

                % Backpropagation
                dd{n} = -e.*(a{n}.*(1 - a{n}));
                for ii = (n - 1) : -1 : 2
                    d_act = 1.7159 * 2/3 * (1 - 1/(1.7159)^2 * a{ii}.^2);

                    if ii+1==n    
                        dd{ii} = (dd{ii + 1} * W{ii}) .* d_act; 
                    else 
                        dd{ii} = (dd{ii + 1}(:,2:end) * W{ii}).* d_act;
                    end          
                end

                for ii = 1 : n-1
                    if ii + 1 == n
                        dW{ii} = dW{ii} + (dd{ii + 1}' * a{ii}) / size(dd{ii + 1}, 1) + lambda*W{ii};
                    else
                        dW{ii} = dW{ii} + (dd{ii + 1}(:,2:end)' * a{ii}) / size(dd{ii + 1}, 1) + lambda*W{ii};      
                    end
                end
            end
        end
        for ii = 1:n-1
            dW_tot{ii} = 0.1*dW{ii} + noise{ii};
        end

        % Parameter update
        for ii = 1 : n - 1       
            W{ii} = W{ii} - learningRate*dW_tot{ii};
        end
    end

    % Test process
    mm = size(test_x,1);
    x = [ones(mm,1) test_x];
    a{1} = x;
    for ii = 2 : n-1    
        a{ii} = 1.7159 * tanh( 2/3 .* (a{ii - 1} * W{ii - 1}'));  
        a{ii} = [ones(mm,1) a{ii}];
    end
    a{n} = 1./(1+exp(-(a{n - 1} * W{n - 1}')));

    [~, i] = max(a{end},[],2);
    labels = i;                         % Prediction variable
    [~, expected] = max(test_y,[],2);
    bad = find(labels ~= expected);     % Error
    er = numel(bad) / size(x, 1);       % Accuracy or error rate
    accuracy_IRS(numepochs/10) = 1-er;
    zeta_tem = zeros(1,100);
    zeta_sq = 0;
end
semilogy(10:10:100,accuracy_IRS,'-b','LineWidth',2,'Marker','s')
hold on
legend('Continuous RIS assisted system','1-bit RIS assisted system','Benchmark')
set(gca,'xgrid','on','gridlinestyle',':','Gridalpha',1.0)
set(gca,'ygrid','on','gridlinestyle',':','Gridalpha',1.0)
xlabel('Learning round T','fontname','Arial')
ylabel('Normalized Optimal Gap','fontname','Arial')