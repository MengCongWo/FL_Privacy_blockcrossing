%% Neural network preparation
para = load('para_FL.mat');
filename = dir('*.csv');
filenum = length(filename);
d = 12002;
K = 26;
eblock = 10;
learningRate = 0.5;
numepochs = 10;
batchsize = 60; 
numbatches = batchsize / batchsize;   % Number of batches
lambda = 0.01;
arc = [16 100 100 2];  % Input:16, inherent:100, output:2
n=numel(arc);  % The number of layers
% W_ori = para1.W_ori; 
epsilon = 1000;     % DP guidelines
delta = 0.01;
N = 30;
u = 2;
L = 5e3;
SNR_d = 5;
SNR = SNR_d;
SNR = 10^(SNR/10);
N0 = 1;
P=SNR*d/eblock;
syms x               % DP guidelines
f(x) = sqrt(pi)*x*exp(x^2);
tmp = vpasolve(f(x) == 1/delta);
Rdp = power(sqrt(epsilon+power(double(tmp),2))-double(tmp),2);
Gk = ones(1,26);     % Parameter Gk
gamma = 2.19;


%% Channel preparation
kappa_IB = 5;  % IRS to BS
kappa_DI = 0;  % Decives to IRS
kappa_DB = 0;  % Devices to BS
G = para.G; % sqrt(kappa_IB/(1+kappa_IB))+sqrt(1/(1+kappa_IB))*r_mat_IB;
hr = para.hr; % sqrt(kappa_DI/(1+kappa_DI))+sqrt(1/(1+kappa_DI))*r_mat_DI;
hd = para.hd; % sqrt(kappa_DB/(1+kappa_DB))+sqrt(1/(1+kappa_DB))*r_mat_DB;
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
noise_noma = normrnd(0,N0,10,d);
Dloc = para.Dloc;
ak = zeros(1,K);
for j = 1:K
    ak(j) = power(Dloc(j)*Gk(j),2)/P;
end


%% DP constrained FL system without RIS
W = W_ori;   
zeta_tem = zeros(1,10);
tem_noma = zeros(1,K,eblock);
temp2 = zeros(1,eblock);
for e = 1:eblock    % The channel won't change, so we can select condition firstly
    for t = 1:K
        tem_noma(:,t,e) = abs(g_mat(:,t,e))/(Gk(:,t)*Dloc(t));
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
else
    % Enum searching
    block_round = zeros(eblock+1,numepochs);  % Compute the number of block with iota=0

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

noise6 = zeros(1,d);
for i = 1 : numepochs  
    
    for ee = 1:eblock
        if ee < 10
            c = sqrt(c_vector(1,ee,i));
            noise_tem = noise_noma(i,1+(ee-1)*(d-2)/eblock : ee*(d-2)/eblock);
            noise6(1+(ee-1)*(d-2)/eblock : ee*(d-2)/eblock) = (1/c)*noise_tem/1798;
        else
            c = sqrt(c_vector(1,ee,i));
            noise_tem = noise_noma(i,1+(ee-1)*(d-2)/eblock : end);
            noise6(1+(ee-1)*(d-2)/eblock : end) = (1/c)*noise_tem/1798;
        end
    end
    
    noise_lay1 = noise6(1,1:1700);
    noise_lay2 = noise6(1,1701:11800);
    noise_lay3 = noise6(1,11801:12002);
    noise{1,1} = reshape(noise_lay1,100,17);
    noise{1,2} = reshape(noise_lay2,100,101);  
    noise{1,3} = reshape(noise_lay3,2,101);
    
    dW_tot{1,1} = zeros(100,17);
    dW_tot{1,2} = zeros(100,101);
    dW_tot{1,3} = zeros(2,101);
    dW{1,1} = zeros(100,17);
    dW{1,2} = zeros(100,101);
    dW{1,3} = zeros(2,101);
    
    for z = 1:filenum-6
        dataset = csvread([int2str(z),'.csv'],1);
        train_x = dataset(:,1:16);
        train_y = dataset(:,17);
        for qq = 1:size(train_y,1)
            if train_y(qq,1) == 3
                train_y(qq,1) = 2;
            end
        end
    
        for l = 1 : numbatches
            batch_x = train_x;
            batch_y = train_y;

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
        dW_tot{ii} = (1/26)*dW{ii}+noise{1,ii};
    end

    % Parameter update
    for ii = 1 : n - 1       
        W{ii} = W{ii} - learningRate*dW_tot{ii};
    end
end

% Test process
U = [];
for z = 21:filenum
    dataset = csvread([int2str(z),'.csv'],1);
    U = [U;dataset];
end
train_x = U(:,1:16);
train_y = U(:,17);
for qq = 1:size(train_y,1)
    if train_y(qq,1) == 3
        train_y(qq,1) = 2;
    end
end
mm = size(train_x,1);
x = [ones(mm,1) train_x];
a{1} = x;
for ii = 2 : n-1    
    a{ii} = 1.7159 * tanh( 2/3 .* (a{ii - 1} * W{ii - 1}'));  
    a{ii} = [ones(mm,1) a{ii}];
end
a{n} = 1./(1+exp(-(a{n - 1} * W{n - 1}')));

[~, i] = max(a{end},[],2);
labels = i;                         % Prediction variable
[~, expected] = max(train_y,[],2);
bad = find(labels ~= expected);     % Error
er = numel(bad) / size(x, 1);       % Accuracy or error rate
accuracy_noma = 1-er;


%% RIS-enabled FL system with RIS
W = W_ori;   
zeta_tem = zeros(1,10);
g_mat_cIRS = zeros(1,K,eblock);
threshould = 1e-8;
theta_tem = zeros(N,N,eblock);
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
            tem_noma(:,t,e) = abs(g_mat_cIRS(:,t,e))/(Gk(:,t)*Dloc(t));
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
        tem_noma(:,t,e) = abs(g_mat_cIRS(:,t,e))/(Gk(:,t)*Dloc(t));
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

noise6 = zeros(1,d);
for i = 1 : numepochs  
    
    for ee = 1:eblock
        if ee < 10
            c = sqrt(c_vector(1,ee,i));
            noise_tem = noise_noma(i,1+(ee-1)*(d-2)/eblock : ee*(d-2)/eblock);
            noise6(1+(ee-1)*(d-2)/eblock : ee*(d-2)/eblock) = (1/c)*noise_tem/1798;
        else
            c = sqrt(c_vector(1,ee,i));
            noise_tem = noise_noma(i,1+(ee-1)*(d-2)/eblock : end);
            noise6(1+(ee-1)*(d-2)/eblock : end) = (1/c)*noise_tem/1798;
        end
    end
    
    noise_lay1 = noise6(1,1:1700);
    noise_lay2 = noise6(1,1701:11800);
    noise_lay3 = noise6(1,11801:12002);
    noise{1,1} = reshape(noise_lay1,100,17);
    noise{1,2} = reshape(noise_lay2,100,101);  
    noise{1,3} = reshape(noise_lay3,2,101);
    
    dW_tot{1,1} = zeros(100,17);
    dW_tot{1,2} = zeros(100,101);
    dW_tot{1,3} = zeros(2,101);
    dW{1,1} = zeros(100,17);
    dW{1,2} = zeros(100,101);
    dW{1,3} = zeros(2,101);
    
    for z = 1:filenum-6
        dataset = csvread([int2str(z),'.csv'],1);
        train_x = dataset(:,1:16);
        train_y = dataset(:,17);
        for qq = 1:size(train_y,1)
            if train_y(qq,1) == 3
                train_y(qq,1) = 2;
            end
        end
    
        for l = 1 : numbatches
            batch_x = train_x;
            batch_y = train_y;

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
        dW_tot{ii} = (1/26)*dW{ii}+noise{1,ii};
    end

    % Parameter update
    for ii = 1 : n - 1       
        W{ii} = W{ii} - learningRate*dW_tot{ii};
    end
end

% Test process
U = [];
for z = 21:filenum
    dataset = csvread([int2str(z),'.csv'],1);
    U = [U;dataset];
end
train_x = U(:,1:16);
train_y = U(:,17);
for qq = 1:size(train_y,1)
    if train_y(qq,1) == 3
        train_y(qq,1) = 2;
    end
end
mm = size(train_x,1);
x = [ones(mm,1) train_x];
a{1} = x;
for ii = 2 : n-1    
    a{ii} = 1.7159 * tanh( 2/3 .* (a{ii - 1} * W{ii - 1}'));  
    a{ii} = [ones(mm,1) a{ii}];
end
a{n} = 1./(1+exp(-(a{n - 1} * W{n - 1}')));

[~, i] = max(a{end},[],2);
labels = i;                         % Prediction variable
[~, expected] = max(train_y,[],2);
bad = find(labels ~= expected);     % Error
er = numel(bad) / size(x, 1);       % Accuracy or error rate
accuracy_cIRS = 1-er;


%% RIS-enabled FL system (without DP)
g_mat_IRS = zeros(1,K,eblock);    % With IRS
W = W_ori;
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
        tem_noma(:,t,e) = abs(g_mat_IRS(:,t,e))/(Gk(:,t)*Dloc(t));
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
noise6 = zeros(1,d);
for i = 1 : numepochs  
    
    for ee = 1:eblock
        if ee < 10
            c = sqrt(c_vector(1,ee,i));
            noise_tem = noise_noma(i,1+(ee-1)*(d-2)/eblock : ee*(d-2)/eblock);
            noise6(1+(ee-1)*(d-2)/eblock : ee*(d-2)/eblock) = (1/c)*noise_tem/1798;
        else
            c = sqrt(c_vector(1,ee,i));
            noise_tem = noise_noma(i,1+(ee-1)*(d-2)/eblock : end);
            noise6(1+(ee-1)*(d-2)/eblock : end) = (1/c)*noise_tem/1798;
        end
    end
    
    noise_lay1 = noise6(1,1:1700);
    noise_lay2 = noise6(1,1701:11800);
    noise_lay3 = noise6(1,11801:12002);
    noise{1,1} = reshape(noise_lay1,100,17);
    noise{1,2} = reshape(noise_lay2,100,101);  
    noise{1,3} = reshape(noise_lay3,2,101);
    
    dW_tot{1,1} = zeros(100,17);
    dW_tot{1,2} = zeros(100,101);
    dW_tot{1,3} = zeros(2,101);
    dW{1,1} = zeros(100,17);
    dW{1,2} = zeros(100,101);
    dW{1,3} = zeros(2,101);
    
    for z = 1:filenum-6
        dataset = csvread([int2str(z),'.csv'],1);
        train_x = dataset(:,1:16);
        train_y = dataset(:,17);
        for qq = 1:size(train_y,1)
            if train_y(qq,1) == 3
                train_y(qq,1) = 2;
            end
        end
    
        for l = 1 : numbatches
            batch_x = train_x;
            batch_y = train_y;

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
        dW_tot{ii} = (1/26)*dW{ii}+noise{1,ii};
    end

    % Parameter update
    for ii = 1 : n - 1       
        W{ii} = W{ii} - learningRate*dW_tot{ii};
    end
end

% Test process
U = [];
for z = 21:filenum
    dataset = csvread([int2str(z),'.csv'],1);
    U = [U;dataset];
end
train_x = U(:,1:16);
train_y = U(:,17);
for qq = 1:size(train_y,1)
    if train_y(qq,1) == 3
        train_y(qq,1) = 2;
    end
end
mm = size(train_x,1);
x = [ones(mm,1) train_x];
a{1} = x;
for ii = 2 : n-1    
    a{ii} = 1.7159 * tanh( 2/3 .* (a{ii - 1} * W{ii - 1}'));  
    a{ii} = [ones(mm,1) a{ii}];
end
a{n} = 1./(1+exp(-(a{n - 1} * W{n - 1}')));

[~, i] = max(a{end},[],2);
labels = i;                         % Prediction variable
[~, expected] = max(train_y,[],2);
bad = find(labels ~= expected);     % Error
er = numel(bad) / size(x, 1);       % Accuracy or error rate
accuracy_IRS = 1-er;