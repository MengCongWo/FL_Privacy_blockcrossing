%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Function: Off-cond RIS-FL vs normal FL (汍). %%%
%%% Author: TSF                                 %%%
%%% Time: 2021.05.13                            %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear all
clc


%% Note
% This model is for offline condition with different privacy level.


%% Model preparation (Ridge regression)
Dtot = 10000;
d = 10;
U = normrnd(0,1,d,Dtot);
N = 30;       % The size of IRS
V = zeros(1, Dtot);
z = normrnd(0,1,1,Dtot);       % Data noise
lamda = 5e-5;
eblock = 5;  % Communication block
for i = 1:Dtot
    V(i) = U(2,i) + 3*U(5,i) + 0.2*z(i);
end
K = 10;        % Number of devices
Dloc = Dtot / K;
U_loc = zeros(d,Dloc,K);
V_loc = zeros(1,Dloc,K);
for i = 1:K
    U_loc(:,:,i) = U(:,1000*(i-1)+1 : 1000*i);
    V_loc(:,:,i) = V(:,1000*(i-1)+1 : 1000*i);
end
temp_matrix = U*U'/Dtot+2*lamda*eye(d);
u = min(eig(temp_matrix));  % PL condition
L = max(eig(temp_matrix));
w_opt = (U*U'+2*Dtot*lamda*eye(d))\U*V';  % Oprimal w
w_ori = zeros(d,1);
F_opt = (1/Dtot)*0.5*(w_opt'*U-V)*(w_opt'*U-V)' + lamda*w_opt'*(w_opt);


%% Model preparation (Federated learning)
T_noma = 30;    % Learning rounds
Block = T_noma*eblock;       % Blocks and iterations
%epsilon = 20;     % DP guidelines (it will change)
delta = 0.01;
SNR_d = 35;
% P = 1e4;       % Suppose (with 考^2=1e-12)
SNR = SNR_d;
SNR = 10^(SNR/10);
% N0 = P/d/SNR*eblock;
N0 = 1;
P = SNR*d/eblock;
w = w_ori';  % Initialize
W = 3.2;     % From the paper
eigen = zeros(1,Dtot);
for i = 1:Dtot
    eigen(i) = max(eig(U(:,i)*U(:,i)'));
end
L_nR = 0.5*max(eigen);
% L_nR = max(eig(U*U'/Dtot));      % Lipschite constant is the maximum eigenvalue of X'X
gamma = 2*W*L_nR;
eta = 1/L;           % Step size
syms x               % DP guidelines
f(x) = sqrt(pi)*x*exp(x^2);
tmp = vpasolve(f(x) == 1/delta);
Gk = zeros(1,K);     % Parameter Gk
for i = 1:K
    Gk(i) = 2*W*max(eig(U_loc(:,:,i)*U_loc(:,:,i)'/Dloc+2*lamda*eye(d)));
end
ak = zeros(1,K);
for j = 1:K
    ak(j) = power(Dloc*Gk(j),2)/P;
end


%% Channel preparation2 (AR Rician channel)
kappa_IB = 5;  % IRS to BS
kappa_DI = 0;  % Decives to IRS
kappa_DB = 5;  % Devices to BS
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
noise_noma = normrnd(0,0.5*N0,T_noma,d);


%% NOMA condition with DP (without RIS)
zeta_tem = zeros(1,T_noma);
noise6 = zeros(1,d);
grad = zeros(K,d);
F_noma = zeros(1,100);
gap_noma = zeros(1,100);
for epsilon = 1:100
    w = w_ori';
    Rdp = power(sqrt(epsilon+power(double(tmp),2))-double(tmp),2);

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
    c_vector = zeros(1,eblock,T_noma);
    if T_noma*temm < Rdp  % We first justify condition (a)
        for i = 1:T_noma
            for e = 1:eblock
                c_vector(1,e,i) = P*temp2(e)^2;
            end
        end
    else  % The others are (b) and (c)   
        % Enum searching
        block_round = zeros(eblock+1,T_noma);  % Compute the number of block with iota=0
        
        % We say iota~=0, then we have iota=(1-u/L)^(-t)
        % We say iota=0, then we have psi~=0, i.e., meet full power
        
        % Firstly, we set every tau=1 for initialization
        for k = 1:T_noma
            val = 0;
            for j = 1:k
                val = val + 2*gamma*(1-u/L)^(-j/2);
            end    
            zeta_sq = val/(Rdp-temm*(T_noma-k));
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
        for i=1:T_noma
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
            zeta_tem = zeros(1,T_noma);
            for k = 1:T_noma
                val = 0;
                for j = 1:k
                    val = val + 2*sqrt(sum(block_round(:,j)))*gamma*(1-u/L)^(-j/2); 
                end    
                zeta_sq = val/(Rdp-temm*(T_noma-k));
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
        
            for i=1:T_noma
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
            for i=1:T_noma
                tmm = 0;
                for e=1:eblock
                    tmm=tmm+1/c_vector(1,e,i);
                end
                obj_fin(2)=obj_fin(2)+(1-u/L)^(-i);
            end
        end
    end

    
    % Update process
    for i = 1:T_noma
        
        % Design the noise vector
        for e = 1:eblock
            c = sqrt(c_vector(1,e,i));
            noise_tem = noise_noma(i,1+(e-1)*d/eblock : e*d/eblock);
            noise6(1+(e-1)*d/eblock : e*d/eblock) = (1/c)*noise_tem;
        end
        
        % Update
        for j = 1:K 
            grad(j,:) = (w*U_loc(:,:,j)-V_loc(:,:,j))*(U_loc(:,:,j)')+2*lamda*w;
        end
        grad_all = (sum(grad)+noise6)/Dtot;
        w = w - eta*grad_all;
    end
    zeta_tem = zeros(1,T_noma);
    zeta_sq = 0;
    F_noma(epsilon) = (1/Dtot)*0.5*(w*U-V)*(w*U-V)' + lamda*w*(w');
    gap_noma(epsilon) = (F_noma(epsilon)-F_opt)/F_opt;
end
semilogy(1:100,gap_noma,'-r','LineWidth',2,'Marker','o','MarkerIndices',1:9:100)
hold on


%% NOMA condition with RIS (without DP)
noise6_IRS = zeros(1,d);
grad = zeros(K,d);
g_mat_IRS = zeros(1,K,eblock);
c_vector = zeros(1,eblock);
theta_tem = zeros(N,N,eblock);
threshould = 1e-8;
for e = 1:eblock
    theta_tem(:,:,e) = diag(exp(1j*unifrnd(0,2*pi,1,N)));
end
w = w_ori';
temp_value = [1e10,1e8];
while temp_value(1)-temp_value(2) > threshould
    
    % We will use CVX to compute 牟 given c and 考
    % Firstly, we compute c and 考 given 牟
    for e = 1:eblock
        for j = 1:K
            g_mat_IRS(:,j,e) = (G(:,:,e)')*theta_tem(:,:,e)*hr(:,j,e)+hd(:,j,e);
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
    for e = 1:eblock
        c_vector(1,e) = P*temp2(e)^2;
    end

    % Secondly, we will optimaize 牟 given c and 考
    % We think that the 牟 will not change in each iteration
    ak_all = zeros(K,eblock);
    for e = 1:eblock
        for j = 1:K
            ak_all(j,e) = c_vector(1,e)^2*ak(j)-abs(hd(:,j,e))^2;
        end
    end

    % We use CVX to obtain a sequence of theta
    cvx_begin quiet
        variable VV(N+1,N+1,eblock) complex semidefinite;
        minimize(0);
        subject to
            for j = 1:K
                for e = 1:eblock
                    trace(R(:,:,j,e)*VV(:,:,e))>=ak_all(j,e);
                end
            end
            for e = 1:eblock
                for n = 1:N
                    VV(n,n,e) == 1;
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
    for t = 1:T_noma
        tmm = 0;
        for e = 1:eblock
            tmm = tmm + 1/c_vector(1,e);
        end
        temp_value(2) = temp_value(2)+tmm;
    end
end

for e = 1:eblock
    for j = 1:K
        g_mat_IRS(:,j,e) = (G(:,:,e)')*theta_tem(:,:,e)*hr(:,j,e)+hd(:,j,e);
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
c_vector = zeros(1,eblock);
for e = 1:eblock
    c_vector(1,e) = P*temp2(e)^2;
end

% Get the vector c 
for i = 1:T_noma

    % Design the noise vector
    for e = 1:eblock
        c = sqrt(c_vector(1,e));
        noise_tem = noise_noma(i,1+(e-1)*d/eblock : e*d/eblock);
        noise6(1+(e-1)*d/eblock : e*d/eblock) = (1/c)*noise_tem;
    end

    % Update
    for j = 1:K 
        grad(j,:) = (w*U_loc(:,:,j)-V_loc(:,:,j))*(U_loc(:,:,j)')+2*lamda*w;
    end
    grad_all = (sum(grad)+noise6)/Dtot;
    w = w - eta*grad_all;
end
F_noma_IRS = (1/Dtot)*0.5*(w*U-V)*(w*U-V)' + lamda*w*(w');
gap_noma_IRS = (F_noma_IRS-F_opt)/F_opt;
semilogy(1:100,gap_noma_IRS*ones(1,100),'-b','LineWidth',2,'Marker','s','MarkerIndices',1:9:100)
hold on


%% NOMA condition with IRS continuous phase (Adaptive PA)
zeta = 0;
zeta_tem = zeros(1,T_noma);
noise6_cIRS = zeros(1,d);
grad = zeros(K,d);
F_noma_cIRS = zeros(1,100);
gap_noma_cIRS = zeros(1,100);
g_mat_cIRS = zeros(1,K,eblock);
c_vector = zeros(1,T_noma);
threshould = 1e-8;
theta_tem = zeros(N,N,eblock);
for e = 1:eblock
    theta_tem(:,:,e) = diag(exp(1j*unifrnd(0,2*pi,1,N)));
end
for epsilon = 1:100
    w = w_ori';
    Rdp = power(sqrt(epsilon+power(double(tmp),2))-double(tmp),2);
    temp_value = [1e10,1e8];
    
    % We will use CVX to compute 牟 given c and 考
    while temp_value(1)-temp_value(2) > threshould
        
        % Firstly, we compute c and 考 given 牟
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
        c_vector = zeros(1,eblock,T_noma);
        if T_noma*temm < Rdp  % We first justify condition (a)
            for i = 1:T_noma
                for e = 1:eblock
                    c_vector(1,e,i) = P*temp2(e)^2;
                end
            end
        else  % The others are (b) and (c)   
            % Enum searching
            block_round = zeros(eblock+1,T_noma);  % Compute the number of block with iota=0

            % We say iota~=0, then we have iota=(1-u/L)^(-t)
            % We say iota=0, then we have psi~=0, i.e., meet full power

            % Firstly, we set every tau=1 for initialization
            for k = 1:T_noma
                val = 0;
                for j = 1:k
                    val = val + 2*gamma*(1-u/L)^(-j/2);
                end    
                zeta_sq = val/(Rdp-temm*(T_noma-k));
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
            for i=1:T_noma
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
                zeta_tem = zeros(1,T_noma);
                for k = 1:T_noma
                    val = 0;
                    for j = 1:k
                        val = val + 2*sqrt(sum(block_round(:,j)))*gamma*(1-u/L)^(-j/2); 
                    end    
                    zeta_sq = val/(Rdp-temm*(T_noma-k));
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

                for i=1:T_noma
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
                for i=1:T_noma
                    tmm = 0;
                    for e=1:eblock
                        tmm=tmm+1/c_vector(1,e,i);
                    end
                    obj_fin(2)=obj_fin(2)+(1-u/L)^(-i);
                end
            end
        end

        
        % Secondly, we will optimaize 牟 given c and 考
        % We think that the 牟 will not change in each iteration
        ak_all = zeros(K,eblock,T_noma);
        for t = 1:T_noma
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
                for t = 1:T_noma
                    for j = 1:K
                        for e = 1:eblock
                            trace(R(:,:,j,e)*VV(:,:,e))>=ak_all(j,e,t);
                        end
                    end
                end
                for e = 1:eblock
                    for n = 1:N
                        VV(n,n,e) == 1;
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
        for t = 1:T_noma
            tmm = 0;
            for e = 1:eblock
                tmm = tmm + 1/c_vector(1,e,t);
            end
            temp_value(2) = temp_value(2)+tmm;
        end
    end

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
    c_vector = zeros(1,eblock,T_noma);
    if T_noma*temm < Rdp  % We first justify condition (a)
        for i = 1:T_noma
            for e = 1:eblock
                c_vector(1,e,i) = P*temp2(e)^2;
            end
        end
    else  % The others are (b) and (c)   
        % Enum searching
        block_round = zeros(eblock+1,T_noma);  % Compute the number of block with iota=0

        % We say iota~=0, then we have iota=(1-u/L)^(-t)
        % We say iota=0, then we have psi~=0, i.e., meet full power

        % Firstly, we set every tau=1 for initialization
        for k = 1:T_noma
            val = 0;
            for j = 1:k
                val = val + 2*gamma*(1-u/L)^(-j/2);
            end    
            zeta_sq = val/(Rdp-temm*(T_noma-k));
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
        for i=1:T_noma
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
            zeta_tem = zeros(1,T_noma);
            for k = 1:T_noma
                val = 0;
                for j = 1:k
                    val = val + 2*sqrt(sum(block_round(:,j)))*gamma*(1-u/L)^(-j/2); 
                end    
                zeta_sq = val/(Rdp-temm*(T_noma-k));
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

            for i=1:T_noma
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
            for i=1:T_noma
                tmm = 0;
                for e=1:eblock
                    tmm=tmm+1/c_vector(1,e,i);
                end
                obj_fin(2)=obj_fin(2)+(1-u/L)^(-i);
            end
        end
    end

    % Get the vector c 
    for i = 1:T_noma
        
        % Design the noise vector
        for e = 1:eblock
            c = sqrt(c_vector(1,e,i));
            noise_tem = noise_noma(i,1+(e-1)*d/eblock : e*d/eblock);
            noise6(1+(e-1)*d/eblock : e*d/eblock) = (1/c)*noise_tem;
        end
        
        % Update
        for j = 1:K 
            grad(j,:) = (w*U_loc(:,:,j)-V_loc(:,:,j))*(U_loc(:,:,j)')+2*lamda*w;
        end
        grad_all = (sum(grad)+noise6)/Dtot;
        w = w - eta*grad_all;
    end
    zeta_tem = zeros(1,T_noma);
    zeta = 0;
    zeta_sq = 0;
    F_noma_cIRS(epsilon) = (1/Dtot)*0.5*(w*U-V)*(w*U-V)' + lamda*w*(w');
    gap_noma_cIRS(epsilon) = (F_noma_cIRS(epsilon)-F_opt)/F_opt;
end
semilogy(1:100,gap_noma_cIRS,'-m','LineWidth',2,'Marker','p','MarkerIndices',1:9:100)
hold on
legend('Benchmark','RIS-enabled system without DP','RIS-enabled system with DP')
set(gca,'xgrid','on','gridlinestyle',':','Gridalpha',1.0)
set(gca,'ygrid','on','gridlinestyle',':','Gridalpha',1.0)
xlabel('Privacy level 汍','fontname','Arial')
ylabel('Normalized Optimal Gap','fontname','Arial')