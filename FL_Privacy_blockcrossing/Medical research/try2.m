clear all
clc

learningRate = 0.5;
lambda = 0.01;
arc = [16 100 100 2];  % Input:16, inherent:100, output:2
n=numel(arc);
filenum = 26;
numbatches = 1;
W_ori = cell(1,4-1); 
for i=2:4
    W_ori{i-1} = (rand(arc(i),arc(i-1)+1)-0.5) * 8 *sqrt(6 / (arc(i)+arc(i-1)));
end
W = W_ori;
numepochs = 10;


for i = 1 : numepochs  
    
    dW_tot{1,1} = zeros(100,17);
    dW_tot{1,2} = zeros(100,101);
    dW_tot{1,3} = zeros(2,101);
    dW{1,1} = zeros(100,17);
    dW{1,2} = zeros(100,101);
    dW{1,3} = zeros(2,101);
    
    for z = 1:filenum-6
        dataset = csvread([int2str(z),'.csv'],1);
        train_x = dataset(:,1:16);
%         for pp = 1:16
%             train_x(:,pp) = train_x(:,pp)/max(train_x(:,pp));
%             mu = mean(train_x(:,pp));    
%             sigma = max(std(train_x(:,pp)),eps);
%             train_x(:,pp) = bsxfun(@minus,train_x(:,pp),mu);  % Noramlized mean value
%             train_x(:,pp) = bsxfun(@rdivide,train_x(:,pp),sigma);
%         end
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
        dW_tot{ii} = (1/26)*dW{ii};
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
    train_tmp = dataset;
%     for pp = 1:16
%         train_tmp(:,pp) = train_tmp(:,pp)/max(train_tmp(:,pp));
%         mu = mean(train_tmp(:,pp));    
%         sigma = max(std(train_tmp(:,pp)),eps);
%         train_tmp(:,pp) = bsxfun(@minus,train_tmp(:,pp),mu);  % Noramlized mean value
%         train_tmp(:,pp) = bsxfun(@rdivide,train_tmp(:,pp),sigma);
%     end
    U = [U;train_tmp];
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