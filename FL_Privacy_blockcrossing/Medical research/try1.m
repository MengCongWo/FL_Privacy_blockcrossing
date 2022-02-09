% W_ori = cell(1,4-1); 
% for i=2:4
%     W_ori{i-1} = (rand(arc(i),arc(i-1)+1)-0.5) * 8 *sqrt(6 / (arc(i)+arc(i-1)));
% end
W=W_ori;

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
accuracy_wonoisr = 1-er;