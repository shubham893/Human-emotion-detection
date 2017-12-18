% Program for  RBF..........................................
% Update weights for a given epoch

clear all
close all
clc

% Load the training data..................................................
X=load('her.tra');
[m,~] = size(X);
K = 3;
features = size(X,2)-1;
% Initialize the Algorithm Parameters.....................................
train_features = [];
x = [];
for i=1:features
    x = X(:,i);
    train_features = [train_features x];
end
perma_train = train_features;
actual_output = X(:,features+1);


centroids = zeros(K, size(train_features,2));
disp(size(centroids))
pause
randidx = randperm(size(train_features,1));
%randidx = [1 ;2 ;3];
centroids = train_features(randidx(1:K), :);
c = zeros(m,1);
idx = zeros(size(X,1), 1); 
for q=1:25
	x = train_features;
	for i=1:K
		c = centroids(i,:);
		distance(:,i) = sum(bsxfun(@minus, x, c).^2,2);
	end
	[~,idx] = min(distance,[],2);
	for i=1:K
		temp = (idx==i);
		t=x.*temp;
		centroids(i,:)=sum(t,1)./sum(temp);
	end
    	
end
disp(centroids)

%plot(x1, x2, 'rx', 'MarkerSize', 10, 'LineWidth', 1.5)
%hold on
%plot(centroids(:,1), centroids(:,2), 'bx', 'MarkerSize', 40, 'LineWidth', 1.5)





for i=1:K
	center = centroids(i,:);
	members = train_features((idx == i),:) ;   
	differences = bsxfun(@minus, members, center);
	sqrdDiffs = sum(differences.^2,2);
	sigmas(i,:) = mean(sqrdDiffs);
end

phi = [];
disp(sigmas);
sigmas = 1./sigmas;
sigmas = -sigmas./2;

%for i=1:K
%	phi(:,i) = sigmas(i).*(sum((train_features-centroids(i,:)).^2,2));
%end
%phi = exp(phi);

for i=1:m
	for j=1:K
		phi(i,j) = exp(sigmas(j).*sum(((train_features(i,:)-centroids(j,:)).^2)));
	end
end


disp(phi)
phi = [phi ones(m,1)]
pause
%phi
weights = pinv(phi'*phi)*phi'*actual_output;
disp('Weights')
disp(weights)
%size(weights)
%-------------------------------------------------------------testing now


Y=load('her.tes');
[n,~] = size(Y);
test_features = [];
y = [];
test_features = Y(:,1:end-1);
test_output = Y(:,end);
bnsf=[];
for i=1:n
	for j=1:K
		bnsf(i,j) = exp(sigmas(j).*sum(((test_features(i,:)-centroids(j,:)).^2)));
	end
end
bnsf = [bnsf ones(n,1)];

%plot(Y(:,1), Y(:,2),'rx', 'MarkerSize', 10, 'LineWidth', 1.5)
%hold on
%plot(centroids(:,1), centroids(:,2), 'bx', 'MarkerSize', 40, 'LineWidth', 1.5)
predictions = bnsf * weights;
error = test_output-predictions;
[test_output predictions error]
error = error.^2;
sqrt(mean(error))
L = 1:n;
plot(L,predictions,'k','LineWidth',1)
hold on
P = 1:n;
disp(bnsf);
plot(P,test_output,'r','LineWidth',1)
