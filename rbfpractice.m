clear;
close all;
clc;
ita=1.e-05;
Ntrain=load('bj.tra');
[PTD,l] = size(Ntrain);
NTD = floor((PTD*3)/4);
inp = l-1;   % No. of input neurons
hid = floor ( NTD / 40);
% No. of hidden neurons
out = 1;
max_iters=100;
xx = Ntrain(1:NTD,1:inp);
centroids = zeros(hid, size(xx, 2));
% Randomly reorder the indices of examples
randidx = randperm(size(xx, 1));

centroids = xx(randidx(1:hid), :);
prevcenter = centroids;
for i = 1 : max_iters
    membership = zeros(NTD,1);
    distances = zeros(NTD,hid);
    for j = 1 : hid
        diffs = bsxfun(@minus,xx, prevcenter(j, :));
        sqrdDiffs = diffs .^ 2;
        distances(:, j) = sum(sqrdDiffs, 2);
    end
    [minVals memberships] = min(distances, [], 2);
    for j = 1 : hid
        if (~any(memberships == j))
            centroids(j, :) = prevcenter(j, :);
        else
% Select the data points assigned to centroid k.
            points = xx((memberships == j), :);
% Compute the new centroid as the mean of the data points.
            centroids(j, :) = mean(points);    
        end
    end
    if(prevcenter ==  centroids)
        break;
    end
    prevcenter = centroids;
end
maxa=0;
for i = 1 : hid-1
    for j = i+1 : hid
        dist = centroids(i,:)-centroids(j,:);
        sqrddist = dist .^2;
        sqrddist1 = sum(sqrddist,2);
        m = sqrddist1(1,1);
        if (m > maxa)
            maxa = m;
        end
    end
end
sigma = maxa / sqrt(hid);
house = zeros (NTD , hid);
for i = 1 : NTD
    for j = 1 : hid
        dist = xx(i,:) - centroids(j,:);
        sqrddist = dist .^2;
        sqrddist1 = sum (sqrddist,2);
        m = sqrddist1(1,1);
        m = m / (2 * (sigma ^ 2));
        house(i,j) = exp(-m);
    end
end
house = bsxfun(@rdivide, house, sum(house, 2));
house(:,hid+1) = 1;
tt = Ntrain(1:NTD,inp+1:end);
weight = pinv(house' * house) * house' * tt;
%gradient decent algorithm
target = house * weight;
err = tt - target;
for i = 1 : 4
 for j = 1 : hid
     y=0;
     for k = 1 : NTD
     dist1 = xx(k,:) - centroids(j,:);
     sqrddist1 = dist1 .^2;
     sqrddist12 = sum (sqrddist1,2);
     m = sqrddist12(1,1);
     m = m / (2 * (sigma ^ 2));
     x=exp(m);
     y = y + err(k,:)*x;
     end
     weight(j,:)=weight(j,:) - ita*y;
 end
end
%validating the data
xx = Ntrain(1+NTD:PTD,1:inp);       
tt = Ntrain(1+NTD:PTD,inp+1:end);
p = PTD - NTD;
house = zeros (p , hid);
for i = 1 : p
            for j = 1 : hid
                dist = xx(i,:) - centroids(j,:);
                sqrddist = dist .^2;
                sqrddist1 = sum (sqrddist,2);
                m = sqrddist1(1,1);
                m = m / (2 * (sigma ^ 2));
                house(i,j) = exp(-m);
            end
 end
house = bsxfun(@rdivide, house, sum(house, 2));
house(:,hid+1) = 1;
target = house * weight;
err = tt - target;
accu = zeros(p,1);
for i = 1 : p
     accu(i,1)=(abs(tt(i,1))- abs(err(i,1)))/abs(tt(i,1));
end
m = sum(accu,1);
disp((m/p)*100)
x=(1:p)';
% plotting graph
figure(1);
hold on;
plot(x,tt,'k-');
plot(x,target,'r-');
legend('original','approx');
title('RBFN Regression');
% testing data
Ntrain=load('bj.tes');
[PTD,~] = size(Ntrain);
xx = Ntrain(1:PTD,1:end);
house = zeros (PTD , hid);
target = zeros (PTD,1);
for i = 1 : PTD
            for j = 1 : hid
                dist = xx(i,:) - centroids(j,:);
                sqrddist = dist .^2;
                sqrddist1 = sum (sqrddist,2);
                m = sqrddist1(1,1);
                m = m / (2 * (sigma ^ 2));
                house(i,j) = exp(-m);
            end
end
house = bsxfun(@rdivide, house, sum(house, 2));
house(:,hid+1) = 1;
target = house * weight;
disp(target)