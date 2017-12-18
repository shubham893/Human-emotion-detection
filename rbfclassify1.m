clear;
close all;
clc;
hidaccu = zeros(3,1);
x = zeros(3,1);
for q = 1 : 5
Ntrain=load('input.txt');
[PTD,l] = size(Ntrain);
inp = l-2;   % No. of input neurons
xx = Ntrain(1:PTD,1:inp);
tt = Ntrain(1:PTD,end);
numcats = size(unique(tt), 1);
zz(1:PTD,1:numcats)=-1;
for j = 1 : PTD
    zz(j,tt(j,1))=1;
end
hid = floor ( PTD / (0.2+q));
free1 = floor (hid/numcats);
hid = numcats * free1;
% No. of hidden neurons
out = 1;
max_iters=100;
centroids = zeros(hid, size(xx, 2));
for z = 1 : numcats
input1=xx((tt == z),:);
% Randomly reorder the indices of examples
randidx = randperm(size(input1, 1));

        centroids((z-1)*free1 + 1 : free1*z, :) = input1(randidx(1 : free1), :);
        prevcenter = centroids ;
        for i = 1 : max_iters
            membership = zeros(size(input1,1),1);
            distances = zeros(size(input1,1),free1);
            for j = 1 : free1
                diffs = bsxfun(@minus,input1, prevcenter((z-1)*free1 + j , :));
                sqrdDiffs = diffs .^ 2;
                distances(:, j) = sum(sqrdDiffs, 2);
            end
            [minVals memberships] = min(distances, [], 2);
            for j =(z-1)*free1 + 1 : free1*z
                if (~any(memberships == j))
                    centroids(j, :) = prevcenter(j, :);
                else
        % Select the data points assigned to centroid k.
                    points = input1((memberships == j - (z-1)*free1 ), :);
% Compute the new centroid as the mean of the data points.
                    centroids(j, :) = mean(points);    
                end
            end
            if(prevcenter ==  centroids)
                break;
            end
            prevcenter = centroids;
        end
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
house = zeros (PTD , hid+1);
house(:,hid + 1) = 1;
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

weight = pinv (house) * zz ;
%testing
 Ntrain=load('output.txt');
 [PTD,l] = size(Ntrain);
 xx = Ntrain(1:PTD,1:l-2);
 house = zeros (PTD , hid+1);
 house(:,hid + 1) = 1;
 cp = zeros(PTD,1);
 tt = Ntrain(1:PTD,end);
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
 finale = house * weight;
 finale = finale';
 [~,cp] = max(finale);
 cp=cp';
 conf=zeros(numcats,numcats);
 misclassify = 0;
for j =1 : PTD
    if(~( cp(j,:) == tt(j,:)))
        misclassify = misclassify + 1;
    end
    conf(cp(j,:),tt(j,:))=conf(cp(j,:),tt(j,:))+1;
 end
 hidaccu(q,1) = ((PTD-misclassify)/PTD)*100;
 x(q,1) = hid;
 disp(hidaccu(q,1))
 disp(hid)
 disp(conf)
end
% plotting graph
figure(1);
hold on;
plot(x,hidaccu,'k-');
xlabel('no of hidden neurons')
ylabel('accuracy')
title('classifiaction of curvelet');
