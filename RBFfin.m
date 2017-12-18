% Program for  MLP..........................................
% Update weights for a given epoch

clear all
close all
clc

% Load the training data..................................................
Ntrain=load('Liver.tra');
[NTD,~] = size(Ntrain);
Nofeatures = size(Ntrain,2)-1;
NumofClasses = size(unique(Ntrain(:,Nofeatures+1)),1);
Labels = Ntrain(:,Nofeatures+1);
Target = [];
for i=1:NTD
    temp = ones(1,NumofClasses)*(-1);
    temp(Labels(i))=1;
    Target = [Target;temp];
end;
%target = ones(1,);
% Initialize the Algorithm Parameters.....................................
inp = Nofeatures;          % No. of input neurons or features
k = 25;
n = NumofClasses;% No. of hidden neurons
out = 1;            % No. of Output Neurons
lam = 0.001;
lam2= 0.003;
lam3 = 0.001;
epo = 50;
max_iters = 25;

X=Ntrain(:,1:inp);   %Unsupervised data of the dataset
% Initialize Random k hidden neurons from Ntrain...........................

hneurons = zeros(k, inp);

% Randomly reorder the indices of examples.................................
randidx = randperm(NTD);

% Take the first k examples as centroids...................................
hneurons = Ntrain(1:k,1:inp);

pre_hneurons = hneurons;

% Running K-means algorithm................................................

for i=1:max_iters
    
    % For each example in X, assign it to the closest centroid
    idx = findClosestCentroids(X , hneurons);
    
    
    pre_hneurons = hneurons;
    % Given the memberships, compute new centroids
    hneurons = computeCentroids(X , idx, k);
    % Check for convergence. If the centroids haven't changed since
    % last iteration, we've converged.
    if (pre_hneurons == hneurons)
       break;
    end
end


idx = findClosestCentroids(X , hneurons);

%disp(hneurons);

%{
sigma = zeros(k,1);
C = zeros(k,1);
distsum = zeros(k,1);
for i = 1:NTD
    distsum(idx(i)) = distsum(idx(i)) + ((sum((X(i,:) - hneurons(idx(i))).^2)));
    C(idx(i)) = C(idx(i)) + 1;
end;

for i = 1:k
    sigma(i) = distsum(i)/C(i);
end;
%}

sigma = zeros(k,1);
ma=0;
for i=1:k
    for j=1:k
       temp=sum((hneurons(i)-hneurons(j)).^2);
        if(temp>ma)
            ma=temp;
       end
    end;
end;

for i=1:k
   sigma(i)=sqrt(ma);
end;

%disp(sigma)
phi = zeros(NTD,k);
for i = 1:NTD
    for j = 1:k
        phi(i,j) = exp(-(sum((X(i,:)-hneurons(j,:)).^2))/sigma(j)^2);
    end
end

%disp(phi)

weights = pinv(phi'*phi)*phi'*Target;
%weights = 0.01*(rand(k,n)*2.0-1.0);  % Input weights
%hneurons = rand(k,inp);

  for ep = 1 : epo
      
      %disp(size(f_ans))
      %disp(size(Ntrain(:,inp+1)))
      
      dwi=zeros(1,n);
      dwi2=zeros(1,inp);
      sigerr=0;
          for i = 1:NTD
              for l = 1:k
                 diff = X(i,:)-hneurons(l,:);
                 diff = diff.^2;
          
                 yphi(1,l) = exp(-sum(diff,2)/sigma(l)^2);
              end;   
              
              f_ans = yphi * weights;
              error = Target(i,:) - yphi*weights;
              for sa = 1 : k
                dwi =  error.*yphi(1,sa);
                gradient = yphi(1,sa)*(error*weights(sa,:)')*(X(i,:)-hneurons(sa,:));
                gradient =  gradient/(sigma(sa)^2);
              	temp = yphi(1,sa)*sum((X(i,:)-hneurons(sa,:)).^2) * (error(1,:)*weights(sa,:)'); 
              	temp = temp/(sigma(j)^3);
              	
               %sigma(sa) = sigma(sa) + lam3*(temp);
               %hneurons(sa,:)=hneurons(sa,:)+ lam2*(gradient/(sigma(sa)^2));
               % weights(sa,:) = weights(sa,:)+ lam*dwi;
                sigma(sa) = sigma(sa) + lam3*(temp);
                hneurons(sa,:)=hneurons(sa,:)+ lam2*(gradient);
                weights(sa,:) = weights(sa,:)+ lam*dwi;
                
         
              end
          
           
          %sigma(sa) = sigma(sa) + lam3*(sigerr);
          %hneurons(sa,:)=hneurons(sa,:)+ lam2*(dwi2);
          %weights(sa,:) = weights(sa,:)+ lam*dwi;
          %hneurons(sa,:)=hneurons(sa,:)+ lam2*dwi2;
          
          end;  
      
    %for i = 1:NTD
    %	for j = 1:k
      %  		phi(i,j) = exp(sigma(j).*sum((X(i,:)-hneurons(j,:)).^2));
    	%end
	%end
      
  end
  
plot(X(:,1),X(:,2),'rx','MarkerSize',5,'LineWidth',1.5)
hold
plot(hneurons(:,1),hneurons(:,2),'rx','MarkerSize',15,'LineWidth',1.5)
figure
%weights = pinv(phi'*phi)*phi'*Target;



%Test data and Mean Squared Error....................................

Ntrain=load('Liver.tes');
[PTD,l] = size(Ntrain);
Ncompute=load('Liver.cla');
[LTD,~]=size(Ncompute);
yy=Ncompute(1:LTD,1);
 xx = Ntrain(1:PTD,1:end);
 house = zeros (PTD , size(hneurons,1));
 cp = zeros(PTD,1);
 for i = 1 : PTD
            for j = 1 : size(hneurons,1)
                dist = xx(i,:) - hneurons(j,:);
                sqrddist = dist .^2;
                sqrddist1 = sum (sqrddist,2);
                m = sqrddist1(1,1);
                m = m / (2 * (sigma(j) ^ 2));
                house(i,j) = exp(-m);
            end
 end
 finale = house * weights;
 finale = finale';
 [~,cp] = max(finale);
 cp=cp';
 conftes = zeros(NumofClasses,NumofClasses);
for sa = 1: PTD
         ca1 = yy(sa,1);
         ca2 = cp(sa,1);
         conftes(ca1,ca2) = conftes(ca1,ca2) + 1;
        
end
disp(conftes)
