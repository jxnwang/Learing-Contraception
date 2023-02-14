clear all; 
T = readtable("cmc.txt");
[n1,n2,n3] = dividerand(size(T,1),0.6,0.2,0.2);
TrainingSet = T(n1, :);
ValidationSet = T(n2, :);
TestSet = T(n3, :);
lambda = (0:2:28)*1/1400;
iteration_limit = 1:2:70;
N1 = numel(lambda);
N2 = numel(iteration_limit);
err1Train = zeros(N1, 1);
err1Val = zeros(N1, 1);
err2Train = zeros(N2, 1);
err2Val = zeros(N2, 1);
for n=1:N1
    nnk1 = fitcnet(TrainingSet(:,1:9),TrainingSet(:,10),...
        "Lambda",lambda(n), ...
        "Standardize",true);
    err1Train(n) = loss(nnk1, TrainingSet(:,1:9),TrainingSet(:,10),"LossFun","classiferror");
    err1Val(n) = loss(nnk1, ValidationSet(:,1:9),ValidationSet(:,10),"LossFun","classiferror");
end
%for n=1:N2
%    nnk2 = fitcnet(TrainingSet(:,1:9),TrainingSet(:,10),...
%        "IterationLimit",iteration_limit(n), ...
%        "Standardize",true);
%    err2Train(n) = loss(nnk2, TrainingSet(:,1:9),TrainingSet(:,10),"LossFun","classiferror");
%    err2Val(n) = loss(nnk2, ValidationSet(:,1:9),ValidationSet(:,10),"LossFun","classiferror");
%end
figure();
hold on;
plot(lambda,err1Train,'-x');
plot(lambda,err1Val,'-o');
hold off;
legend("training", "cross validation")
xlabel('lambda');
ylabel('error');
%figure();
%hold on;
%plot(iteration_limit,err2Train,'-x');
%plot(iteration_limit,err2Val,'-o');
%hold off;
%legend("training", "cross validation")
%xlabel('iteration limit');
%ylabel('error');
nnkFinal = fitcnet(TrainingSet(:,1:9),TrainingSet(:,10),"Lambda",0.0001,"IterationLimit",60,...
        "Standardize",true);
errFinal = loss(nnkFinal,TestSet(:,1:9),TestSet(:,10),"LossFun","classiferror");