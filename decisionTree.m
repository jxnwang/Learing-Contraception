clear all; 
T = readtable("cmc.txt");
[n1,n2,n3] = dividerand(size(T,1),0.6,0.2,0.2);
TrainingSet = T(n1, :);
ValidationSet = T(n2, :);
TestSet = T(n3, :);
%Mdl = fitctree(TrainingSet(:,1:8),TrainingSet(:,11),'OptimizeHyperparameters','auto');
m_leaf_size = 1:3:90;
M_num_splits = 1:2.5:50;
N1 = numel(m_leaf_size);
N2 = numel(M_num_splits);
err1Train = zeros(N1, 1);
err1Val = zeros(N1, 1);
err2Train = zeros(N2, 1);
err2Val = zeros(N2, 1);
for n=1:N1
    curtree1 = fitctree(TrainingSet(:,1:9),TrainingSet(:,10),...
        'MinLeafSize',m_leaf_size(n));
    err1Train(n) = loss(curtree1, TrainingSet(:,1:9),TrainingSet(:,10),"LossFun","classiferror");
    err1Val(n) = loss(curtree1, ValidationSet(:,1:9),ValidationSet(:,10),"LossFun","classiferror");
end
for n=1:N2
    curtree2 = fitctree(TrainingSet(:,1:9),TrainingSet(:,10),...
        'MaxNumSplits',M_num_splits(n));
    err2Train(n) = loss(curtree2, TrainingSet(:,1:9),TrainingSet(:,10),"LossFun","classiferror");
    err2Val(n) = loss(curtree2, ValidationSet(:,1:9),ValidationSet(:,10),"LossFun","classiferror");
end
figure();
hold on;
plot(m_leaf_size,err1Train,'-x');
plot(m_leaf_size,err1Val,'-o');
hold off;
legend("training", "cross validation")
xlabel('minimum leaf size');
ylabel('error');
figure();
hold on;
plot(M_num_splits,err2Train,'-x');
plot(M_num_splits,err2Val,'-o');
hold off;
legend("training", "cross validation")
xlabel('maximum number of splits');
ylabel('error');
treeFinal = fitctree(TrainingSet(:,1:9),TrainingSet(:,10),'MinLeafSize',30,'MaxNumSplits', 15);
errFinal = loss(treeFinal, TestSet(:,1:9),TestSet(:,10));

