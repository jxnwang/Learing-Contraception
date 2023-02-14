clear all; 
T = readtable("cmc.txt");
[n1,n2,n3] = dividerand(size(T,1),0.6,0.2,0.2);
TrainingSet = T(n1, :);
ValidationSet = T(n2, :);
TestSet = T(n3, :);
method = ["Bag","AdaBoostM2","LPBoost","RUSBoost","TotalBoost"];
no_learning_cycle = 1:10:200;
err1Train = zeros(length(method), 1);
err1Val = zeros(length(method), 1);
err2Train = zeros(length(no_learning_cycle), 1);
err2Val = zeros(length(no_learning_cycle), 1);
for i = 1 : length(method)
    bst = fitcensemble(TrainingSet(:,1:9),TrainingSet(:,10),'Method', method(i), 'Learners','Tree');
    err1Train(i) = loss(bst,TrainingSet(:,1:9),TrainingSet(:,10),"LossFun","classiferror");
    err1Val(i) = loss(bst, ValidationSet(:,1:9),ValidationSet(:,10),"LossFun","classiferror");
end
for i = 1 : length(no_learning_cycle)
    bst = fitcensemble(TrainingSet(:,1:9),TrainingSet(:,10), 'Method','AdaBoostM2', 'NumLearningCycles', no_learning_cycle(i),'Learners','Tree');
    err2Train(i) = loss(bst, TrainingSet(:,1:9),TrainingSet(:,10),"LossFun","classiferror");
    err2Val(i) = loss(bst, ValidationSet(:,1:9),ValidationSet(:,10),"LossFun","classiferror");
end
figure;
hold on;
plot([1 : length(method)],err1Train,'x','MarkerSize',12);
plot([1 : length(method)],err1Val,'o','MarkerSize',12);
xticks(1:length(method))
xticklabels(method)
hold off;
xlabel("ensenble aggregation method") 
ylabel("Error")
legend("training", "cross validation")
figure;
hold on;
plot(no_learning_cycle,err2Train,'-x');
plot(no_learning_cycle,err2Val,'-o');
hold off;
xlabel("nunber of learning cycles")
ylabel("Error")
legend("training", "cross validation")
bstFinal = fitcensemble(TrainingSet(:,1:9),TrainingSet(:,10), 'Method','AdaBoostM2', 'NumLearningCycles', 75,'Learners','Tree');
errFinal = loss(bstFinal, TestSet(:,1:9),TestSet(:,10),"LossFun","classiferror");