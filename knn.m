clear all; 
T = readtable("cmc.txt");
[n1,n2,n3] = dividerand(size(T,1),0.6,0.2,0.2);
TrainingSet = T(n1, :);
ValidationSet = T(n2, :);
TestSet = T(n3, :);
k = 1:3:60;
distance = ["cityblock" , "chebychev" , "correlation" , "cosine" , "euclidean" , "hamming"];
err1Train = zeros(length(k), 1);
err1Val = zeros(length(k), 1);
err2Train = zeros(6, 1);
err2Val = zeros(6, 1);
for i = 1 : length(k)
    k_nn = fitcknn(TrainingSet(:,1:9),TrainingSet(:,10), "Distance", "cosine", "NumNeighbors", ...
        k(i));
    err1Train(i) = loss(k_nn,TrainingSet(:,1:9),TrainingSet(:,10),"LossFun","classiferror");
    err1Val(i) = loss(k_nn, ValidationSet(:,1:9),ValidationSet(:,10),"LossFun","classiferror");
end
for i = 1 : 6
    k_nn = fitcknn(TrainingSet(:,1:9),TrainingSet(:,10), "Distance", distance(i), "NumNeighbors", 40);
    err2Train(i) = loss(k_nn, TrainingSet(:,1:9),TrainingSet(:,10),"LossFun","classiferror");
    err2Val(i) = loss(k_nn, ValidationSet(:,1:9),ValidationSet(:,10),"LossFun","classiferror");
end
figure;
hold on;
plot(k,err1Train,'-x');
plot(k,err1Val,'-o');
hold off;
xlabel("k") 
ylabel("Error")
legend("training", "cross validation")
figure;
hold on;
plot([1 : 6],err2Train,'x','MarkerSize',12);
plot([1 : 6],err2Val,'o','MarkerSize',12);
xticks(1:6)
xticklabels(distance)
hold off;
xlabel("Distance Defination")
ylabel("Error")
legend("training", "cross validation")
knnFinal = fitcknn(TrainingSet(:,1:9),TrainingSet(:,10), "Distance", "cosine", "NumNeighbors", 40);
errFinal = loss(knnFinal, TestSet(:,1:9),TestSet(:,10),"LossFun","classiferror");