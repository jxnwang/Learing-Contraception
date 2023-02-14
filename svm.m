clear all; 
T = readtable("cmc.txt");
[n1,n2,n3] = dividerand(size(T,1),0.6,0.2,0.2);
TrainingSet = T(n1, :);
ValidationSet = T(n2, :);
TestSet = T(n3, :);
kernal_function = ["linear", "gaussian", "polynomial"];
box_constraint = 1:3:24;
N = numel(box_constraint);
err1Train = zeros(3, 1);
err1Val = zeros(3, 1);
err2Train = zeros(N, 1);
err2Val = zeros(N, 1);
%for n=1:3
 %   t = templateSVM('KernelFunction',kernal_function(n), 'BoxConstraint', 8);
  %  svm1 = fitcecoc(TrainingSet(:,1:9),TrainingSet(:,10),...
   %    'Learners',t);
%    err1Train(n) = loss(svm1, TrainingSet(:,1:9),TrainingSet(:,10),"LossFun","classiferror");
 %   err1Val(n) = loss(svm1, ValidationSet(:,1:9),ValidationSet(:,10),"LossFun","classiferror");   
%end
for n=1:N
    t = templateSVM('KernelFunction',"linear", 'BoxConstraint',box_constraint(n));
    svm2 = fitcecoc(TrainingSet(:,1:9),TrainingSet(:,10),...
       'Learners',t);
    err2Train(n) = loss(svm2, TrainingSet(:,1:9),TrainingSet(:,10),"LossFun","classiferror");
    err2Val(n) = loss(svm2, ValidationSet(:,1:9),ValidationSet(:,10),"LossFun","classiferror");
end
figure;
hold on;
plot([1 : 3],err1Train,'x','MarkerSize',12);
plot([1 : 3],err1Val,'o','MarkerSize',12);
xticks(0:3)
xticklabels([" ",kernal_function])
xlim([1,3])
hold off;
xlabel("Kernal Function")
ylabel("Error")
legend("training", "cross validation")
figure;
hold on;
plot(box_constraint,err2Train,'-x');
plot(box_constraint,err2Val,'-o');
hold off;
xlabel("Box Constraint")
ylabel("Error")
legend("training", "cross validation");
t = templateSVM('KernelFunction',"linear", 'BoxConstraint', 8);
svmFinal = fitcecoc(TrainingSet(:,1:9),TrainingSet(:,10),... 
     'Learners',t);
errFinal = loss(svmFinal, TestSet(:,1:9),TestSet(:,10),"LossFun","classiferror");