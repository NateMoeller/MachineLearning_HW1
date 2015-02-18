function J = Bayes_testing(test_data, p1, p2, pc1, pc2)
    data = importdata(test_data);
    
    classes = data(:, end);
    numErrors = 0;
    numExamples = size(data, 1);
    for i = 1 : size(data, 1) %loop through each row
        runningProb1 = 1;
        runningProb0 = 1;
        for j = 1 : size(data, 2) - 1 %loop through each column
            if data(i, j) == 1
               prob1 = p1^1;
               prob0 = p2^1;
            elseif data(i, j) == 0
               prob1 = (1-p1)^1;
               prob0 = (1-p2)^1;
            end
            runningProb1 = runningProb1 * prob1;
            runningProb0 = runningProb0 * prob0;
        end
        
        %calculate posteriors
        posterior1 = runningProb1 * pc1;
        posterior0 = runningProb0 * pc2;
        
        %classify
        class = classes(i, 1);
        if posterior1 > posterior0
            %disp('class1');
            if class ~= 1
               numErrors = numErrors + 1; 
            end
        elseif posterior0 > posterior1
            %disp('class0');
            if class ~= 0
               numErrors = numErrors + 1; 
            end
        end
    end
    
    %output error rate
    errorRate = numErrors / numExamples
end