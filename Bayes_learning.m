function J = Bayes_learning(train, valid)
    %TODO: change names of parameters?!
    
    A = importdata(train);
    B = importdata(valid);

    %priors
    PC1 = [.1 .2 .3 .4 .5 .6 .7 .8 .9];
    PC0 = [.9 .8 .7 .6 .5 .4 .3 .2 .1];

    output = [];

    %calculate likelihood for class 1 and 0
    N = size(A, 1);
    runningEq1 = 0;
    runningEq0 = 0;
    for i = 1 : size(A, 2) -1 %loop through each feature (column)
        numOnes1 = 0;
        numOnes0 = 0;
        for j = 1 : size(A, 1) %loop through each row (sample)
            class = A(j, end);
            if A(j, i) == 1 && class == 1
               numOnes1 = numOnes1 + 1; 
            elseif A(j, i) == 1 && class == 0
                numOnes0 = numOnes0 + 1;
            end
        end
        
        %probability = numOnes1 / N;
        numZeros1 = N - numOnes1;
        numZeros0 = N - numOnes0;
        
        %add all of the likelihood equations together
        syms p1
        f1 = log((p1^numOnes1)) + log((1-p1)^numZeros1);
        runningEq1 = runningEq1 + f1;
        
     
        syms p0
        f0 = log((p0^numOnes0)) + log((1-p0)^numZeros0);
        runningEq0 = runningEq0 + f0;
        
    end
    
    
    %maximize class 1
    eq1 = diff(runningEq1) == 0;
    ans1 = vpa(solve(eq1));
    
    %maximize class 0
    eq0 = diff(runningEq0) == 0;
    ans0 = vpa(solve(eq0));
    
    output(1) = ans1;
    output(2) = ans0;
    
    %TODO: change calculation to logs?!
    
    %loop through each prior
    numExamples = size(B, 1);
    errorRates = [];
    bestPrior = 1;
    prevErrorRate = 1;
    for k = 1 : size(PC1, 2)
        prior = k;
        
        %CLASSIFY VALIDATION DATA
        classes = B(:, end);
        numErrors = 0;
        for i = 1 : size(B, 1) %loop through each row
            runningProb1 = 1;
            runningProb0 = 1;
            for j = 1 : size(B, 2) - 1 %loop through each column
                if B(i, j) == 1
                    prob1 = ans1^1;
                    prob0 = ans0^1;
                elseif B(i, j) == 0
                    prob1 = (1-ans1)^1;
                    prob0 = (1-ans0)^1;
                end
                runningProb1 = runningProb1 * prob1;
                runningProb0 = runningProb0 * prob0;
            end
        
            %calculate posteriors
            posterior1 = runningProb1 * PC1(1, k);
            posterior0 = runningProb0 * PC0(1, k);
        
            %classify
            if posterior1 > posterior0
                %disp('class1');
                class = classes(i, 1);
                if class ~= 1
                    numErrors = numErrors + 1; 
                end
            elseif posterior0 > posterior1
                %disp('class0');
                class = classes(i, 1);
                if class ~= 0
                    numErrors = numErrors + 1; 
                end
            end
        end
        %calculate the error rate and add to a vector for display
        errorRate = numErrors / numExamples;
        errorRates(end+1, 1)= errorRate;
        
        %look to see if this prior is better than the previous one
        if errorRate < prevErrorRate
           bestPrior = PC1(1, k); 
        end
        prevErrorRate = errorRate;
    end
    class1Priors = {'P(c1) = .1';'P(c1) = .2';'P(c1) = .3';'P(c1) = .4';
        'P(c1) = .5';'P(c1) = .6';'P(c1) = .7';'P(c1) = .8';'P(c1) = .9'};
    Table = table(errorRates, 'RowNames', class1Priors)
    output(3) = bestPrior;
    output(4) = 1 - bestPrior;
    J = output;
end







