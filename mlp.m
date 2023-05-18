% Clear the console and workspace
clear;
clc;

data = readtable("creditcard.csv");
% Remove the time column
data.Time = [];
% Sample the test data to allow to run on low performance machines the
% testing set contains 284,807 transactions which only 492 are positive
sampleRatio = 0.2;
sampleSize = round(height(data) * sampleRatio);
% Assign the sample to the data table
data = datasample(data, sampleSize, 'Replace', false);
% Shuffle the rows by randomly selecting rows without repition with randperm
data = data(randperm(size(data,1)),:);

% Split the data into training and testing sets
% Split 70% of the data table between training and testing
trainRatio = 0.7;
trainIndexes = 1:round(trainRatio*size(data,1));
testIndexes = (trainIndexes(end)+1):size(data,1);

% Select only the rows for each dataset
trainData = data(trainIndexes,:);
testData = data(testIndexes,:);
% Output the data's class information for both sets
trainingFrauds = sum(trainData.Class == 1);
trainingNonFrauds = sum(trainData.Class == 0);
fprintf('The training set contains %d fraudulent transactions and %d non fraudulent transactions.\n', trainingFrauds, trainingNonFrauds);
testingFrauds = sum(testData.Class == 1);
testingNonFrauds = sum(testData.Class == 0);
fprintf('The testing set contains %d fraudulent transactions and %d non fraudulent transactions.\n', testingFrauds, testingNonFrauds);

% Use the train model function to train the model, it will automatically run
% the test on the best model
[bTargets, bPredictions] = trainModel(trainData, testData);
[gTargets,gPredictions] = trainGridModel(trainData, testData);
[dTargets,dPredictions] = trainBaselineModel(trainData, testData);

% Calculate Area under precision recall curve
bayesModelAUPRC = calcAUPRC(bTargets, bPredictions);
gridModelAUPRC = calcAUPRC(gTargets, gPredictions);
baselineModelAUPRC = calcAUPRC(dTargets, dPredictions);
% Print values
fprintf("Bayes: %f \nGrid %f \nDecision Tree %f \n", bayesModelAUPRC, gridModelAUPRC, baselineModelAUPRC);
% Plot bar chart
plotBarChart(bayesModelAUPRC, gridModelAUPRC, baselineModelAUPRC)
plotPRCurve(bTargets, bPredictions, gTargets, gPredictions, dTargets, dPredictions)

function plotPRCurve(targets1, predictions1, targets2, predictions2, targets3, predictions3)
	% Create precision recall curves for each of the models
    [recall1, precision1, ~] = perfcurve(targets1, predictions1, 1);
    [recall2, precision2, ~] = perfcurve(targets2, predictions2, 1);
    [recall3, precision3, ~] = perfcurve(targets3, predictions3, 1);
	% Use trapz to create the AUPRC value from the recall and precision of each model
    AUPRC1 = trapz(recall1, precision1);
    AUPRC2 = trapz(recall2, precision2);
    AUPRC3 = trapz(recall3, precision3);

    % Plot the precision-recall curves
    figure;
    hold on;
    % Plot the three curves
    plot(recall1, precision1, 'b-', 'LineWidth', 2);
    plot(recall2, precision2, 'g-', 'LineWidth', 2);
    plot(recall3, precision3, 'r-', 'LineWidth', 2);
    % Set the top of the graph to be just above the maximum value for
    % clearer viewing
    ylim([0,1.1]);
    % Label, the axis
    xlabel('Recall');
    ylabel('Precision');
    title('Area Under Precision-Recall Curve');
    legend(['Bayes optimisation: ' num2str(AUPRC1)], ['Grid search: ' num2str(AUPRC2)], ['Decision tree: ' num2str(AUPRC3)]);
    grid on;
    hold off;
end

function plotBarChart(bayesModelAUPRC, gridModelAUPRC, baselineModelAUPRC)
    figure;
    hold on
    % Define the axis
    x = categorical(["Bayesian Opt MLP","Grid search MLP","Decision Tree"]);
    y = [bayesModelAUPRC, gridModelAUPRC, baselineModelAUPRC];
    % Plot bar chart
    bar(x,y);
    % Set the top of the chart to be just above the maximum value
    ylim([0,1.1]);
    % Label the axis
    ylabel('AUPRC');
    xlabel('Model');
    title('AUPRC Comparison of Models');
    hold off
end


% Training function, will automatically run the testing function with its
% best training model
function [targets,predictions] = trainModel(data, testData) 
    fprintf("Training MLP-ANN with bayesian optimisation and SMOTE\n"); 
    % Split the data into input features and target
    xTrain = data{:, 1:end-1};
    yTrain = data{:, end};

    % Calculate the coefficient to identify relevant columns
    corrCoeff = corr(xTrain, yTrain);
    % Select features with correlation coefficient greater than threshold,
    % adjust the threshold lower to get more fields from the table
    threshold = 0.2;
    relevantColumns = find(abs(corrCoeff) > threshold); % find relevant features
    
    % Select only the relevant columns
    xTrain = data{:, relevantColumns};
    % Normalise to account for scale
    normalisedInput = normalize(xTrain);
    % Adding gaussian noise
    inputWithNoise = awgn(normalisedInput,10,'measured');

    % Apply SMOTE using the library from Manohar (2023). SMOTE (Synthetic Minority Over-Sampling Technique) (https://www.mathworks.com/matlabcentral/fileexchange/38830-smote-synthetic-minority-over-sampling-technique), MATLAB Central File Exchange.
    % The smote function requires this https://uk.mathworks.com/matlabcentral/fileexchange/12574-nearestneighbour-m
    [finalFeatures, finalMark] = SMOTE(inputWithNoise, yTrain);
    % Print the total number of positive and negative samples in the
    % training for debugging purposes
    trainingFrauds = sum(finalMark == 1);
    trainingNonFrauds = sum(finalMark == 0);
    fprintf('The training set contains %d fraudulent transactions and %d non fraudulent transactions.\n', trainingFrauds, trainingNonFrauds);

    % Re-assigning input and target data to x and t and transpose 
    inputs = finalFeatures';
    targets = finalMark';

    % Optimise training options, learning rate and momentum ranges from
    % below sources
    % https://uk.mathworks.com/help/deeplearning/ug/experiment-using-bayesian-optimization.html
    % https://github.com/annabelkay/credit-card-fraud-detection-ML/blob/master/Optimisation_MLP_Model.m
    vars = [
        % Optimise for learn rate using logarithm
        optimizableVariable('InitialLearnRate', [0.001 0.01], 'Transform', 'log')
        % Optimise momentum
        optimizableVariable('Momentum', [0.1 0.9])
        % Optimise for layer size in range 5 to 20 neurons
        optimizableVariable('Layer1Size',[5 20],'Type','integer')
        optimizableVariable('Layer2Size',[5 20],'Type','integer')
        % Optimise for number of epochs from 100 to 200
        optimizableVariable('MaxEpochs', [100 150], 'Type', 'integer')
        optimizableVariable('MaxFails', [10 30], 'Type', 'integer')
    ];
    % Use the makeObjFcn from the Mathworks deep learning bayesian
    % optimisation experiment
    ObjFcn = makeObjFcn(inputs, targets);
    % Define a bayesoptimisation object using the objective function and
    % the variables defined above, set the max time to 30 minutes this can
    % be adjusted to needs but the model typically concludes in 3 minutes.
    % Set the maxobj to 30, this means the optimisation will conclude after
    % 30 iterations, set the function to not use parallel programming
    BayesObject = bayesopt(ObjFcn,vars,...
    'MaxObj',30,...
    'MaxTime',1800,...
    'IsObjectiveDeterministic',false,...
    'UseParallel',false);

    % Get the data from the best performing neural network
    bestIdx = BayesObject.IndexOfMinimumTrace(end);
    % Use this to get the file its saved too
    file = BayesObject.UserDataTrace(bestIdx);
    fprintf("Best model saved to");
    disp(file(1));
    % Load the file containing the best network
    savedStruct = load(file(1) + ".mat");
    % Use the test model function, passing the saved neural network and
    % passing the relevant columns and testing data
    [targets,predictions] = testModel(savedStruct.net, relevantColumns, testData);
end

% The objective function creates the neural net using the optimised
% parameters from the bayesopt function.
function ObjFcn = makeObjFcn(x, t)
    ObjFcn = @auPRCFun;
    function [auPRC,constraints,file] = auPRCFun(vars)
        % Create the network using fitnnet, it appears to perform much
        % better with this network than other feed forward networks
        net = fitnet([vars.Layer1Size vars.Layer2Size], 'trainlm');
        % Set training parameters to optimise
        net.trainParam.mc = vars.Momentum;
        net.trainParam.lr = vars.InitialLearnRate;
        net.trainParam.epochs = vars.MaxEpochs;
        net.trainParam.max_fail = vars.MaxFails;
        % Set the training ratios and loss function
        net.divideParam.trainRatio = 70/100;
        net.divideParam.valRatio = 15/100;
        net.divideParam.testRatio = 15/100;
        net.performFcn = "mse";
        % Hide the training windows
        net.trainParam.showWindow = false;
        net.trainParam.showCommandLine = false;
        % Train the network
        [net, ~] = train(net, x, t);
        % Make predictions using the trained net
        predictions = net(x);
        auPRC = calcAUPRC(t,predictions);
        % Calculate error
        file = "models/" + sprintf("%d", num2str(auPRC));
        save(file, 'net','auPRC')
        constraints = [];
    end
end

% Training function, will automatically run the testing function with its
% best training model
function [targets, predictions] = trainBaselineModel(data, testData) 
    fprintf("Training decision tree model\n"); 
    % Split the data into input features and target
    testinputs = testData;
    testinputs.Class = [];
    testFraudClasses = testData.Class;
    % Normalise data to get better results to take scale into account
    normalisedData = normalize(testinputs);
    
    % Create decision tree using training data
    tree = fitctree(data, "Class");
    % Make prediction using test data
    [predictions, ~] = tree.predict(normalisedData);
    
    % Create a table of the predictions and the true classes for assessing
    % model performance
    predictionOutput = table(predictions, testFraudClasses);

    % Create confusion matrix using actual fraud cases and predictions
    cm = confusionmat(predictionOutput.testFraudClasses, predictionOutput.predictions);
    % Print data from confusion matrix
    printConfusionMatrix(cm);
    
    % Set the values for returning
    targets = predictionOutput.testFraudClasses;
    predictions = predictionOutput.predictions;
end

function [targets, predictions] = trainGridModel(data, testData)
    fprintf("Training MLP-ANN with grid search and SMOTE\n"); 
    % Split the data into input features and target
    xTrain = data{:, 1:end-1};
    yTrain = data{:, end};

    % Calculate the coefficient to identify relevant columns
    corrCoeff = corr(xTrain, yTrain);
    % Select features with correlation coefficient greater than threshold,
    % adjust the threshold lower to get more fields from the table
    threshold = 0.2;
    relevantColumns = find(abs(corrCoeff) > threshold); % find relevant features
    
    % Select only the relevant columns
    xTrain = data{:, relevantColumns};

    normalisedInput = normalize(xTrain);
    inputWithNoise = awgn(normalisedInput,10,'measured');

    % Apply SMOTE using the library from Manohar (2023). SMOTE (Synthetic Minority Over-Sampling Technique) (https://www.mathworks.com/matlabcentral/fileexchange/38830-smote-synthetic-minority-over-sampling-technique), MATLAB Central File Exchange.
    % The smote function requires this https://uk.mathworks.com/matlabcentral/fileexchange/12574-nearestneighbour-m
    [finalFeatures, finalMark] = SMOTE(inputWithNoise, yTrain);
    % Print the total number of positive and negative samples in the
    % training for debugging purposes
    trainingFrauds = sum(finalMark == 1);
    trainingNonFrauds = sum(finalMark == 0);
    fprintf('The training set contains %d fraudulent transactions and %d non fraudulent transactions.\n', trainingFrauds, trainingNonFrauds);

    % Re-assigning input and target data to x and t and transpose 
    inputs = finalFeatures';
    targets = finalMark';

    % Define a grid of parameters to search over
    hiddenLayerSizes = {5, 10, [5, 5], [5,10], [10,10], [15,15]};
    epochs = [50, 150];
    max_fails = [10, 20, 30];
    trainingFunctions = {'trainrp', 'trainscg', 'trainlm'};

    bestAUPRC = 0;
    bestEpochs = 0;
    bestFails = 0;
    bestLayerSizes = [];
    bestTrainingFunction = '';

    bestNet = [];

    % Perform grid search on the chosen parameters
    for i = 1:numel(hiddenLayerSizes)
        for j = 1:numel(epochs)
            for k = 1:numel(max_fails)
                for n = 1:numel(trainingFunctions)
                    % Create the network
                    net = fitnet(hiddenLayerSizes{i}, trainingFunctions{n});
                    %change the hyperparameters
                    net.trainParam.epochs = epochs(j);
                    net.trainParam.max_fail = max_fails(k);
                    net.trainParam.showWindow = false;
                    net.trainParam.showCommandLine = false;
                    % Set the training ratios and loss function
                    net.divideParam.trainRatio = 70/100;
                    net.divideParam.valRatio = 15/100;
                    net.divideParam.testRatio = 15/100;
                    net.performFcn = 'mse';
                    % Train the network
                    [net, ~] = train(net, inputs, targets);
                    % Testing the network on the train set
                    predictions = net(inputs);

                    % Calculate AUPRC
                    curAuprc = calcAUPRC(targets,predictions);
                
                    % Check if this configuration is the best so far
                    if curAuprc > bestAUPRC
                        % Store the best values in variables for outputting
                        bestAUPRC = curAuprc;
                        bestEpochs = epochs(j);
                        bestFails = max_fails(k);
                        bestLayerSizes = hiddenLayerSizes{i};
                        bestTrainingFunction = trainingFunctions{n};
                        % Define the best network and store it
                        bestNet = net;
                    end
                end
            end
        end
    end

    % Output the best trained network
    fprintf("Optimal Epochs %d\n", bestEpochs);
    fprintf("Optimal Max Fails %d\n", bestFails);
    fprintf("Best training function %s\n", bestTrainingFunction);
    fprintf("Layer sizes used: \n");
    disp(bestLayerSizes);
    
    % Use the test model function, passing the saved neural network and
    % passing the relevant columns and testing data
    [targets,predictions] = testModel(bestNet, relevantColumns, testData);
end

function [targets,predictions] = testModel(net, relevantColumns, testData)
    fprintf("Testing model \n");
    % Assigning input and target data.
    inputs = testData{:,relevantColumns};
    targets = testData{:,end};

    % Re-assigning input and target data to x and t.
    x = inputs';
    t = targets';
    predictions = net(x);
    
    % Adjust threshold to get better results, in this dataset the default
    % of 0.5 seems to be the best to use
    threshold = 0.5;
    thresholdedPredictions = predictions >= threshold;

    % Print the results on the test set
    fprintf("Results on the Test Set: \n");
    
    % Create confusion matrix using predictions
    [~, cm] = confusion(t, thresholdedPredictions);
    
    % Print results to console
    printConfusionMatrix(cm); 
    targets = t;
    % Print end message
    fprintf("Training and testing complete \n");
end

function printConfusionMatrix(cm)
    % Define the true positives etc for ease of reading
    tp = cm(2,2);
    tn = cm(1,1);
    fp = cm(1,2);
    fn = cm(2,1);
    % Calculating the accuracy score using confusion matrix results.
    accuracy = (tp + tn) / (tp + tn + fp + fn);
    precision = tp / (tp + fp);
    recall = tp / (tp + fn);
    specificity = tn / (tn + fp);
    fprintf("Accuracy %f \n", accuracy);
    fprintf("Precision %f \n", precision);
    fprintf("Recall %f \n", recall);
    fprintf("Specificity %f \n", specificity);
end

function returnAuprc = calcAUPRC(targets, predictions)
    % Calculate AUPRC
    [recall, precision, ~] = perfcurve(targets, predictions, 1);
    returnAuprc = trapz(recall, precision);
end
