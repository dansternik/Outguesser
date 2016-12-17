%% EE 368 Project - Results Analysis - Dominique Piens & Nathan Staffa

%% Read in responses and predictions from CSVs.
resp = csvread('resp04.csv');
pred = csvread('pred04_final_0.csv');
%predInt = csvread('pred04Int.csv');
predInt = double(pred < 0);

%% Show ROC
[X, Y, T, AUC] = perfcurve(resp, double(pred<0), 1);
figure, plot(X,Y)
xlabel('False positive rate');
ylabel('True positive rate');
title('ROC with a Radial-Basis-Function Support Vector Machine andTwo Blockiness Metrics');


%% Tally number that are positive for negative predicted distances.
a = 0 : -0.001 : min(pred);
di = zeros(length(a), 1);
for i = 1 : length(a)
    C = confusionmat(resp(pred <= a(i)), predInt(pred <= a(i)));
    temp = C ./ numel(pred(pred <= a(i)));
    if numel(temp) < 4
        break;
    end
     di(i) = temp(4);
end
lp = min(find(di == 0))
figure, plot(a(1:(lp-1)), di(1:(lp-1)));
% Extract valid values
a1 = a(1:(lp-1));
di1 = di(1:(lp-1));

% Fit a logistic function to "confidence"
predicted = @(a,xdata) a(1)./(1 + exp(a(2) .* (xdata + a(3)))); 
a0 = [1 1 1];
[ahat,resnorm,residual,exitflag,output,lambda,jacobian] =...
   lsqcurvefit(predicted,a0,a1,di1');

% Plot results of fit.
figure, plot(a1,di1,a1,predicted(ahat,a1))


% Confidence of true negative given dist.
a = 0 : 0.001 : max(pred);
di = zeros(length(a), 1);
for i = 1 : length(a)
    C = confusionmat(resp(pred >= a(i)), predInt(pred >= a(i)));
    temp = C ./ numel(pred(pred >= a(i)));
    if numel(temp) < 4
        break;
    end
     di(i) = temp(1);
end
lp = min(find(di == 0))
figure, plot(a(1:(lp-1)), di(1:(lp-1)));
a2 = a(1:(lp-1));
di2 = di(1:(lp-1)); % - di(1);

% Fit logistic function to "confidence"
predicted2 = @(a,xdata) a(1)./(1 + exp(-a(2) .* (xdata - a(3)))); 
a02 = [2 2 2];
[ahat2,resnorm,residual,exitflag,output,lambda,jacobian] =...
   lsqcurvefit(predicted2,a02,a2,di2');

% Plot fit.
figure, plot(a2,di2,a2,predicted2(ahat2,a2))
zs2 = predInt(predInt == 1);
prs2 = pred(resp == 1);

%% Compute Bootstrap statistics.
% Accuracy
Ax = [.788, .669, .761, .672, .652, .772, .656, .651, .616, .682, .655, .648, .650, .626, .642, ...
    .655, .777, .773, .654, .780, .643, .634, .662, .656, .763, .658, .662, .635, .662, .780, .762, .644, ...
    .657, .781, .647, .666, .647, .636, .654, .766, .790, .657, .771, .795, .787, .652, ...
    .647, .621, .657, .631, .656, .656, .784, .622, .656, .765, .673, .760, .784, .640, ...
    .646, .782, .651, .647, .674, .784, .662, .638, .761, .640, .772, .645, .742, .603, ...
    .642, .652, .744, .661, .644, .668, .660, .708, .766, .634, .627, .655, .654, .645, ...
    .772, .677, .666, .651, .645, .668, .646, .670, .654, .655, .674, .665];
% False negative rate
fn = [.092, .102, .066, .184, .108, .082, .124, .256, .321, .284, .213, .241, .081, ...
    .046, .258, .250, .079, .144, .209, .082, .248, .056, .118, .097, .151, .094, ...
    .178, .273, .184, .099, .087, .238, .204, .081, .110, .206, .175, .270, ...
    .092, .155 .092, .209, .075, .087, .089, .201, .258, .312, .206, .290, ...
    .100, .140, .133, .308, .221, .159, .165, .162, .085, .228, .194, .090, ...
    .230, .241, .124, .089, .190, .237, .152, .079, .142, .071, .060, .352, ...
    .227, .224, .059, .129, .157, .099, .102, .235, .145, .269, .065, .185, ...
    .198, .206, .093, .161, .193, .085, .247, .183, .229, .101, .107, .156, .191, .175];

% False positive rate
 fp = 1 - Ax - fn;
