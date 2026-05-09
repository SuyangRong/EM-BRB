% function metrics = evaluate_metrics_5class(y_true, y_pred, thresholds)
% %% EVALUATE_METRICS_5CLASS - Enhanced evaluation metrics for 5-class risk assessment
% % Input:
% %   y_true: Ground truth values (continuous)
% %   y_pred: Predicted values (continuous)
% %   thresholds: Risk level thresholds for 5 classes [low-medium1, medium1-medium2, medium2-medium3, medium3-high]
% %
% % Output:
% %   metrics: Structure containing all evaluation metrics
% 
%     %% Handle optional thresholds with defaults for 5 classes
%     if nargin < 3
%         % Use adaptive thresholds for 5 classes
%         thresholds = quantile(y_true, [0.2, 0.4, 0.6, 0.8]);
%         fprintf('Using adaptive thresholds for 5 classes: [%.3f, %.3f, %.3f, %.3f]\n', ...
%             thresholds(1), thresholds(2), thresholds(3), thresholds(4));
%     end
%     
%     % Ensure column vectors
%     y_true = y_true(:);
%     y_pred = y_pred(:);
%     
%     %% Data validation
%     if length(y_true) ~= length(y_pred)
%         error('y_true and y_pred must have same length');
%     end
%     
%     % Remove any NaN values
%     valid_indices = ~isnan(y_true) & ~isnan(y_pred);
%     y_true = y_true(valid_indices);
%     y_pred = y_pred(valid_indices);
%     
%     if isempty(y_true) || isempty(y_pred)
%         error('No valid data after removing NaN values');
%     end
%     
%     %% 1. REGRESSION METRICS (Risk Score Prediction)
%     metrics.regression = struct();
%     
%     % Mean Absolute Error
%     metrics.regression.MAE = mean(abs(y_pred - y_true));
%     
%     % Root Mean Square Error
%     metrics.regression.RMSE = sqrt(mean((y_pred - y_true).^2));
%     
%     % Spearman Correlation
%     [metrics.regression.Spearman_rho, metrics.regression.Spearman_p] = ...
%         corr(y_true, y_pred, 'Type', 'Spearman');
%     
%     % Pearson Correlation
%     [metrics.regression.Pearson_r, metrics.regression.Pearson_p] = ...
%         corr(y_true, y_pred, 'Type', 'Pearson');
%     
%     % R-squared
%     ss_res = sum((y_true - y_pred).^2);
%     ss_tot = sum((y_true - mean(y_true)).^2);
%     if ss_tot > 0
%         metrics.regression.R2 = 1 - (ss_res / ss_tot);
%     else
%         metrics.regression.R2 = 0;
%     end
%     
%     % Mean Squared Error (MSE)
%     metrics.regression.MSE = mean((y_pred - y_true).^2);
%     
%     % Continuous Accuracy with adaptive threshold
%     error_std = std(y_pred - y_true);
%     acc_threshold = min(0.2, max(0.1, error_std * 2)); % Adaptive threshold
%     
%     diff_abs = abs(y_pred - y_true);
%     metrics.regression.ContinuousAccuracy = sum(diff_abs < acc_threshold) / length(y_true);
%     metrics.regression.AccuracyThreshold = acc_threshold;
%     
%     % Additional accuracy metrics
%     metrics.regression.PerfectPredictions = sum(diff_abs == 0) / length(y_true);
%     metrics.regression.Within10Percent = sum(diff_abs < 0.1) / length(y_true);
%     metrics.regression.Within20Percent = sum(diff_abs < 0.2) / length(y_true);
%     
%     %% 2. CONVERT TO 5 RISK CLASSES
%     % Define risk levels
%     risk_levels = {'0', '0.25', '0.5', '0.75', '1'};
%     class_labels = {'Very Low', 'Low', 'Medium', 'High', 'Very High'};
%     n_classes = length(risk_levels);
%     
%     % Discretize true and predicted values
%     y_true_class = discretize_risk_5class(y_true, thresholds);
%     y_pred_class = discretize_risk_5class(y_pred, thresholds);
%     
%     % Display class distribution
%     fprintf('\n5-Class Distribution:\n');
%     for i = 1:n_classes
%         true_count = sum(y_true_class == i);
%         pred_count = sum(y_pred_class == i);
%         fprintf('  %s (Risk=%s): True=%d (%.1f%%), Pred=%d (%.1f%%)\n', ...
%             class_labels{i}, risk_levels{i}, true_count, true_count/length(y_true_class)*100, ...
%             pred_count, pred_count/length(y_pred_class)*100);
%     end
%     
%     %% 3. 5-CLASS CLASSIFICATION METRICS
%     metrics.classification = struct();
%     
%     % Confusion Matrix
%     y_true_class = max(1, min(n_classes, y_true_class));
%     y_pred_class = max(1, min(n_classes, y_pred_class));
%     
%     metrics.classification.ConfusionMatrix = confusionmat(y_true_class, y_pred_class, ...
%         'Order', 1:n_classes);
%     
%     % Accuracy
%     metrics.classification.Accuracy = sum(diag(metrics.classification.ConfusionMatrix)) / ...
%         sum(metrics.classification.ConfusionMatrix(:));
%     
%     % Precision, Recall, F1 for each class
%     precision_per_class = zeros(n_classes, 1);
%     recall_per_class = zeros(n_classes, 1);
%     f1_per_class = zeros(n_classes, 1);
%     support_per_class = zeros(n_classes, 1);
%     
%     for i = 1:n_classes
%         TP = metrics.classification.ConfusionMatrix(i, i);
%         FP = sum(metrics.classification.ConfusionMatrix(:, i)) - TP;
%         FN = sum(metrics.classification.ConfusionMatrix(i, :)) - TP;
%         support_per_class(i) = sum(metrics.classification.ConfusionMatrix(i, :));
%         
%         % Precision
%         if (TP + FP) > 0
%             precision_per_class(i) = TP / (TP + FP);
%         else
%             precision_per_class(i) = 0;
%         end
%         
%         % Recall
%         if (TP + FN) > 0
%             recall_per_class(i) = TP / (TP + FN);
%         else
%             recall_per_class(i) = 0;
%         end
%         
%         % F1-Score
%         if (precision_per_class(i) + recall_per_class(i)) > 0
%             f1_per_class(i) = 2 * (precision_per_class(i) * recall_per_class(i)) / ...
%                 (precision_per_class(i) + recall_per_class(i));
%         else
%             f1_per_class(i) = 0;
%         end
%     end
%     
%     metrics.classification.Precision = precision_per_class;
%     metrics.classification.Recall = recall_per_class;
%     metrics.classification.F1 = f1_per_class;
%     metrics.classification.Support = support_per_class;
%     
%     % Macro-Averaged F1
%     metrics.classification.MacroF1 = mean(f1_per_class);
%     
%     % Weighted-Averaged F1
%     metrics.classification.WeightedF1 = sum(f1_per_class .* support_per_class) / sum(support_per_class);
%     
%     % Balanced Accuracy
%     metrics.classification.BalancedAccuracy = mean(recall_per_class);
%     
%     % Class-wise metrics table
%     metrics.classification.ClassMetrics = table(...
%         risk_levels', class_labels', precision_per_class, recall_per_class, f1_per_class, support_per_class, ...
%         'VariableNames', {'RiskLevel', 'RiskLabel', 'Precision', 'Recall', 'F1', 'Support'});
%     
%     %% 4. HIGH-RISK FOCUS ANALYSIS (Class 5: Very High Risk)
%     high_risk_class = 5; % Very High risk class index
%     
%     % Check if high-risk class exists
%     if sum(y_true_class == high_risk_class) == 0
%         fprintf('Warning: No very high-risk samples in true labels\n');
%         metrics.high_risk.Precision = 0;
%         metrics.high_risk.Recall = 0;
%         metrics.high_risk.F1 = 0;
%         metrics.high_risk.Specificity = 1;
%     else
%         % High-risk metrics
%         TP_high = metrics.classification.ConfusionMatrix(high_risk_class, high_risk_class);
%         FP_high = sum(metrics.classification.ConfusionMatrix(:, high_risk_class)) - TP_high;
%         FN_high = sum(metrics.classification.ConfusionMatrix(high_risk_class, :)) - TP_high;
%         TN_high = sum(metrics.classification.ConfusionMatrix(:)) - ...
%             (TP_high + FP_high + FN_high);
%         
%         % Precision for high-risk class
%         if (TP_high + FP_high) > 0
%             metrics.high_risk.Precision = TP_high / (TP_high + FP_high);
%         else
%             metrics.high_risk.Precision = 0;
%         end
%         
%         % Recall for high-risk class
%         if (TP_high + FN_high) > 0
%             metrics.high_risk.Recall = TP_high / (TP_high + FN_high);
%         else
%             metrics.high_risk.Recall = 0;
%         end
%         
%         % F1 for high-risk class
%         if (metrics.high_risk.Precision + metrics.high_risk.Recall) > 0
%             metrics.high_risk.F1 = 2 * (metrics.high_risk.Precision * metrics.high_risk.Recall) / ...
%                 (metrics.high_risk.Precision + metrics.high_risk.Recall);
%         else
%             metrics.high_risk.F1 = 0;
%         end
%         
%         % Specificity for high-risk class
%         if (TN_high + FP_high) > 0
%             metrics.high_risk.Specificity = TN_high / (TN_high + FP_high);
%         else
%             metrics.high_risk.Specificity = 0;
%         end
%     end
%     
%     %% 5. ADDITIONAL METRICS
%     % Matthews Correlation Coefficient
%     metrics.classification.MCC = matthews_correlation_5class(metrics.classification.ConfusionMatrix);
%     
%     % Cohen's Kappa
%     metrics.classification.Kappa = cohens_kappa(metrics.classification.ConfusionMatrix);
%     
%     % Hamming Loss
%     metrics.classification.HammingLoss = 1 - metrics.classification.Accuracy;
%     
%     %% 6. CREATE ENHANCED SUMMARY STRING
%     metrics.summary = sprintf([...
%         '=== 5-CLASS EVALUATION METRICS SUMMARY ===\n', ...
%         'Data Info:\n', ...
%         '  Samples: %d, Classes: %d, Thresholds: [%.3f, %.3f, %.3f, %.3f]\n', ...
%         'Regression (Risk Score):\n', ...
%         '  MAE: %.4f, RMSE: %.4f, Spearman ρ: %.4f (p=%.4f)\n', ...
%         '  R²: %.4f, Continuous Accuracy (|error|<%.3f): %.4f\n', ...
%         'Classification (5 Risk Levels):\n', ...
%         '  Accuracy: %.4f, Macro-F1: %.4f, Balanced Acc: %.4f\n', ...
%         '  MCC: %.4f, Weighted F1: %.4f\n', ...
%         'Very High-Risk (Class 5) Focus:\n', ...
%         '  Precision: %.4f, Recall: %.4f, F1: %.4f\n', ...
%         '  Specificity: %.4f\n'], ...
%         length(y_true), n_classes, thresholds(1), thresholds(2), thresholds(3), thresholds(4), ...
%         metrics.regression.MAE, metrics.regression.RMSE, ...
%         metrics.regression.Spearman_rho, metrics.regression.Spearman_p, ...
%         metrics.regression.R2, metrics.regression.AccuracyThreshold, ...
%         metrics.regression.ContinuousAccuracy, ...
%         metrics.classification.Accuracy, metrics.classification.MacroF1, ...
%         metrics.classification.BalancedAccuracy, ...
%         metrics.classification.MCC, metrics.classification.WeightedF1, ...
%         metrics.high_risk.Precision, metrics.high_risk.Recall, metrics.high_risk.F1, ...
%         metrics.high_risk.Specificity);
% end
% 
% %% Helper function: Discretize to 5 risk classes
% function risk_class = discretize_risk_5class(values, thresholds)
%     risk_class = zeros(size(values));
%     
%     % Ensure thresholds are in correct order
%     thresholds = sort(thresholds);
%     
%     % Class 1: < thresholds(1)
%     risk_class(values < thresholds(1)) = 1;
%     
%     % Class 2: thresholds(1) <= x < thresholds(2)
%     risk_class(values >= thresholds(1) & values < thresholds(2)) = 2;
%     
%     % Class 3: thresholds(2) <= x < thresholds(3)
%     risk_class(values >= thresholds(2) & values < thresholds(3)) = 3;
%     
%     % Class 4: thresholds(3) <= x < thresholds(4)
%     risk_class(values >= thresholds(3) & values < thresholds(4)) = 4;
%     
%     % Class 5: >= thresholds(4)
%     risk_class(values >= thresholds(4)) = 5;
% end
% 
% %% Helper function: Calculate Matthews Correlation Coefficient for 5 classes
% function mcc = matthews_correlation_5class(confusion_mat)
%     % Simplified MCC calculation for multi-class
%     n = sum(confusion_mat(:));
%     if n == 0
%         mcc = 0;
%         return;
%     end
%     
%     cov_xy = 0;
%     
%     for k = 1:size(confusion_mat, 1)
%         for l = 1:size(confusion_mat, 2)
%             for m = 1:size(confusion_mat, 1)
%                 cov_xy = cov_xy + confusion_mat(k,k) * confusion_mat(l,m) - ...
%                     confusion_mat(k,l) * confusion_mat(m,k);
%             end
%         end
%     end
%     
%     denominator = 0;
%     for k = 1:size(confusion_mat, 1)
%         row_sum = sum(confusion_mat(k, :));
%         col_sum = sum(confusion_mat(:, k));
%         denominator = denominator + (row_sum * col_sum);
%     end
%     
%     if denominator == 0
%         mcc = 0;
%     else
%         mcc = cov_xy / sqrt((n^2 - sum((sum(confusion_mat, 2).^2))) * ...
%                             (n^2 - sum((sum(confusion_mat, 1).^2))));
%     end
% end
% 
% %% Helper function for Cohen's Kappa
% function kappa = cohens_kappa(confusion_mat)
%     n = sum(confusion_mat(:));
%     if n == 0
%         kappa = 0;
%         return;
%     end
%     
%     % Observed agreement (diagonal sum)
%     po = sum(diag(confusion_mat)) / n;
%     
%     % Expected agreement by chance
%     row_sums = sum(confusion_mat, 2);
%     col_sums = sum(confusion_mat, 1);
%     pe = sum(row_sums .* col_sums') / n^2;
%     
%     if pe == 1
%         kappa = 0;
%     else
%         kappa = (po - pe) / (1 - pe);
%     end
% end

function metrics = evaluate_metrics_5class(y_true, y_pred, thresholds)
%% EVALUATE_METRICS_5CLASS - Enhanced evaluation metrics for 5-class risk assessment
% Input:
%   y_true: Ground truth values (continuous)
%   y_pred: Predicted values (continuous)
%   thresholds: Risk level thresholds for 5 classes [low-medium1, medium1-medium2, medium2-medium3, medium3-high]
%
% Output:
%   metrics: Structure containing all evaluation metrics

    %% Handle optional thresholds with defaults for 5 classes
    if nargin < 3
        thresholds = [];
    end

    % Ensure column vectors
    y_true = y_true(:);
    y_pred = y_pred(:);

    %% Data validation
    if length(y_true) ~= length(y_pred)
        error('y_true and y_pred must have same length');
    end

    % Remove any NaN values
    valid_indices = ~isnan(y_true) & ~isnan(y_pred);
    y_true = y_true(valid_indices);
    y_pred = y_pred(valid_indices);

    if isempty(y_true) || isempty(y_pred)
        error('No valid data after removing NaN values');
    end

    %% 自动计算阈值（修复空阈值BUG）
    if isempty(thresholds)
        try
            thresholds = quantile(y_true, [0.2, 0.4, 0.6, 0.8]);
        catch
            mini = min(y_true);
            maxi = max(y_true);
            thresholds = linspace(mini, maxi, 6);
            thresholds = thresholds(2:5);
        end
    end

    thresholds = sort(thresholds);
    thresholds(1) = min(thresholds(1), max(y_true)*0.99);
    thresholds(end) = max(thresholds(end), min(y_true)*1.01);

    fprintf('Using adaptive thresholds for 5 classes: [%.3f, %.3f, %.3f, %.3f]\n', ...
        thresholds(1), thresholds(2), thresholds(3), thresholds(4));

    %% 1. REGRESSION METRICS (Risk Score Prediction)
    metrics.regression = struct();

    % Mean Absolute Error
    metrics.regression.MAE = mean(abs(y_pred - y_true));

    % Root Mean Square Error
    metrics.regression.RMSE = sqrt(mean((y_pred - y_true).^2));

    % Spearman Correlation
    try
        [metrics.regression.Spearman_rho, metrics.regression.Spearman_p] = ...
            corr(y_true, y_pred, 'Type', 'Spearman');
    catch
        metrics.regression.Spearman_rho = 0;
        metrics.regression.Spearman_p = 1;
    end

    % Pearson Correlation
    try
        [metrics.regression.Pearson_r, metrics.regression.Pearson_p] = ...
            corr(y_true, y_pred, 'Type', 'Pearson');
    catch
        metrics.regression.Pearson_r = 0;
        metrics.regression.Pearson_p = 1;
    end

    % R-squared
    ss_res = sum((y_true - y_pred).^2);
    ss_tot = sum((y_true - mean(y_true)).^2);
    if ss_tot > 0
        metrics.regression.R2 = 1 - (ss_res / ss_tot);
    else
        metrics.regression.R2 = 0;
    end

    % Mean Squared Error (MSE)
    metrics.regression.MSE = mean((y_pred - y_true).^2);

    % Continuous Accuracy with adaptive threshold
    error_std = std(y_pred - y_true);
    acc_threshold = min(0.2, max(0.1, error_std * 2)); % Adaptive threshold

    diff_abs = abs(y_pred - y_true);
    metrics.regression.ContinuousAccuracy = sum(diff_abs < acc_threshold) / length(y_true);
    metrics.regression.AccuracyThreshold = acc_threshold;

    % Additional accuracy metrics
    metrics.regression.PerfectPredictions = sum(diff_abs == 0) / length(y_true);
    metrics.regression.Within10Percent = sum(diff_abs < 0.1) / length(y_true);
    metrics.regression.Within20Percent = sum(diff_abs < 0.2) / length(y_true);

    %% 2. CONVERT TO 5 RISK CLASSES
    risk_levels = {'0', '0.25', '0.5', '0.75', '1'};
    class_labels = {'Very Low', 'Low', 'Medium', 'High', 'Very High'};
    n_classes = length(risk_levels);

    % Discretize true and predicted values
    y_true_class = discretize_risk_5class(y_true, thresholds);
    y_pred_class = discretize_risk_5class(y_pred, thresholds);

    % Display class distribution
    fprintf('\n5-Class Distribution:\n');
    for i = 1:n_classes
        true_count = sum(y_true_class == i);
        pred_count = sum(y_pred_class == i);
        fprintf('  %s (Risk=%s): True=%d (%.1f%%), Pred=%d (%.1f%%)\n', ...
            class_labels{i}, risk_levels{i}, true_count, true_count/length(y_true_class)*100, ...
            pred_count, pred_count/length(y_pred_class)*100);
    end

    %% 3. 5-CLASS CLASSIFICATION METRICS
    metrics.classification = struct();

    % Confusion Matrix
    try
        y_true_class = max(1, min(n_classes, y_true_class));
        y_pred_class = max(1, min(n_classes, y_pred_class));
        metrics.classification.ConfusionMatrix = confusionmat(y_true_class, y_pred_class, 'Order', 1:n_classes);
    catch
        metrics.classification.ConfusionMatrix = zeros(5,5);
    end

    % Accuracy
    metrics.classification.Accuracy = sum(diag(metrics.classification.ConfusionMatrix)) / ...
        max(sum(metrics.classification.ConfusionMatrix(:)), 1);

    % Precision, Recall, F1 for each class
    precision_per_class = zeros(n_classes, 1);
    recall_per_class = zeros(n_classes, 1);
    f1_per_class = zeros(n_classes, 1);
    support_per_class = zeros(n_classes, 1);

    for i = 1:n_classes
        TP = metrics.classification.ConfusionMatrix(i, i);
        FP = sum(metrics.classification.ConfusionMatrix(:, i)) - TP;
        FN = sum(metrics.classification.ConfusionMatrix(i, :)) - TP;
        support_per_class(i) = sum(metrics.classification.ConfusionMatrix(i, :));

        % Precision
        if (TP + FP) > 0
            precision_per_class(i) = TP / (TP + FP);
        else
            precision_per_class(i) = 0;
        end

        % Recall
        if (TP + FN) > 0
            recall_per_class(i) = TP / (TP + FN);
        else
            recall_per_class(i) = 0;
        end

        % F1-Score
        if (precision_per_class(i) + recall_per_class(i)) > 0
            f1_per_class(i) = 2 * (precision_per_class(i) * recall_per_class(i)) / ...
                (precision_per_class(i) + recall_per_class(i));
        else
            f1_per_class(i) = 0;
        end
    end

    metrics.classification.Precision = precision_per_class;
    metrics.classification.Recall = recall_per_class;
    metrics.classification.F1 = f1_per_class;
    metrics.classification.Support = support_per_class;

    % Macro-Averaged F1
    metrics.classification.MacroF1 = mean(f1_per_class);

    % Weighted-Averaged F1
    metrics.classification.WeightedF1 = sum(f1_per_class .* support_per_class) / max(sum(support_per_class),1);

    % Balanced Accuracy
    metrics.classification.BalancedAccuracy = mean(recall_per_class);

    % Class-wise metrics table
    metrics.classification.ClassMetrics = table(...
        risk_levels', class_labels', precision_per_class, recall_per_class, f1_per_class, support_per_class, ...
        'VariableNames', {'RiskLevel', 'RiskLabel', 'Precision', 'Recall', 'F1', 'Support'});

    %% 4. HIGH-RISK FOCUS ANALYSIS (Class 5: Very High Risk)
    high_risk_class = 5; % Very High risk class index

    % Check if high-risk class exists
    if sum(y_true_class == high_risk_class) == 0
        fprintf('Warning: No very high-risk samples in true labels\n');
        metrics.high_risk.Precision = 0;
        metrics.high_risk.Recall = 0;
        metrics.high_risk.F1 = 0;
        metrics.high_risk.Specificity = 1;
    else
        % High-risk metrics
        TP_high = metrics.classification.ConfusionMatrix(high_risk_class, high_risk_class);
        FP_high = sum(metrics.classification.ConfusionMatrix(:, high_risk_class)) - TP_high;
        FN_high = sum(metrics.classification.ConfusionMatrix(high_risk_class, :)) - TP_high;
        TN_high = sum(metrics.classification.ConfusionMatrix(:)) - ...
            (TP_high + FP_high + FN_high);

        % Precision for high-risk class
        if (TP_high + FP_high) > 0
            metrics.high_risk.Precision = TP_high / (TP_high + FP_high);
        else
            metrics.high_risk.Precision = 0;
        end

        % Recall for high-risk class
        if (TP_high + FN_high) > 0
            metrics.high_risk.Recall = TP_high / (TP_high + FN_high);
        else
            metrics.high_risk.Recall = 0;
        end

        % F1 for high-risk class
        if (metrics.high_risk.Precision + metrics.high_risk.Recall) > 0
            metrics.high_risk.F1 = 2 * (metrics.high_risk.Precision * metrics.high_risk.Recall) / ...
                (metrics.high_risk.Precision + metrics.high_risk.Recall);
        else
            metrics.high_risk.F1 = 0;
        end

        % Specificity for high-risk class
        if (TN_high + FP_high) > 0
            metrics.high_risk.Specificity = TN_high / (TN_high + FP_high);
        else
            metrics.high_risk.Specificity = 0;
        end
    end

    %% 5. ADDITIONAL METRICS
    % Matthews Correlation Coefficient
    try
        metrics.classification.MCC = matthews_correlation_5class(metrics.classification.ConfusionMatrix);
    catch
        metrics.classification.MCC = 0;
    end

    % Cohen's Kappa
    try
        metrics.classification.Kappa = cohens_kappa(metrics.classification.ConfusionMatrix);
    catch
        metrics.classification.Kappa = 0;
    end

    % Hamming Loss
    metrics.classification.HammingLoss = 1 - metrics.classification.Accuracy;

    %% 6. CREATE ENHANCED SUMMARY STRING
    metrics.summary = sprintf([...
        '=== 5-CLASS EVALUATION METRICS SUMMARY ===\n', ...
        'Data Info:\n', ...
        '  Samples: %d, Classes: %d, Thresholds: [%.3f, %.3f, %.3f, %.3f]\n', ...
        'Regression (Risk Score):\n', ...
        '  MAE: %.4f, RMSE: %.4f, Spearman ρ: %.4f (p=%.4f)\n', ...
        '  R²: %.4f, Continuous Accuracy (|error|<%.3f): %.4f\n', ...
        'Classification (5 Risk Levels):\n', ...
        '  Accuracy: %.4f, Macro-F1: %.4f, Balanced Acc: %.4f\n', ...
        '  MCC: %.4f, Weighted F1: %.4f\n', ...
        'Very High-Risk (Class 5) Focus:\n', ...
        '  Precision: %.4f, Recall: %.4f, F1: %.4f\n', ...
        '  Specificity: %.4f\n'], ...
        length(y_true), n_classes, thresholds(1), thresholds(2), thresholds(3), thresholds(4), ...
        metrics.regression.MAE, metrics.regression.RMSE, ...
        metrics.regression.Spearman_rho, metrics.regression.Spearman_p, ...
        metrics.regression.R2, metrics.regression.AccuracyThreshold, ...
        metrics.regression.ContinuousAccuracy, ...
        metrics.classification.Accuracy, metrics.classification.MacroF1, ...
        metrics.classification.BalancedAccuracy, ...
        metrics.classification.MCC, metrics.classification.WeightedF1, ...
        metrics.high_risk.Precision, metrics.high_risk.Recall, metrics.high_risk.F1, ...
        metrics.high_risk.Specificity);
end

%% Helper function: Discretize to 5 risk classes
function risk_class = discretize_risk_5class(values, thresholds)
    risk_class = ones(size(values)); % 默认=1，防止越界
    t = sort(thresholds);

    if length(t)>=1
        risk_class(values < t(1)) = 1;
    end
    if length(t)>=2
        risk_class(values >= t(1) & values < t(2)) = 2;
    end
    if length(t)>=3
        risk_class(values >= t(2) & values < t(3)) = 3;
    end
    if length(t)>=4
        risk_class(values >= t(3) & values < t(4)) = 4;
        risk_class(values >= t(4)) = 5;
    end
end

%% Helper function: Calculate Matthews Correlation Coefficient for 5 classes
function mcc = matthews_correlation_5class(confusion_mat)
    try
        n = sum(confusion_mat(:));
        if n == 0
            mcc = 0;
            return;
        end

        cov_xy = 0;
        for k = 1:size(confusion_mat, 1)
            for l = 1:size(confusion_mat, 2)
                for m = 1:size(confusion_mat, 1)
                    cov_xy = cov_xy + confusion_mat(k,k)*confusion_mat(l,m) - confusion_mat(k,l)*confusion_mat(m,k);
                end
            end
        end

        row2 = sum(confusion_mat,2).^2;
        col2 = sum(confusion_mat,1).^2;
        den = sqrt((n^2 - sum(row2)) * (n^2 - sum(col2)));
        mcc = cov_xy / max(den, 1e-6);
    catch
        mcc = 0;
    end
end

%% Helper function for Cohen's Kappa
function kappa = cohens_kappa(confusion_mat)
    try
        n = sum(confusion_mat(:));
        if n == 0
            kappa = 0;
            return;
        end

        po = sum(diag(confusion_mat)) / n;
        row_sums = sum(confusion_mat, 2);
        col_sums = sum(confusion_mat, 1);
        pe = sum(row_sums .* col_sums') / (n^2 + 1e-6);

        if pe == 1
            kappa = 0;
        else
            kappa = (po - pe) / (1 - pe + 1e-6);
        end
    catch
        kappa = 0;
    end
end