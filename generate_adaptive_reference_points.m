function [ref_points1, ref_points2] = generate_adaptive_reference_points(train_data, num_points)
    % 版本3：使用加权中位数+多次迭代优化生成最精确参考点
    % 适配8个参考点
    
    if nargin < 2
        num_points = 8;
    end
    
    input1 = train_data(:, 1);
    input2 = train_data(:, 2);
    output = train_data(:, 3);
    
    fprintf('\n=== V3超精确参考点生成（8个点，加权中位数+迭代优化）===\n');
    
    % 定义输出类别
    output_classes = [0, 0.25, 0.5, 0.75, 1.0];
    
    % 为每个输出类别计算参考点
    % 由于有8个参考点但只有5个输出类别，我们需要在类别之间插值
    % 策略：在数据密集区域增加更多参考点
    
    ref_points1 = zeros(8, 1);
    ref_points2 = zeros(8, 1);
    
    % 首先为5个主要类别计算参考点
    main_ref_points1 = zeros(5, 1);
    main_ref_points2 = zeros(5, 1);
    
    for i = 1:5
        target_output = output_classes(i);
        tolerance = 0.15;
        mask = abs(output - target_output) < tolerance;
        
        if sum(mask) > 10
            class_data1 = input1(mask);
            class_data2 = input2(mask);
            
            % 计算到目标输出的距离，用作权重
            class_outputs = output(mask);
            distances_to_target = abs(class_outputs - target_output);
            
            % 使用高斯权重：距离越近权重越大
            weights = exp(-distances_to_target.^2 / (2*0.05^2));
            weights = weights / sum(weights);
            
            % 加权中心
            main_ref_points1(i) = sum(class_data1 .* weights);
            main_ref_points2(i) = sum(class_data2 .* weights);
            
            % 去除极端异常值后的二次优化
            dist_to_center1 = abs(class_data1 - main_ref_points1(i));
            dist_to_center2 = abs(class_data2 - main_ref_points2(i));
            
            combined_dist = dist_to_center1 + dist_to_center2;
            threshold_dist = quantile(combined_dist, 0.8);
            refined_mask = combined_dist <= threshold_dist;
            
            if sum(refined_mask) > 5
                refined_weights = weights(refined_mask);
                refined_weights = refined_weights / sum(refined_weights);
                
                main_ref_points1(i) = sum(class_data1(refined_mask) .* refined_weights);
                main_ref_points2(i) = sum(class_data2(refined_mask) .* refined_weights);
            end
            
            fprintf('  类别%.2f: %d有效样本, 输入1=%.4f, 输入2=%.4f\n', ...
                target_output, sum(mask), main_ref_points1(i), main_ref_points2(i));
        else
            % 数据不足时的备用方案
            main_ref_points1(i) = quantile(input1, 1 - target_output);
            main_ref_points2(i) = quantile(input2, 1 - target_output);
            fprintf('  类别%.2f: 样本不足，使用全局分位数\n', target_output);
        end
    end
    
    % 现在扩展到8个参考点
    % 策略：在5个主要点之间插入3个额外点
    fprintf('\n扩展到8个参考点...\n');
    
    % 分配8个点：
    % 点1: 类别0 (索引1)
    % 点2: 类别0-0.25之间
    % 点3: 类别0.25 (索引2)
    % 点4: 类别0.25-0.5之间
    % 点5: 类别0.5 (索引3)
    % 点6: 类别0.5-0.75之间
    % 点7: 类别0.75 (索引4)
    % 点8: 类别1.0 (索引5)
    
    ref_points1(1) = main_ref_points1(1);  % 类别0
    ref_points1(2) = 0.67 * main_ref_points1(1) + 0.33 * main_ref_points1(2);  % 0到0.25之间
    ref_points1(3) = main_ref_points1(2);  % 类别0.25
    ref_points1(4) = 0.5 * main_ref_points1(2) + 0.5 * main_ref_points1(3);  % 0.25到0.5之间
    ref_points1(5) = main_ref_points1(3);  % 类别0.5
    ref_points1(6) = 0.5 * main_ref_points1(3) + 0.5 * main_ref_points1(4);  % 0.5到0.75之间
    ref_points1(7) = main_ref_points1(4);  % 类别0.75
    ref_points1(8) = main_ref_points1(5);  % 类别1.0
    
    ref_points2(1) = main_ref_points2(1);
    ref_points2(2) = 0.67 * main_ref_points2(1) + 0.33 * main_ref_points2(2);
    ref_points2(3) = main_ref_points2(2);
    ref_points2(4) = 0.5 * main_ref_points2(2) + 0.5 * main_ref_points2(3);
    ref_points2(5) = main_ref_points2(3);
    ref_points2(6) = 0.5 * main_ref_points2(3) + 0.5 * main_ref_points2(4);
    ref_points2(7) = main_ref_points2(4);
    ref_points2(8) = main_ref_points2(5);
    
    % 迭代优化：确保参考点之间有最佳分离度
    fprintf('\n开始迭代优化参考点间距...\n');
    
    % 最小间距要求（对于8个点，间距会更小）
    min_spacing1 = 0.06;  % 比5个点时更小
    min_spacing2 = 0.06;
    
    % 最大迭代次数
    max_iterations = 15;
    
    for iter = 1:max_iterations
        adjusted = false;
        
        for i = 1:7
            for j = i+1:8
                dist1 = abs(ref_points1(i) - ref_points1(j));
                dist2 = abs(ref_points2(i) - ref_points2(j));
                
                % 属性1调整
                if dist1 < min_spacing1
                    adjustment = (min_spacing1 - dist1) / 2;
                    
                    if i == 1
                        ref_points1(j) = ref_points1(j) + adjustment * 2.0;
                    elseif j == 8
                        ref_points1(i) = ref_points1(i) - adjustment * 2.0;
                    else
                        ref_points1(i) = ref_points1(i) - adjustment * 1.2;
                        ref_points1(j) = ref_points1(j) + adjustment * 1.2;
                    end
                    adjusted = true;
                end
                
                % 属性2调整
                if dist2 < min_spacing2
                    adjustment = (min_spacing2 - dist2) / 2;
                    
                    if i == 1
                        ref_points2(j) = ref_points2(j) + adjustment * 2.0;
                    elseif j == 8
                        ref_points2(i) = ref_points2(i) - adjustment * 2.0;
                    else
                        ref_points2(i) = ref_points2(i) - adjustment * 1.2;
                        ref_points2(j) = ref_points2(j) + adjustment * 1.2;
                    end
                    adjusted = true;
                end
            end
        end
        
        if ~adjusted
            fprintf('  迭代%d: 已达到最优间距\n', iter);
            break;
        end
        
        if iter == max_iterations
            fprintf('  完成%d次迭代优化\n', max_iterations);
        end
    end
    
    % 边界限制
    ref_points1 = max(0.05, min(0.95, ref_points1));
    ref_points2 = max(0.05, min(0.95, ref_points2));
    
    fprintf('\n最终优化参考点（8个）:\n');
    fprintf('  属性1: [');
    for i = 1:8
        fprintf('%.4f', ref_points1(i));
        if i < 8, fprintf(', '); end
    end
    fprintf(']\n');
    
    fprintf('  属性2: [');
    for i = 1:8
        fprintf('%.4f', ref_points2(i));
        if i < 8, fprintf(', '); end
    end
    fprintf(']\n');
    
    fprintf('  覆盖范围: 类别0到1.0\n');
    
    % 显示相邻点间距离
    fprintf('\n相邻参考点间距离:\n');
    for i = 1:7
        d1 = abs(ref_points1(i) - ref_points1(i+1));
        d2 = abs(ref_points2(i) - ref_points2(i+1));
        fprintf('  点%d -> 点%d: 属性1距离=%.4f, 属性2距离=%.4f\n', ...
            i, i+1, d1, d2);
    end
    
    % 计算总体分离度
    avg_dist1 = mean(abs(diff(ref_points1)));
    avg_dist2 = mean(abs(diff(ref_points2)));
    fprintf('\n平均分离度: 属性1=%.4f, 属性2=%.4f\n', avg_dist1, avg_dist2);
end