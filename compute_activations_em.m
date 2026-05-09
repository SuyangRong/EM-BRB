function ActivationW = compute_activations_em(data, ref1, ref2, sigma1, sigma2, thr1, thr2, RuleWT, AttributeWT)
%% COMPUTE_ACTIVATIONS_EM  v2
%%
%% v2 改进：
%%   - 激活幂次从 1.5 降至 1.2（减少稀疏激活问题）
%%   - 加入零激活保护（防止全部规则权重为0）
%%   - sigma 下界保护更严格

global L

N_grid  = 8;
T       = size(data,1);
act_pow = 1.2;              % v1=1.5，降低避免过度稀疏
att1    = AttributeWT(1);
att2    = AttributeWT(2);

x1 = data(:,1);
x2 = data(:,2);

AM = zeros(T, L);

for i = 1:N_grid
    s1 = max(0.020, sigma1(i));
    d1 = abs(x1 - ref1(i));
    m1 = exp(-(d1.^2) ./ (2*s1^2));
    In1 = max(0, m1 - thr1).^act_pow;

    for j = 1:N_grid
        k  = (i-1)*N_grid + j;
        s2 = max(0.020, sigma2(j));
        d2 = abs(x2 - ref2(j));
        m2 = exp(-(d2.^2) ./ (2*s2^2));
        In2 = max(0, m2 - thr2).^act_pow;

        oa = sqrt(In1 .* In2);
        valid = oa > 0.001;
        if any(valid)
            AM(valid, k) = RuleWT(k) .* oa(valid) .* ...
                (In1(valid).^att1) .* (In2(valid).^att2);
        end
    end
end

%% 行归一化
AU = sum(AM, 2);
zero_act = AU < 1e-8;

if any(zero_act)
    %% 对零激活样本：不再使用均匀分布（会污染EM）
    %% 改为：找最近的参考点，给予最小但有意义的激活
    for n = find(zero_act)'
        d_sq = zeros(1, L);
        xi1 = data(n,1); xi2 = data(n,2);
        for i2 = 1:N_grid
            for j2 = 1:N_grid
                kk = (i2-1)*N_grid + j2;
                d_sq(kk) = (xi1-ref1(i2))^2 + (xi2-ref2(j2))^2;
            end
        end
        [~, nearest_k] = min(d_sq);
        AM(n, nearest_k) = 1.0;     % 分配到最近规则
    end
    AU = sum(AM, 2);
end

ActivationW = AM ./ AU;
end
