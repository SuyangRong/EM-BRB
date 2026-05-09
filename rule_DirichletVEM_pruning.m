function [active_rules, vem_stats, kl_crit] = rule_DirichletVEM_pruning( ...
    x_struct, beta, ref1, ref2, sigma1, sigma2, train_data, alpha_level)
%% RULE_DIRICHLETVEM_PRUNING  v3 (变分EM + 稀疏Dirichlet先验 在线剪枝)
%%
%% 核心机制 (自动相关性判定 ARD):
%%   1. 将剪枝融入学习过程：在内部进行 VEM 迭代，参数(RuleWT, beta)始终保持自洽。
%%   2. 稀疏惩罚：对规则权重施加稀疏惩罚，无用规则的激活概率趋近于0。
%%   3. 后验塌缩：对规则置信度 beta_k 施加先验(alpha_0=1.0)，
%%      当规则不再被数据支撑时，其 beta 会自动塌缩至均匀分布(无信息状态)。
%%   4. 通过计算 beta_k 与均匀分布的 KL散度，自动识别并剔除无用规则。

global L M
if nargin < 8, alpha_level = 0.20; end   

N = 5;  T = size(train_data,1);
Doutput = [0, 0.25, 0.5, 0.75, 1.0];
true_labels = train_data(:,3);
true_class  = zeros(T,1);
for n = 1:T
    [~, true_class(n)] = min(abs(true_labels(n) - Doutput));
end

% 提取当前参数
RuleWT      = max(0.01, x_struct(1:L));
AttributeWT = max(0.05, min(0.95, x_struct(L+1:L+M)));
AttributeWT = AttributeWT / sum(AttributeWT);
thr1 = max(0.05, min(0.65, x_struct(end-1)));
thr2 = max(0.05, min(0.65, x_struct(end)));

fprintf('\n[规则在线稀疏学习 (VEM + Dirichlet Prior)]\n');

%% --- 变分 EM (VEM) 迭代过程 ---
max_vem_iter = 15;
eta = 0.5;       % RuleWT的稀疏Dirichlet先验 ( <1 促进稀疏，让无用规则权重趋于0 )
alpha_0 = 1.0;   % beta的先验 ( =1.0 使得没有数据支撑的规则自动塌缩为均匀分布 )

for iter = 1:max_vem_iter
    % 1. E-Step: 计算当前的规则激活度 (保持接口依赖不变)
    AW = compute_activations_em(train_data, ref1, ref2, sigma1, sigma2, ...
        thr1, thr2, RuleWT, AttributeWT);
    
    row_sum = sum(AW, 2);
    zero_rows = row_sum < 1e-8;
    if any(zero_rows)
        AW(zero_rows, :) = 1 / max(1, L);
        row_sum(zero_rows) = 1;
    end
    AW_norm = AW ./ row_sum;  % Size: [T, L]
    
    % 2. M-Step: 更新规则置信度 beta
    for k = 1:L
        C_k = zeros(1, N); % 规则k在各个类别的激活证据
        for c = 1:N
            idx = (true_class == c);
            C_k(c) = sum(AW_norm(idx, k));
        end
        % 施加 Dirichlet 先验：若 C_k 全为0，则 alpha_hat全为1 -> beta塌缩为均分
        alpha_hat = alpha_0 + C_k;
        beta(k, :) = alpha_hat / sum(alpha_hat);
    end
    
    % 3. M-Step: 更新规则权重 RuleWT (引入自动相关性判定 ARD)
    E_k = sum(AW_norm, 1)'; % 规则的总激活质量 Size: [L, 1]
    % 稀疏更新: 扣除 (1 - eta) 的惩罚项
    RuleWT = max(1e-5, E_k - (1 - eta));
    RuleWT = RuleWT / sum(RuleWT);
end

%% --- 剪枝评估 (计算 KL散度 与 信息量) ---
% 统计量矩阵: [KL散度(信息增益), 规则权重质量, 是否保留]
vem_stats = zeros(L, 3);
kl_div    = zeros(L, 1);

for k = 1:L
    b = beta(k, :);
    % 计算 beta_k 与 均匀分布 的 KL 散度: sum( p * log(p / q) ) 其中 q = 1/N
    % 如果 beta 塌缩为均匀分布，KL 会接近 0
    kl_div(k) = sum(b .* log2(b .* N + 1e-10)); 
    
    % 结合 KL散度 和 收敛后的规则权重 作为最终显著性指标
    info_score = kl_div(k) * (RuleWT(k) * L); 
    vem_stats(k, 1) = kl_div(k);
    vem_stats(k, 2) = RuleWT(k);
end

%% 确定剪枝阈值
% 最大理论 KL 散度为 log2(5) ≈ 2.32
% 我们将 alpha_level 转换为一个动态的 KL 散度/信息阈值 (替代原有的 chi2_crit)
kl_crit = 0.05 + 0.5 * (1 - alpha_level); % 越宽松(alpha大) 阈值越低

% 判定规则是否活跃：KL散度大于阈值 且 规则权重未被压榨到极小值
active_rules = (vem_stats(:,1) > kl_crit) & (vem_stats(:,2) > 1e-4);
vem_stats(:, 3) = double(active_rules);

%% --- 防止过度剪枝：至少保留 30% 的规则 ---
min_rules = max(10, round(L * 0.45));
if sum(active_rules) < min_rules
    % 根据 KL 散度 (信息量) 降序排序
    [~, sorted_k] = sort(vem_stats(:,1), 'descend');
    active_rules = false(L,1);
    active_rules(sorted_k(1:min_rules)) = true;
    vem_stats(:, 3) = double(active_rules);
    fprintf('  [保护] VEM收敛过度稀疏，强制保留前%d条(KL散度最大)规则\n', min_rules);
end

n_act = sum(active_rules);
fprintf('  VEM 剪枝结果: %d/%d 规则保留 (%.1f%%)\n', n_act, L, 100*n_act/L);
fprintf('  KL散度(信息)范围: [%.3f, %.3f], 阈值=%.3f\n', ...
    min(vem_stats(:,1)), max(vem_stats(:,1)), kl_crit);

end