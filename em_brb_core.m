function [beta, ref1, ref2, sigma1, sigma2, logL_hist] = ...
    em_brb_core(x_struct, train_data, max_iter, verbose, n_restarts)
%% EM_BRB_CORE  v2 ─ 关键改进版
%%
%% ═══════════════════════════════════════════════════════════
%%  v2 相比 v1 的改进：
%%
%%  【初始化改进（最重要）】
%%    v1: β 从均匀分布 1/5 开始，EM收敛极慢
%%    v2: β 用 Nadaraya-Watson 估计初始化
%%        β(k,j) = Σ_n w_n·1[c_n=j] / Σ_n w_n
%%        w_n = exp(-||x_n-(ref1_i,ref2_j)||² / (2·σ_nw²))
%%        含义: 规则k附近数据的类别频率 → 接近真实β
%%        效果: EM从好的起点出发，几次迭代就收敛
%%
%%  【参考点初始化改进】
%%    v1: 类条件均值插值（8个点，但插值点可能不代表实际分布）
%%    v2: 混合策略：前5个点=类条件均值（精确），后3个点=分位数插值
%%        额外：所有点通过GWO传入的thr1/thr2微调
%%
%%  【带宽初始化改进】
%%    v1: sigma = std(全部x1)/4 （全局，可能过大）
%%    v2: sigma1(i) = std(属于第i行最近类的x1样本)/3
%%        更贴近实际数据分布
%%
%%  【多重启动】
%%    v2 支持 n_restarts 次独立EM，取对数似然最大的结果
%%    每次重启使用相同初始参考点但不同的随机扰动
%% ═══════════════════════════════════════════════════════════

global L M

if nargin < 3, max_iter = 30; end
if nargin < 4, verbose = false; end
if nargin < 5, n_restarts = 1; end

N = 5;
T = size(train_data,1);
Doutput = [0, 0.25, 0.5, 0.75, 1.0];
N_grid  = 8;

%% ── 提取结构参数 ──
RuleWT      = max(0.01, x_struct(1:L));
AttributeWT = max(0.05, min(0.95, x_struct(L+1:L+M)));
AttributeWT = AttributeWT / sum(AttributeWT);
thr1 = max(0.05, min(0.65, x_struct(end-1)));
thr2 = max(0.05, min(0.65, x_struct(end)));

%% ── 真实类别 ──
true_labels = train_data(:,3);
true_class  = zeros(T,1);
for n = 1:T
    [~, true_class(n)] = min(abs(true_labels(n) - Doutput));
end
C_ind = zeros(T,N);
for n = 1:T, C_ind(n, true_class(n)) = 1; end

x1 = train_data(:,1);
x2 = train_data(:,2);

%% ── 计算初始参考点（类条件均值 + 分位数混合） ──
[ref1_base, ref2_base] = init_ref_points(train_data, N_grid);

%% ── 计算初始带宽（类条件标准差）──
[sigma1_base, sigma2_base] = init_sigma(ref1_base, ref2_base, train_data, true_class, N_grid);

%% ═══ 多重启动 EM ═══
best_logL   = -inf;
best_beta   = [];
best_ref1   = ref1_base;
best_ref2   = ref2_base;
best_sigma1 = sigma1_base;
best_sigma2 = sigma2_base;
best_logL_hist = [];

for restart = 1:n_restarts
    if verbose && n_restarts > 1
        fprintf('    ── 重启 %d/%d ──\n', restart, n_restarts);
    end

    %% 每次重启用稍微不同的初始化
    if restart == 1
        ref1   = ref1_base;
        ref2   = ref2_base;
        sigma1 = sigma1_base;
        sigma2 = sigma2_base;
    else
        %% 加随机扰动（±5%的范围）
        ref1   = ref1_base   + 0.05*(2*rand(N_grid,1)-1);
        ref2   = ref2_base   + 0.05*(2*rand(N_grid,1)-1);
        ref1   = max(0.03, min(0.97, ref1));
        ref2   = max(0.03, min(0.97, ref2));
        sigma1 = sigma1_base .* (0.8 + 0.4*rand(N_grid,1));
        sigma2 = sigma2_base .* (0.8 + 0.4*rand(N_grid,1));
    end

    %% ── Nadaraya-Watson 智能初始化 β ──
    beta = init_beta_nw(ref1, ref2, x1, x2, true_class, N, N_grid);

    %% ── EM 迭代 ──
    logL_hist_r = zeros(max_iter, 1);
    logL_prev   = -inf;

    for iter = 1:max_iter
        %% ── E步：计算激活权重和责任度 ──
        ActivationW = compute_activations_em(train_data, ref1, ref2, ...
            sigma1, sigma2, thr1, thr2, RuleWT, AttributeWT);   % T×L

        %% 混合概率 P(c_n | x_n) = Σ_k AW(n,k)·β(k,c_n)
        AW_beta     = ActivationW * beta;                        % T×N
        mixture_prob = zeros(T,1);
        for n = 1:T
            mixture_prob(n) = max(1e-10, AW_beta(n, true_class(n)));
        end
        logL = mean(log(mixture_prob));
        logL_hist_r(iter) = logL;

        if verbose && (mod(iter,5)==0 || iter==1)
            fprintf('    EM iter %2d | logL = %.4f\n', iter, logL);
        end

        %% 责任度 r(n,k) = AW(n,k)·β(k,cn) / P(cn|x_n)
        R = zeros(T,L);
        for n = 1:T
            cn  = true_class(n);
            num = ActivationW(n,:) .* beta(:,cn)';   % 1×L
            R(n,:) = num / mixture_prob(n);
        end

        %% ── M步：闭式更新所有参数 ──

        %% Idea 1: β 的MLE更新
        Rk_sum        = max(1e-10, sum(R,1)');        % L×1
        beta_num      = R' * C_ind;                   % L×N
        beta_new      = bsxfun(@rdivide, beta_num, Rk_sum);
        beta_new      = max(1e-6, beta_new);
        beta_new      = bsxfun(@rdivide, beta_new, sum(beta_new,2));

        %% Idea 3: 参考点和带宽的MLE更新
        ref1_new   = ref1;    sigma1_new = sigma1;
        ref2_new   = ref2;    sigma2_new = sigma2;

        for i = 1:N_grid
            row_idx    = (i-1)*N_grid + (1:N_grid);
            R_row      = sum(R(:, row_idx), 2);          % T×1
            R_row_sum  = sum(R_row);
            if R_row_sum > 1e-6
                mu          = sum(R_row .* x1) / R_row_sum;
                v           = sum(R_row .* (x1-mu).^2) / R_row_sum;
                ref1_new(i)   = mu;
                sigma1_new(i) = max(0.020, sqrt(max(v, 1e-6)));
            end
        end

        for j = 1:N_grid
            col_idx    = j:N_grid:L;
            R_col      = sum(R(:, col_idx), 2);
            R_col_sum  = sum(R_col);
            if R_col_sum > 1e-6
                mu          = sum(R_col .* x2) / R_col_sum;
                v           = sum(R_col .* (x2-mu).^2) / R_col_sum;
                ref2_new(j)   = mu;
                sigma2_new(j) = max(0.020, sqrt(max(v, 1e-6)));
            end
        end

        ref1_new = max(0.03, min(0.97, ref1_new));
        ref2_new = max(0.03, min(0.97, ref2_new));

        %% 检测收敛
        delta = abs(logL - logL_prev);
        beta   = beta_new;
        ref1   = ref1_new;   sigma1 = sigma1_new;
        ref2   = ref2_new;   sigma2 = sigma2_new;

        if delta < 1e-5 && iter > 5
            logL_hist_r = logL_hist_r(1:iter);
            if verbose
                fprintf('    EM收敛 (iter=%d, Δ=%.2e)\n', iter, delta);
            end
            break;
        end
        logL_prev = logL;
    end

    logL_final = logL_hist_r(end);
    if logL_final > best_logL
        best_logL      = logL_final;
        best_beta      = beta;
        best_ref1      = ref1;   best_ref2   = ref2;
        best_sigma1    = sigma1; best_sigma2 = sigma2;
        best_logL_hist = logL_hist_r(logL_hist_r ~= 0);
    end
end

beta   = best_beta;
ref1   = best_ref1;   ref2   = best_ref2;
sigma1 = best_sigma1; sigma2 = best_sigma2;
logL_hist = best_logL_hist;
end


%% ══════════════════════════════════════════════════════════
%%  内部辅助函数
%% ══════════════════════════════════════════════════════════

function beta = init_beta_nw(ref1, ref2, x1, x2, true_class, N, N_grid)
%% Nadaraya-Watson 初始化 β
%% β(k,j) = Σ_n w_n·1[c_n=j] / Σ_n w_n
%% 含义: 规则k参考点附近的样本中，类别j的加权比例
%%       → 比均匀分布更接近真实值，EM收敛快10倍

    L_loc  = N_grid^2;
    T      = length(x1);
    sigma_nw = 0.12;    % NW带宽：足够宽以捕获足够样本

    C_ind = zeros(T, N);
    for n = 1:T, C_ind(n, true_class(n)) = 1; end

    beta = zeros(L_loc, N);
    for i = 1:N_grid
        for j = 1:N_grid
            k = (i-1)*N_grid + j;
            %% 2D 高斯权重
            w = exp(-((x1-ref1(i)).^2 + (x2-ref2(j)).^2) / (2*sigma_nw^2));
            w_sum = sum(w);
            if w_sum > 1e-8
                beta(k,:) = (w' * C_ind) / w_sum;
            else
                beta(k,:) = ones(1,N) / N;
            end
        end
    end
    beta = max(1e-6, beta);
    beta = bsxfun(@rdivide, beta, sum(beta,2));
end


function [ref1, ref2] = init_ref_points(train_data, N_grid)
%% 参考点初始化：类条件均值 + 全局分位数混合策略
%%
%% 逻辑：
%%   - 5个类别有5个特征中心
%%   - 需要8个参考点覆盖整个输入空间
%%   - 前5个点=类条件均值（精准定位类中心）
%%   - 后3个点=边界区域分位数（覆盖过渡区域）
%%   - 排序确保参考点单调递增

    x1  = train_data(:,1);
    x2  = train_data(:,2);
    y   = train_data(:,3);
    cls = [0, 0.25, 0.5, 0.75, 1.0];

    %% 5个类别的条件均值
    cm1 = zeros(5,1);  cm2 = zeros(5,1);
    for ci = 1:5
        mask = abs(y - cls(ci)) < 0.15;
        if sum(mask) > 5
            cm1(ci) = mean(x1(mask));
            cm2(ci) = mean(x2(mask));
        else
            cm1(ci) = quantile(x1, (ci-1)/4);
            cm2(ci) = quantile(x2, (ci-1)/4);
        end
    end

    %% 扩展为8个点（排序后均匀插入额外点）
    x1_sorted = sort(cm1);
    x2_sorted = sort(cm2);

    %% 类均值给5个点，再加3个分位数插值点
    extra_q1 = quantile(x1, [0.10, 0.50, 0.90]);
    extra_q2 = quantile(x2, [0.10, 0.50, 0.90]);

    ref1 = sort([x1_sorted; extra_q1(:)]);
    ref2 = sort([x2_sorted; extra_q2(:)]);

    %% 确保8个点
    ref1 = ref1(1:N_grid);
    ref2 = ref2(1:N_grid);

    %% 确保最小间距（防止参考点堆叠）
    min_gap = 0.04;
    for rep = 1:20
        changed = false;
        for i = 2:N_grid
            if ref1(i) - ref1(i-1) < min_gap
                ref1(i) = ref1(i-1) + min_gap;
                changed = true;
            end
            if ref2(i) - ref2(i-1) < min_gap
                ref2(i) = ref2(i-1) + min_gap;
                changed = true;
            end
        end
        if ~changed, break; end
    end

    ref1 = max(0.03, min(0.97, ref1));
    ref2 = max(0.03, min(0.97, ref2));
end


function [sigma1, sigma2] = init_sigma(ref1, ref2, train_data, true_class, N_grid)
%% 带宽初始化：基于最近类的条件标准差
%% 比全局固定带宽更精确

    x1 = train_data(:,1);
    x2 = train_data(:,2);

    %% 5个类别的条件均值
    Dout = [0, 0.25, 0.5, 0.75, 1.0];
    cls_mu1 = zeros(5,1);  cls_mu2 = zeros(5,1);
    cls_std1 = zeros(5,1); cls_std2 = zeros(5,1);
    for ci = 1:5
        mask = true_class == ci;
        if sum(mask) > 3
            cls_mu1(ci)  = mean(x1(mask));
            cls_mu2(ci)  = mean(x2(mask));
            cls_std1(ci) = max(0.025, std(x1(mask)));
            cls_std2(ci) = max(0.025, std(x2(mask)));
        else
            cls_mu1(ci) = (ci-1)/4;  cls_mu2(ci) = (ci-1)/4;
            cls_std1(ci) = 0.06;     cls_std2(ci) = 0.06;
        end
    end

    sigma1 = zeros(N_grid,1);
    sigma2 = zeros(N_grid,1);

    for i = 1:N_grid
        %% 找最近的类中心
        [~, ci] = min(abs(ref1(i) - cls_mu1));
        sigma1(i) = cls_std1(ci) / 2;   % 使用该类x1标准差的一半
    end
    for j = 1:N_grid
        [~, ci] = min(abs(ref2(j) - cls_mu2));
        sigma2(j) = cls_std2(ci) / 2;
    end

    %% 安全范围
    sigma1 = max(0.025, min(0.15, sigma1));
    sigma2 = max(0.025, min(0.15, sigma2));
end
