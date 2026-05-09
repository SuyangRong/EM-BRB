% =========================================================================
% 函数：RunPSO
% 功能：粒子群优化算法，接口与 GWO 完全兼容
% 输出：Alpha_pos = 最优参数
%       convergence = 收敛曲线
% =========================================================================
function [Alpha_pos, convergence] = RunPSO(...
    SearchAgents, Max_iter, dim_gwo, ...
    lb_gwo, ub_gwo, L, M, Positions, Fitness)

    % PSO 超参数
    w_max = 0.9;
    w_min = 0.4;
    c1    = 2.0;
    c2    = 2.0;

    % 初始化速度、个体最优、全局最优
    V          = zeros(SearchAgents, dim_gwo);
    pBest_pos  = Positions;
    pBest_score = Fitness;
    [gBest_score, gBest_idx] = min(pBest_score);
    gBest_pos  = pBest_pos(gBest_idx, :);
    convergence = zeros(1, Max_iter);
    t0 = tic;

    % 主循环
    for iter = 1:Max_iter
        w = w_max - (w_max - w_min) * (iter / Max_iter); % 线性递减权重

        for i = 1:SearchAgents
            % PSO 速度更新
            V(i,:) = w * V(i,:) ...
                + c1 * rand(1, dim_gwo) .* (pBest_pos(i,:) - Positions(i,:)) ...
                + c2 * rand(1, dim_gwo) .* (gBest_pos - Positions(i,:));

            % 位置更新
            np = Positions(i,:) + V(i,:);

            % 边界限制
            np = max(lb_gwo, min(ub_gwo, np));

            % 属性权重归一化（保持不变）
            aw = np(L+1:L+M);
            aw = (aw - min(aw)) / (max(aw) - min(aw) + eps);
            np(L+1:L+M) = aw / sum(aw + eps);
            Positions(i,:) = np;

            % 计算适应度
            fv = fun_gwo_em(Positions(i,:));

            % 更新最优
            if fv < pBest_score(i)
                pBest_score(i) = fv;
                pBest_pos(i,:) = Positions(i,:);
            end
            if fv < gBest_score
                gBest_score = fv;
                gBest_pos = Positions(i,:);
            end
        end

        % 收敛曲线
        convergence(iter) = gBest_score;

        % 日志
        if mod(iter, 20) == 0 || iter == 1
            fprintf('  iter %3d/%d | NLL=%.4f | %.1fs\n', ...
                iter, Max_iter, gBest_score, toc(t0));
        end
    end

    % 输出全局最优
    Alpha_pos = gBest_pos;
end