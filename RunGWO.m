% =========================================================================
% 函数：RunGWO.m 风格的内部函数
% 功能：GWO 优化，接口清晰，主程序完全解耦
% =========================================================================
function [Alpha_pos, convergence] = RunGWO(...
    SearchAgents, Max_iter, dim_gwo, ...
    lb_gwo, ub_gwo, L, M, Positions, Fitness)

    % 初始化
    a_vals = 2 - (1:Max_iter)*(2/Max_iter);
    convergence = zeros(1, Max_iter);
    
    [~, si] = sort(Fitness);
    Alpha_pos = Positions(si(1), :); Alpha_score = Fitness(si(1));
    Beta_pos  = Positions(si(2), :); Beta_score  = Fitness(si(2));
    Delta_pos = Positions(si(3), :); Delta_score = Fitness(si(3));
    
    t0 = tic;

    for iter = 1:Max_iter
        a = a_vals(iter);
        for i = 1:SearchAgents
            % 随机数
            r1 = rand(1, dim_gwo);
            r2 = rand(1, dim_gwo);
            
            % GWO 位置更新
            A1 = 2*a*r1 - a; C1 = 2*r2;
            X1 = Alpha_pos - A1 .* abs(C1.*Alpha_pos - Positions(i,:));
            
            A2 = 2*a*r1 - a; C2 = 2*r2;
            X2 = Beta_pos - A2 .* abs(C2.*Beta_pos - Positions(i,:));
            
            A3 = 2*a*r1 - a; C3 = 2*r2;
            X3 = Delta_pos - A3 .* abs(C3.*Delta_pos - Positions(i,:));
            
            np = (X1 + X2 + X3) / 3;
            np = max(lb_gwo, min(ub_gwo, np));
            
            % 属性权重归一化（最大最小）
            aw = np(L+1:L+M);
            aw = (aw - min(aw)) / (max(aw) - min(aw) + eps);
            np(L+1:L+M) = aw / sum(aw + eps);
            Positions(i,:) = np;
            
            % 适应度
            fv = fun_gwo_em(Positions(i,:));
            
            % 更新狼
            if fv < Alpha_score
                Delta_pos = Beta_pos;
                Beta_pos  = Alpha_pos;
                Alpha_pos = Positions(i,:);
                Alpha_score = fv;
            elseif fv < Beta_score
                Delta_pos = Beta_pos;
                Beta_pos  = Positions(i,:);
                Beta_score = fv;
            elseif fv < Delta_score
                Delta_pos = Positions(i,:);
                Delta_score = fv;
            end
        end
        
        convergence(iter) = Alpha_score;
        if mod(iter,20)==0 || iter==1
            fprintf('  iter %3d/%d | NLL=%.4f | %.1fs\n', ...
                iter, Max_iter, Alpha_score, toc(t0));
        end
    end
end