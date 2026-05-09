function f = fun_gwo_em(x_struct)
%% FUN_GWO_EM  v2
%%
%% v2 改进:
%%   - EM迭代次数: 15 → 25（更充分的收敛）
%%   - 智能初始化 β（见 em_brb_core）使得25次就足够收敛

global L M TrainData

[~, ~, ~, ~, ~, logL_hist] = em_brb_core(x_struct, TrainData, 25, false, 1);

if isempty(logL_hist)
    f = 1e6;
    return;
end

logL = logL_hist(end);
if isnan(logL) || isinf(logL)
    f = 1e6;
else
    f = -logL;    % GWO 最小化 → 取负的对数似然
end
end
