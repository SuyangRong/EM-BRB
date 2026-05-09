function [y_pred, BeliefOut_all] = em_brb_predict(x_struct, beta, ref1, ref2, sigma1, sigma2, data, active_rules)
%% EM_BRB_PREDICT  v2 ─ 用MLE估计参数做BRB推理

global L M

N = 5;  T = size(data,1);
Doutput = [0, 0.25, 0.5, 0.75, 1.0];

if nargin < 8 || isempty(active_rules)
    active_rules = true(L,1);
end

RuleWT      = max(0.01, x_struct(1:L));
AttributeWT = max(0.05, min(0.95, x_struct(L+1:L+M)));
AttributeWT = AttributeWT / sum(AttributeWT);
thr1 = max(0.05, min(0.65, x_struct(end-1)));
thr2 = max(0.05, min(0.65, x_struct(end)));

%% 计算激活权重
ActivationW = compute_activations_em(data, ref1, ref2, sigma1, sigma2, ...
    thr1, thr2, RuleWT, AttributeWT);

%% 对不显著规则置零并重归一化
ActivationW(:, ~active_rules) = 0;
rs = sum(ActivationW, 2);
zero_act = rs < 1e-8;
if any(zero_act)
    n_act = sum(active_rules);
    ActivationW(zero_act, :) = 0;
    if n_act > 0
        ActivationW(zero_act, active_rules) = 1/n_act;
    else
        ActivationW(zero_act, :) = 1/L;
    end
    rs(zero_act) = 1;
end
ActivationW = ActivationW ./ rs;

%% ER 推理（Yang 2006 标准析取公式，不做任何修改）
beta_act = beta;
beta_act(~active_rules, :) = 0;

y_pred        = zeros(T,1);
BeliefOut_all = zeros(T,N);

for n = 1:T
    AW   = ActivationW(n,:);
    Sum1 = sum(beta_act, 2)';

    temp1 = ones(1,N);
    for jj = 1:N
        for k = 1:L
            if active_rules(k)
                Belief1 = AW(k)*beta_act(k,jj) + 1 - AW(k)*Sum1(k);
                temp1(jj) = temp1(jj) * Belief1;
            end
        end
    end

    temp3 = 1;
    for k = 1:L
        if active_rules(k)
            temp3 = temp3 * (1 - AW(k)*Sum1(k));
        end
    end

    denom = sum(temp1) - (N-1)*temp3;
    Value = 1;
    if abs(denom) > eps && ~isnan(denom)
        Value = 1 / denom;
    end

    temp4_vals = 1 - AW(active_rules);
    temp4 = prod(temp4_vals);

    BO = zeros(1,N);
    for jj = 1:N
        num2   = Value*(temp1(jj)-temp3);
        denom2 = 1 - Value*temp4;
        if abs(denom2) < eps || isnan(denom2)
            BO(jj) = 1/N;
        else
            BO(jj) = max(0, min(1, num2/denom2));
        end
    end
    s = sum(BO);
    if s > eps, BO = BO/s; else, BO = ones(1,N)/N; end

    BeliefOut_all(n,:) = BO;
    y_pred(n) = Doutput * BO';
end
end
