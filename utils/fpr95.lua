require 'torch'
require 'math'

roc = function(targets, outputs, order)
    local L,I = torch.sort(outputs, 1, order)
    local labels = targets:index(1,I)
    local fprev = - math.huge
    local FP = 0.0
    local TP = 0.0
    local n = labels:size(1)
    local N = n/2
    local P = n/2
    local j = 1
    local x = torch.zeros(n/2)
    local y = torch.zeros(n/2)
    for i=1,n do
        if L[i] > fprev then
            x:resize(j)
            y:resize(j)
            x[j] = FP/N
            y[j] = TP/P
            j = j + 1
            fprev = L[i]
        end
        if labels[i] > 0 then
            TP = TP + 1
        else
            FP = FP +1
        end
    end
    return x,y
end

fpr95= function(targets, outputs, order)
    local tpr, fpr = roc(targets, outputs, order)
    local _,k = (fpr -0.95):abs():min(1)
    local FPR95 = tpr[k[1]]
    return FPR95 * 100 
end

