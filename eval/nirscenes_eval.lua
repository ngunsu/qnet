---------------------------------------------------------------------------------------------------
-- This script evaluates the nirscenes patch dataset from
-- "Learning cross-spectral similarity measures with deep convolutional neural networks", CVPRW16
-- It requires:
-- 1.- nirscenes patches dataset in t7 format
---------------------------------------------------------------------------------------------------
-- Cristhian Aguilera

---------------------------------------------------------------------------------------------------
-- Required libs
---------------------------------------------------------------------------------------------------
require 'torch'
require 'math'
require 'nn'
require 'xlua'
require 'paths'
require 'math'
require 'image'
require '../utils/metrics.lua'
log = require '../utils/log.lua'
moses = require 'moses'
cudnn = require 'cudnn'
cunn = require 'cunn'
cutorch = require 'cutorch'
require '../utils/fpr95.lua'

---------------------------------------------------------------------------------------------------
-- Nirscenes sequences
---------------------------------------------------------------------------------------------------
sequences = {'country', 'field', 'forest', 'indoor', 'mountain', 'oldbuilding', 'street', 'urban', 'water'}

---------------------------------------------------------------------------------------------------
-- Argument parsing
---------------------------------------------------------------------------------------------------
-- Info message
cmd = torch.CmdLine()
cmd:text()
cmd:text('NIRSCENES patches evaluation')
cmd:text()
cmd:text('Options:')

-- Global
cmd:option('-seed', 1, 'Fixed input seed for repeatable experiments')

-- CPU/GPU related options
cmd:option('-batchsize', 256, 'Batchsize')

-- Sequence path
cmd:option('-dataset_path', '../datasets/nirscenes/', 't7 sequences filepath')

-- Network
cmd:option('-net', '../trained_networks/qnet.t7', 'Network model')

cmd:text()

-- Parse
opt = cmd:parse(arg or {})

---------------------------------------------------------------------------------------------------
-- Print options
---------------------------------------------------------------------------------------------------
print(opt)

---------------------------------------------------------------------------------------------------
-- Set global configuration
---------------------------------------------------------------------------------------------------
torch.manualSeed(opt.seed)

---------------------------------------------------------------------------------------------------
-- Load network
---------------------------------------------------------------------------------------------------
log.info('[INFO] Loading network ...')
net = torch.load(opt.net)

cudnn.convert(net, cudnn)
net:cuda()
net:evaluate()
print(net)

---------------------------------------------------------------------------------------------------
-- For each sequence
---------------------------------------------------------------------------------------------------
torch.setdefaulttensortype('torch.FloatTensor')

full_err_95 = {}
for __,s in pairs(sequences) do
    log.info('Processing ' .. s .. '...')

    -- Load sequence
    local seq_path = paths.concat(opt.dataset_path, s .. '.t7')
    local seq = torch.load(seq_path, 'ascii')

    local new_seq_data = torch.Tensor(seq.data:size(1),2,32,32):float()
    for i=1,seq.data:size()[1] do
        local scaled_im_1 = image.scale(seq.data[{ {i},1,{},{} }]:clone(),32,32):float():div(255)
        local scaled_im_2 = image.scale(seq.data[{ {i},2,{},{} }]:clone(),32,32):float():div(255)
        new_seq_data[{ {i},1,{},{} }] = scaled_im_1:add(-scaled_im_1:mean())
        new_seq_data[{ {i},2,{},{} }] = scaled_im_2:add(-scaled_im_2:mean())
    end
    seq.data = new_seq_data:clone()
    collectgarbage()
    collectgarbage()

    -- Number of patches
    local n_patches = seq.labels:size(1)

    -- Prepare for results
    local scores = torch.Tensor(n_patches)

    for i,input in ipairs(seq.data:split(opt.batchsize)) do
        xlua.progress(i, n_patches/opt.batchsize)

        input = input:cuda()

        -- Predict
        score_idx_low = (i-1) * opt.batchsize + 1
        score_idx_high = score_idx_low + input:size(1) -1
        
        local des_1 = net:forward(input[{ {},{1},{},{} }]):clone():float()
        local des_2 = net:forward(input[{ {},{2},{},{} }]):clone():float()
        scores[{ {score_idx_low, score_idx_high} }] = -1*torch.norm(des_1-des_2, 2, 2)
        
        collectgarbage()
        collectgarbage()
    end

    fpr95err = fpr95(seq.labels, scores*-1, false)
    full_err_95[s] = fpr95err
    log.info(s .. 'Error:' .. fpr95err )
    x, y = roc(seq.labels, scores*-1, false)
    collectgarbage()
    collectgarbage()
end

---------------------------------------------------------------------------------------------------
-- Show results
---------------------------------------------------------------------------------------------------
log.info('[RESULT] Error at 95% Recall:')
print(full_err_95)
