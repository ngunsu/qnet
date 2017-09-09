--------------------------------------------------------------------------------
-- Authors: Cristhian Aguilera & Francisco aguilera
--------------------------------------------------------------------------------

--------------------------------------------------------------------------------
-- Libs
--------------------------------------------------------------------------------
require 'torch'
require 'nn'
require 'cudnn'
require 'cutorch'
require 'cunn'
require 'xlua'
require 'os'
require 'paths'
require 'image'
require '../utils/metrics.lua'
local log = require '../utils/log.lua'
require 'pl.stringx'
moses = require 'moses'

--------------------------------------------------------------------------------
-- Parse command line arguments
--------------------------------------------------------------------------------
if not opt then
   cmd = torch.CmdLine()
   cmd:text()
   cmd:text('LWIR_VIS local features evaluation')
   cmd:text()
   cmd:text('Options:')
   -- Global
   cmd:option('-seed', 1, 'Fixed input seed for repeatable experiments')

   -- Dataset
   cmd:option('-dataset_path', '../datasets/icip2015', 'Dataset path')
   -- Models
   cmd:option('-net', '../trained_networks/qnet.t7', 'Network model')
   cmd:text()
   -- Parse
   opt = cmd:parse(arg or {})
end

--------------------------------------------------------------------------------
-- Print options
--------------------------------------------------------------------------------

print(opt)

--------------------------------------------------------------------------------
-- Preparation for testing
--------------------------------------------------------------------------------
torch.setdefaulttensortype('torch.FloatTensor')

-- Load the data
data = torch.load(paths.concat(opt.dataset_path, 'icip2015eval.t7'))

--------------------------------------------------------------------------------
-- Load model
--------------------------------------------------------------------------------
log.info('Loading model ...')
net = torch.load(opt.net)

cudnn.convert(net, cudnn)
net:cuda()
print(net)
net:evaluate()

--------------------------------------------------------------------------------
--Process dataset
--------------------------------------------------------------------------------
-- Average precision
ap = torch.Tensor(44)

for i =1, 44 do
    log.info('Processing pair # ' .. i)
    local lwir_patches = data[i].lwir_patches
    local rgb_patches = data[i].rgb_patches
    local gt_pairs_lwir_rgb = data[i].gt_pairs_lwir_rgb

    -- Prepare vars for results
    local scores = torch.Tensor(lwir_patches:size(1)):fill(0)
    local lwir_matches = {}

    -- Normalize data
    lwir_patches = lwir_patches:float()
    rgb_patches = rgb_patches:float()
    local size_patch=64
    lwir_patches:div(255)
    rgb_patches:div(255)
    for j=1, lwir_patches:size(1) do
        lwir_patches[j]:add(-lwir_patches[j]:mean())
    end
    for j=1, rgb_patches:size(1) do
        rgb_patches[j]:add(-rgb_patches[j]:mean())
    end

    local lwir_patches32 = torch.Tensor(lwir_patches:size(1),1,32,32)
    local rgb_patches32 = torch.Tensor(rgb_patches:size(1),1,32,32)
    for z=1, lwir_patches:size(1) do
        lwir_patches32[{ {z},{1},{},{} }]= image.scale(lwir_patches[{ {z},1,{},{} }]:clone(),32,32)
        lwir_patches32[{ {z},{1},{},{} }]:add(-lwir_patches32[{ {z},{1},{},{} }]:mean())
    end
    for z=1, rgb_patches:size(1) do
        rgb_patches32[{ {z},{1},{},{} }]= image.scale(rgb_patches[{ {z},1,{},{} }]:clone(),32,32)
        rgb_patches32[{ {z},{1},{},{} }]:add(-rgb_patches32[{ {z},{1},{},{} }]:mean())
    end

    lwir_patches32 = lwir_patches32:cuda()
    rgb_patches32 = rgb_patches32:cuda()

    des_1 = net:forward(lwir_patches32):clone():float()
    des_2 = net:forward(rgb_patches32):clone():float()


    -- For each lwir patch
    local tmp_patches = torch.Tensor(rgb_patches:size(1),2,size_patch,size_patch):float()
    tmp_patches[{ {},{2},{},{} }] = rgb_patches:float():clone()

    for j =1, lwir_patches:size(1) do
        xlua.progress(j, lwir_patches:size(1))

        local r_des_1 = torch.repeatTensor(des_1[j], des_2:size(1),1)
        local dif = r_des_1-des_2
        local norm2 = torch.norm(dif, 2, 2)
        local min_val, min_idx = norm2:min(1)
        scores[j] = min_val[1]
        lwir_matches[j] = min_idx[1][1]
    end

    -- Compare with gt
    local labels = torch.Tensor(#lwir_matches):fill(0)
    for j=1,#lwir_matches do
        if gt_pairs_lwir_rgb[j] ~= nil then
            if gt_pairs_lwir_rgb[j] == lwir_matches[j] then
                labels[j] = 1
            end
        end
    end

    descending = false

    ap[i] = metrics.average_precision{labels=labels, scores=scores, descending=descending}
    log.info('AP ' .. i .. ': ' .. ap[i])
end
log.info('MAP: ' .. ap:mean())
