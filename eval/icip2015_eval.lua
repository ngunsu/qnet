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
   cmd:option('-net_type', 'qnet', 'Net type')
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
if moses.contains({'iccv2015','pnnet'}, opt.net_type) then
    mean = net.mean
    std = net.std
    net = net.desc
end

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
    if opt.net_type == 'cvpr2015' then
        lwir_patches:div(255)
        rgb_patches:div(255)
        for j=1, lwir_patches:size(1) do
            lwir_patches[j]:add(-lwir_patches[j]:mean())
        end
        for j=1, rgb_patches:size(1) do
            rgb_patches[j]:add(-rgb_patches[j]:mean())
        end
    elseif opt.net_type == 'pnnet_mean' or opt.net_type == 'quadruplet' then
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

    elseif opt.net_type == 'cvpr2015siaml2' then
        lwir_patches:div(255)
        rgb_patches:div(255)
        for j=1, lwir_patches:size(1) do
            lwir_patches[j]:add(-lwir_patches[j]:mean())
        end
        for j=1, rgb_patches:size(1) do
            rgb_patches[j]:add(-rgb_patches[j]:mean())
        end
        local tmp_lwir_patches = lwir_patches:clone()
        local tmp_rgb_patches = rgb_patches:clone()

        tmp_lwir_patches = tmp_lwir_patches:cuda()
        tmp_rgb_patches = tmp_rgb_patches:cuda()

        local normalize = nn.Normalize(2):float()
        des_1 = net:get(1):get(1):forward(tmp_lwir_patches):clone():float()
        des_2 = net:get(1):get(1):forward(tmp_rgb_patches):clone():float()
        des_1 = normalize:forward(des_1):clone()
        des_2 = normalize:forward(des_2):clone()
    elseif opt.net_type == 'iccv2015' then
        for z=1, lwir_patches:size(1) do
            lwir_patches[{ {z},{1},{},{} }]:add( -mean ):cdiv( std )
        end
        for z=1, rgb_patches:size(1) do
            rgb_patches[{ {z},{1},{},{} }]:add( -mean ):cdiv( std )
        end
        local tmp_lwir_patches = lwir_patches:clone()
        local tmp_rgb_patches = rgb_patches:clone()

        tmp_lwir_patches = tmp_lwir_patches:cuda()
        tmp_rgb_patches = tmp_rgb_patches:cuda()

        des_1 = net:forward(tmp_lwir_patches):clone():float()
        des_2 = net:forward(tmp_rgb_patches):clone():float()
    elseif opt.net_type == 'qnet' then
        local lwir_patches32 = torch.Tensor(lwir_patches:size(1),1,32,32)
        local rgb_patches32 = torch.Tensor(rgb_patches:size(1),1,32,32)
        for z=1, lwir_patches:size(1) do
            lwir_patches32[{ {z},{1},{},{} }]= image.scale(lwir_patches[{ {z},1,{},{} }]:clone(),32,32):clone():float():div(255):add(-mean):div(std)
        end
        for z=1, rgb_patches:size(1) do
            rgb_patches32[{ {z},{1},{},{} }]= image.scale(rgb_patches[{ {z},1,{},{} }]:clone(),32,32):clone():float():div(255):add(-mean):div(std)
        end

        lwir_patches32 = lwir_patches32:cuda()
        rgb_patches32 = rgb_patches32:cuda()

        des_1 = net:forward(lwir_patches32):clone():float()
        des_2 = net:forward(rgb_patches32):clone():float()
    end

    -- For each lwir patch
    local tmp_patches = torch.Tensor(rgb_patches:size(1),2,size_patch,size_patch):float()
    tmp_patches[{ {},{2},{},{} }] = rgb_patches:float():clone()

    for j =1, lwir_patches:size(1) do
        xlua.progress(j, lwir_patches:size(1))

        if opt.net_type == 'cvpr2015' then
            local r_lwir = torch.repeatTensor(lwir_patches[j], rgb_patches:size(1),1,1)
            tmp_patches[{ {},{1},{},{} }] = r_lwir
            local j_tmp_patches = tmp_patches:clone()

            j_tmp_patches= j_tmp_patches:cuda()

            local out = net:forward(j_tmp_patches):clone():float()
            local max_val, max_idx = out:max(1)
            scores[j] = max_val[1]
            lwir_matches[j] = max_idx[1][1]

        elseif moses.contains({'iccv2015', 'qnet', 'cvpr2015siaml2', 'pnnet_mean', 'quadruplet'}, opt.net_type) then
            local r_des_1 = torch.repeatTensor(des_1[j], des_2:size(1),1)
            local dif = r_des_1-des_2
            local norm2 = torch.norm(dif, 2, 2)
            local min_val, min_idx = norm2:min(1)
            scores[j] = min_val[1]
            lwir_matches[j] = min_idx[1][1]
        end
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

    descending = true
    if moses.contains({'iccv2015', 'qnet', 'cvpr2015siaml2', 'pnnet_mean', 'quadruplet'}, opt.net_type) then
        descending = false
    end

    ap[i] = metrics.average_precision{labels=labels, scores=scores, descending=descending}
    log.info('AP ' .. i .. ': ' .. ap[i])
end
log.info('MAP: ' .. ap:mean())
