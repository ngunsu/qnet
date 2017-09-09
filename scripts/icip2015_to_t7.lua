--------------------------------------------------------------------------------
-- Libs
--------------------------------------------------------------------------------
require 'torch'
require 'xlua'
require 'os'
require 'paths'
require 'image'
require 'pl.stringx'

--------------------------------------------------------------------------------
-- Parse command line arguments
--------------------------------------------------------------------------------
if not opt then
   cmd = torch.CmdLine()
   cmd:text()
   cmd:text('ICIP2015 eval dataset to t7')
   cmd:text()
   cmd:text('Options:')
   -- Dataset
   cmd:option('-dataset_path', '/opt/datasets/icip2015/', 'Dataset path')
   cmd:text()
   -- Parse
   opt = cmd:parse(arg or {})
end

--------------------------------------------------------------------------------
-- Print options
--------------------------------------------------------------------------------
print(opt)

--------------------------------------------------------------------------------
-- Useful functions
--------------------------------------------------------------------------------
function file_to_table_kps(filename)
    local kps = {}
    local idx = 1
    print('[INFO] Loading... ' .. filename)
    for line in io.lines(filename) do
        splited_line = stringx.split(line, '\t')
        kps[idx] = {x=tonumber(splited_line[1])+1, y=tonumber(splited_line[2])+1, size=tonumber(splited_line[3])}
        idx = idx + 1
     end
    return kps
end

function file_to_table_gt_pairs(filename)
    local pairs_gt= {}
    print('[INFO] Loading... ' .. filename)
    for line in io.lines(filename) do
        splited_line = stringx.split(line, '\t')
        local lwir_idx=tonumber(splited_line[5])+1
        local rgb_idx=tonumber(splited_line[6])+1
        pairs_gt[lwir_idx] = rgb_idx
    end
    return pairs_gt
end

function load_patches_from_table_kps(kps, im)
    local patches = torch.Tensor(#kps, 1, 64, 64)
    for j=1, #kps do
        if im:nDimension() == 2 then
            local patch = im[{{kps[j].y-31,kps[j].y+32}, {kps[j].x-31,kps[j].x+32}}]:clone()
            patches[{{j},{1},{},{}}] = patch
        else
            local patch = im[{{1},{kps[j].y-31,kps[j].y+32}, {kps[j].x-31,kps[j].x+32}}]:clone()
            patches[{{j},{},{},{}}] = patch
        end
    end
    return patches
end

torch.setdefaulttensortype('torch.FloatTensor')
--------------------------------------------------------------------------------
-- Generate dataset
--------------------------------------------------------------------------------
icip2015 = {}
for i =1, 44 do
    -- Load image pair
    local lwir_im_path = paths.concat(opt.dataset_path, 'lwir', 'lwir' .. i .. '.ppm')
    local rgb_im_path = paths.concat(opt.dataset_path, 'rgb', 'rgb' .. i .. '.ppm')
    print('[INFO] Loading... ' .. lwir_im_path)
    local lwir_im = image.load(lwir_im_path,1, 'byte')
    print('[INFO] Loading... ' .. rgb_im_path)
    local rgb_im = image.load(rgb_im_path,1, 'byte')

    local lwir_kp_path = paths.concat(opt.dataset_path, 'gt', 'gt', 'lwir_kps','lwir' .. i .. '.kps')
    local rgb_kp_path = paths.concat(opt.dataset_path, 'gt', 'gt', 'rgb_kps', 'rgb' .. i .. '.kps')
    local pairs_gt_path = paths.concat(opt.dataset_path, 'gt', 'gt', 'pairs', 'lwir' .. i .. '_' ..'rgb' .. i .. '.gt')

    local lwir_kps = file_to_table_kps(lwir_kp_path)
    local rgb_kps = file_to_table_kps(rgb_kp_path)
    local pairs_gt = file_to_table_gt_pairs(pairs_gt_path)

    -- Load patches
    local lwir_patches = load_patches_from_table_kps(lwir_kps, lwir_im)
    local rgb_patches = load_patches_from_table_kps(rgb_kps, rgb_im)
    icip2015[i] = {}
    icip2015[i].lwir_patches = lwir_patches:clone()
    icip2015[i].rgb_patches = rgb_patches:clone()
    icip2015[i].gt_pairs_lwir_rgb = pairs_gt
end

torch.save('icip2015eval.t7', icip2015)
