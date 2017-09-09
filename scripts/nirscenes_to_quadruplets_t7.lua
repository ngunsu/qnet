---------------------------------------------------------------------------------------------------
-- This script generates gt image patches using the nirscenes dataset. It generates 9 files
-- country.t7 ... water.t7
-- It requires:
-- 1.- Nirscenes dataset (uncompressed): http://ivrl.epfl.ch/supplementary_material/cvpr11/
--    *It is necessary to convert the .tiff images to ppm (Torch doesn't support tiff)
--
-- Lua/Torch libs required and not in the standard torch installation
-- 1.- csvigo (to install) -> luarocks install csvigo
---------------------------------------------------------------------------------------------------
-- Cristhian Aguilera

---------------------------------------------------------------------------------------------------
-- Required libs
---------------------------------------------------------------------------------------------------
torch = require 'torch'
paths = require 'paths'
csvigo = require 'csvigo'
image = require 'image'

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
cmd:text('Nirscenes to t7 dataset')
cmd:text()
cmd:text('Options:')

-- Dataset Options
cmd:option('-dataset_path', '/opt/datasets/nirscenes/', 'Dataset path')
cmd:option('-csv_path', '../others/csv/', 'CSV folder with patches information')

cmd:text()

-- Parse options
opt = cmd:parse(arg or {})

---------------------------------------------------------------------------------------------------
-- Processing dataset
---------------------------------------------------------------------------------------------------
-- Set default tensor to ByteTensor to save storage
torch.setdefaulttensortype('torch.ByteTensor')

-- For each sequence
for __,s in pairs(sequences) do
    print('Processing ' .. s .. '...')

    -- Read csv file
    local s_csv_path = paths.concat(opt.csv_path, s .. '.csv')
    local s_csv = csvigo.load(s_csv_path)

    -- Sequence path
    local s_path = paths.concat(opt.dataset_path, s)

    -- Num of patches in the csv file
    local n_patches = #s_csv.rgb

    -- Prepare torch tensor
    local data = torch.Tensor(n_patches, 2, 32, 32):byte()
    local labels = torch.Tensor(n_patches)

    -- For every patch
    local rgb_image_name = nil
    local nir_image_name = nil
    local last_rgb_image_name = ''
    local last_nir_image_name = ''
    local rgb_image = nil
    local nir_image = nil

    -- Load all cross-spectral pairs
    for i=1, n_patches do
        -- Image names
        rgb_image_name = s_csv.rgb[i]
        nir_image_name = s_csv.nir[i]
        local rgb_path = paths.concat(s_path,rgb_image_name)
        local nir_path = paths.concat(s_path,nir_image_name)
        if rgb_image_name ~= last_rgb_image_name  then
            rgb_image = image.load(rgb_path, 1, 'byte')
            nir_image = image.load(nir_path, 1, 'byte')
            last_rgb_image_name = rgb_image_name
        end
        local patch_type = s_csv.type[i]
        local x = s_csv.rgb_x[i]
        local y = s_csv.rgb_y[i]

        -- Extract patch
        data[{ {i},{1},{},{} }] = image.scale(rgb_image[{ {y-31,y+32},{x-31, x+32} }], 32, 32)
        data[{ {i},{2},{},{} }] = image.scale(nir_image[{ {y-31,y+32},{x-31, x+32} }], 32 ,32)
    end

    -- Set default tensor to ByteTensor to save storage
    torch.setdefaulttensortype('torch.FloatTensor')

    --Create quadruplets
    local quadruplets = torch.Tensor(n_patches/2, 4, 32, 32):byte()
    local shuffle = torch.randperm(n_patches)
    for i=1, n_patches/2 do
        -- RGB true pair
        quadruplets[{ {i},{1},{},{} }] = data[{ {shuffle[i]},{1},{},{} }]
        -- NIR true pair
        quadruplets[{ {i},{2},{},{} }] = data[{ {shuffle[i]},{2},{},{} }]
        -- RGB false pair
        quadruplets[{ {i},{3},{},{} }] = data[{ {shuffle[(n_patches/2)+i]},{1},{},{} }]
        -- NIR false pair
        quadruplets[{ {i},{4},{},{} }] = data[{ {shuffle[(n_patches/2)+i]},{2},{},{} }]
    end

    -- Store data in the current folder
    torch.save(s .. '_quadruplets.t7', quadruplets)
end

