--------------------------------------------------------------------------------
-- Libs
--------------------------------------------------------------------------------
require 'torch'
require 'optim'
require 'nn'
require 'cudnn'
require 'cutorch'
require 'cunn'
require 'xlua'
require 'paths'
require '../utils/metrics.lua'
require '../utils/DistanceRatioCriterion.lua'
require '../utils/pnnet_utils.lua'
require '../utils/utils.lua'
local pl_path = require 'pl.path'
local log = require '../utils/log.lua'

--------------------------------------------------------------------------------
-- Parse command line arguments
--------------------------------------------------------------------------------
if not opt then
   cmd = torch.CmdLine()
   cmd:text()
   cmd:text('Phototour-path evaluation')
   cmd:text()
   cmd:text('Options:')

   -- Global
   cmd:option('-seed', 1, 'Fixed input seed for repeatable experiments')
   cmd:option('-save', 'results', 'Folder to store the results')

   -- Dataset
   cmd:option('-dataset_path', '/opt/datasets/phototour/', 'Dataset path')
   cmd:option('-seq', 'liberty', 'Dataset train sequence')
   cmd:option('-val_p', 0.05, 'Percentage of the data used as validation')

   -- Models
   cmd:option('-net', '../external/2ch_yosemite_nn.t7', 'Network model')

   -- Data
   cmd:option('-preprocessing', 'g_mean_std', 'mean|g_min_std')
   cmd:option('-shuffle_quadruplets', 0, 'shuffle_quadruplets')

   -- SGD
   cmd:option('-learning_rate', 0.1, 'SGD learning rate')
   cmd:option('-weight_decay', 1e-4, 'SGD weight decay')
   cmd:option('-momentum', 0.9, 'SGD momentum')
   cmd:option('-learning_rate_decay', 1e-6, 'SGD learning rate decay')

   -- Others
   cmd:option('-batchsize', 128, 'Batchsize')
   cmd:option('-num_triplets', 1280000, 'Num of triplets')
   cmd:option('augmentation', 0 , 'Data augmentation')
   cmd:option('-min_iter', 20, 'Minimum number of iterations')
   cmd:option('-max_iter', 1000, 'Maximyn number of iterations')
   cmd:option('-patience', 30, 'Maximun number of iterations to wait for a better val result')
   cmd:option('-debug', false, 'Debug')

   cmd:text()

   -- Parse
   opt = cmd:parse(arg or {})
end


if opt.debug == true then
    require('mobdebug').start()
end

--------------------------------------------------------------------------------
-- Print options
-------------------------------------------------------------------------------

-- Save options in a file
if pl_path.exists(opt.save) == false then
    pl_path.mkdir(opt.save)
end
table_to_file(paths.concat(opt.save,'train_config.json'), opt)
print(opt)

--------------------------------------------------------------------------------
-- Set global configuration
--------------------------------------------------------------------------------
torch.manualSeed(opt.seed)

-- Load images as bytes to save memory
torch.setdefaulttensortype('torch.FloatTensor')

--------------------------------------------------------------------------------
-- Read data and do normalization
--------------------------------------------------------------------------------
torch.manualSeed(opt.seed)
-- Read train data
local seq_t7 = paths.concat(opt.dataset_path, opt.seq .. '.t7')
local seq_data = torch.load(seq_t7)
seq_data.labels:add(1)

-- Prepare data
seq_data.patches32 = seq_data.patches32:float():div(255)
mean = seq_data.patches32:view(seq_data.patches32:size(1), -1):mean()
std = seq_data.patches32:view(seq_data.patches32:size(1), -1):std()

-- Make space
seq_data.patches64 = torch:Tensor()
collectgarbage()
collectgarbage()

-- Preprocess the data
if opt.preprocessing == 'g_mean_std' then
    seq_data.patches32:add(-mean):div(std)
elseif opt.preprocessing == 'mean' then
    local p = seq_data.patches32:view(seq_data.patches32:size(1), 1, 32*32)
    p:add(-p:mean(3):expandAs(p))
end

-- Generate tripples
num_triplets = opt.num_triplets
training_triplets = generate_triplets(seq_data, num_triplets)

-- Generate quadruplets
num_quadruplets = training_triplets:size(1)/2
print ('Num of quadruplets: ' .. num_quadruplets)

shuffle = torch.randperm(training_triplets:size(1))
quadruplets = torch.Tensor(num_quadruplets,4):int()
for i=1, num_quadruplets do
    quadruplets[i][1] = training_triplets[shuffle[i]][1]
    quadruplets[i][2] = training_triplets[shuffle[i]][3]
    quadruplets[i][3] = training_triplets[shuffle[num_quadruplets+i]][1]
    quadruplets[i][4] = training_triplets[shuffle[num_quadruplets+i]][3]
end

train_quadruplets = torch.floor((quadruplets:size(1)*(1-opt.val_p))/128)*128
val_quadruplets = quadruplets:size(1) - train_quadruplets
log.info('Train quadruplets:' .. train_quadruplets)
log.info('Val quadruplets:' .. val_quadruplets)

--------------------------------------------------------------------------------
-- Load model
--------------------------------------------------------------------------------
log.info('Loading model ...')
net = torch.load(opt.net)
cudnn.convert(net, cudnn)
net:cuda()
-- Set criterion
crit=nn.DistanceRatioCriterion()
crit = crit:cuda()
print(net)

-------------------------------------------------------------------------------
-- Train
---------------------------------------------------------------------------------

-- SGD params
optimState = {
  learningRate = opt.learning_rate,
  weightDecay =  opt.weight_decay,
  momentum = opt.momentum,
  learningRateDecay = opt.learning_rate_decay
}

-- Net parameters
parameters, gradParameters = net:getParameters()

--Create x, y and z for tripplet batch processing
w=torch.zeros(opt.batchsize,1,32,32):cuda()
x=torch.zeros(opt.batchsize,1,32,32):cuda()
y=torch.zeros(opt.batchsize,1,32,32):cuda()
z=torch.zeros(opt.batchsize,1,32,32):cuda()

function train()
    epoch = epoch + 1
    Gerr = 0
    shuffle = torch.randperm(train_quadruplets)
    nbatches = train_quadruplets/opt.batchsize

    for k=1,nbatches-1 do
        xlua.progress(k+1, nbatches)

        s = shuffle[{ {k*opt.batchsize,k*opt.batchsize+opt.batchsize} }]
        for i=1,opt.batchsize do
            w[i] = seq_data.patches32[quadruplets[s[i]][1]]
            x[i] = seq_data.patches32[quadruplets[s[i]][2]]
            y[i] = seq_data.patches32[quadruplets[s[i]][3]]
            z[i] = seq_data.patches32[quadruplets[s[i]][4]]
        end

        local feval = function(f)
        if f ~= parameters then parameters:copy(f) end
            gradParameters:zero()
            inputs = {w,x,y,z}
            local outputs = net:forward(inputs)
            local f = crit:forward(outputs, 1)
            Gerr = Gerr+f
            local df_do = crit:backward(outputs)
            net:backward(inputs, df_do)
            loss_curve_logger:add{['loss_curve']=f}
            return f,gradParameters
        end
        optim.sgd(feval, parameters, optimState)
   end
   print('==> epoch '..epoch)
   print(Gerr/nbatches)
   loss_logger:add{['loss']=Gerr}
   print('')
end

val_data_index = torch.Tensor(val_quadruplets, 3)
val_index = 1
for i=train_quadruplets+1, quadruplets:size(1) do
    if torch.rand(1)[1] > 0.5 then
        val_data_index[val_index][1] = quadruplets[i][1]
        val_data_index[val_index][2] = quadruplets[i][2]
        val_data_index[val_index][3] = 1
    else
        val_data_index[val_index][1] = quadruplets[i][1]
        val_data_index[val_index][2] = quadruplets[i][3]
        val_data_index[val_index][3] = 0
    end
    val_index = val_index + 1
end
log.info('Val_data_index:' .. val_data_index:size(1))
-- Load validation data
function validation()
    -- Prepare Tensor to store prediction scores
    scores = torch.Tensor(val_data_index:size(1)):float()
    labels = torch.Tensor(val_data_index:size(1)):float()
    desc = net:get(1):get(1):clone()
    for i=1, val_data_index:size(1) do
        xlua.progress(i, val_data_index:size(1))
        dl = desc:forward(seq_data.patches32[val_data_index[i][1]]:clone():cuda()):clone():float()
        dr = desc:forward(seq_data.patches32[val_data_index[i][2]]:clone():cuda()):clone():float()
        scores[i] = torch.dist(dl, dr)
        labels[i] = val_data_index[i][3]
    end

    val_error = metrics.error_rate_at_95recall(labels, scores, false)
    log.info('Error at 95% Recall:' .. val_error)
    net_layer = net:get(1):get(1):clone():float()
    cudnn.convert(net_layer, nn)
    net_layer = net_layer:clearState()
    if opt.preprocessing == 'g_mean_std' and val_error < best_val then
        to_save = {}
        to_save.desc = net_layer
        to_save.mean = mean
        to_save.std = std
        torch.save(paths.concat(opt.save, 'best_net.t7'),to_save)
    elseif opt.preprocessing == 'mean' then
        torch.save(paths.concat(opt.save, 'best_net.t7'),to_save)
    end
end

function shuffle_quadruplets()
    log.info('Shuffle quadruplets')
    shuffle_p1 = torch.randperm(quadruplets:size(1))
    shuffle_p2 = torch.randperm(quadruplets:size(1))
    new_quadruplets = torch.Tensor(num_quadruplets,4):int()
    for i=1, num_quadruplets do
        new_quadruplets[i][1] = quadruplets[shuffle_p1[i]][1]
        new_quadruplets[i][2] = quadruplets[shuffle_p1[i]][2]
        new_quadruplets[i][3] = quadruplets[shuffle_p2[i]][3]
        new_quadruplets[i][4] = quadruplets[shuffle_p2[i]][4]
    end
    quadruplets = new_quadruplets
end

--Prepare logger
loss_logger = optim.Logger(paths.concat('results/','loss_logger.log'))
loss_curve_logger = optim.Logger(paths.concat('results/','loss_curve_logger.log'))
accuracy_logger = optim.Logger(paths.concat('results/','accuracy_logger.log'))
epoch = 0
best_val = 10000
patience = 0
while epoch < opt.max_iter do
    train()
    validation()
    accuracy_logger:add{['val']=val_error}

    --Early stop
    if val_error < best_val then
        best_val = val_error
        patience = 0
        log.info('Best val at epoch .. ' .. epoch)
    else
        patience = patience + 1
        log.info('paitence is running out ...' .. patience)
    end

    if patience >= opt.patience then
        log.info('Early stopping')
        break
    end

    if  opt.shuffle_quadruplets == 1 and patience == 10 then
        shuffle_quadruplets()
    end

end
loss_logger:style{['loss']='-'}
loss_curve_logger:style{['loss_curve']='-'}
accuracy_logger:style{['val']='-'}
loss_logger:plot()
loss_curve_logger:plot()
accuracy_logger:plot()
