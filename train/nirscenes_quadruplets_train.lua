--------------------------------------------------------------------------------
-- Authors: Cristhian Aguilera & Francisco aguilera
--------------------------------------------------------------------------------

--------------------------------------------------------------------------------
-- Libs
--------------------------------------------------------------------------------
require 'torch'
require 'optim'
require 'image'
require 'nn'
require 'cudnn'
require 'cutorch'
require 'cunn'
require 'xlua'
require 'paths'
require 'torchx'
require '../utils/metrics.lua'
require '../utils/DistanceRatioCriterion2.lua'
require '../utils/utils.lua'
local pl_path = require 'pl.path'
local log = require '../utils/log.lua'


--------------------------------------------------------------------------------
-- Parse command line arguments
--------------------------------------------------------------------------------
if not opt then
    cmd = torch.CmdLine()
    cmd:text()
    cmd:text('Nirscenes triplets training')
    cmd:text()
    cmd:text('Options:')

    -- Global
    cmd:option('-seed', 1, 'Fixed input seed for repeatable experiments')
    cmd:option('-save', 'results', 'save folder')

    -- Dataset
    cmd:option('-dataset_path', '../datasets/nirscenes/train', 'Dataset path')
    cmd:option('-seq', 'country', 'Dataset train sequence')
    cmd:option('-val_p',0.05 , 'Percentage of training data used  as validation')

    -- Models
    cmd:option('-net', '../nets/qnet.t7', 'Network model')

    -- SGD
    cmd:option('-learning_rate', 0.1, 'SGD learning rate')
    cmd:option('-weight_decay', 1e-4, 'SGD weight decay')
    cmd:option('-momentum', 0.9, 'SGD momentum')
    cmd:option('-learning_rate_decay', 1e-6, 'SGD learning rate decay')

    -- Others
    cmd:option('-batchsize', 128, 'Batchsize')
    cmd:option('-augmentation', 0, 'Data augmentation')
    cmd:option('-min_iter', 20, 'Minimum number of iterations')
    cmd:option('-max_iter', 1000, 'Maximyn number of iterations')
    cmd:option('-patience', 30, 'Maximun number of iterations to wait for a better val result')
    cmd:option('-debug', 1, 'Debug mode')
    cmd:option('-image_server', 'cvc.crisale.net', 'Image server')

    cmd:text()

    -- Parse
    opt = cmd:parse(arg or {})
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

--------------------------------------------------------------------------------
-- Load dataset
--------------------------------------------------------------------------------
torch.setdefaulttensortype('torch.FloatTensor')

-- Read train data
local seq_t7 = paths.concat(opt.dataset_path, opt.seq .. '_quadruplets.t7')
log.info('Loading ... ' .. seq_t7)
tmp_seq = torch.load(seq_t7)

--------------------------------------------------------------------------------
-- Data augmentation
--------------------------------------------------------------------------------
if opt.augmentation == 0 then
    seq = tmp_seq
else -- Data augmentation
    seq = torch.Tensor(tmp_seq:size(1)*opt.augmentation, 4, 32, 32):float()
    seq_index = 1
    for j=1, opt.augmentation do
        for i=1, tmp_seq:size(1) do
            if j == 1 then -- Original
                seq[{ {seq_index},1,{},{} }] = tmp_seq[{ {i},{1},{},{} }]
                seq[{ {seq_index},2,{},{} }] = tmp_seq[{ {i},{2},{},{} }]
                seq[{ {seq_index},3,{},{} }] = tmp_seq[{ {i},{3},{},{} }]
                seq[{ {seq_index},4,{},{} }] = tmp_seq[{ {i},{4},{},{} }]
            elseif j == 2 then -- vflip
                seq[{ {seq_index},1,{},{} }] = image.vflip(tmp_seq[i][1]:clone())
                seq[{ {seq_index},2,{},{} }] = image.vflip(tmp_seq[i][2]:clone())
                seq[{ {seq_index},3,{},{} }] = image.vflip(tmp_seq[i][3]:clone())
                seq[{ {seq_index},4,{},{} }] = image.vflip(tmp_seq[i][4]:clone())
            elseif j == 3 then -- hflip
                seq[{ {seq_index},1,{},{} }] = image.hflip(tmp_seq[i][1]:clone())
                seq[{ {seq_index},2,{},{} }] = image.hflip(tmp_seq[i][2]:clone())
                seq[{ {seq_index},3,{},{} }] = image.hflip(tmp_seq[i][3]:clone())
                seq[{ {seq_index},4,{},{} }] = image.hflip(tmp_seq[i][4]:clone())
            elseif j == 4 then -- rot 90
                seq[{ {seq_index},1,{},{} }] = image.rotate(tmp_seq[i][{ {1},{},{} }]:clone(), 90)
                seq[{ {seq_index},2,{},{} }] = image.rotate(tmp_seq[i][{ {2},{},{} }]:clone(), 90)
                seq[{ {seq_index},3,{},{} }] = image.rotate(tmp_seq[i][{ {3},{},{} }]:clone(), 90)
                seq[{ {seq_index},4,{},{} }] = image.rotate(tmp_seq[i][{ {4},{},{} }]:clone(), 90)
            elseif j == 5 then -- rot 180
                seq[{ {seq_index},1,{},{} }] = image.rotate(tmp_seq[i][{ {1},{},{} }]:clone(), 180)
                seq[{ {seq_index},2,{},{} }] = image.rotate(tmp_seq[i][{ {2},{},{} }]:clone(), 180)
                seq[{ {seq_index},3,{},{} }] = image.rotate(tmp_seq[i][{ {3},{},{} }]:clone(), 180)
                seq[{ {seq_index},4,{},{} }] = image.rotate(tmp_seq[i][{ {4},{},{} }]:clone(), 180)
            elseif j == 6 then -- rot 270
                seq[{ {seq_index},1,{},{} }] = image.rotate(tmp_seq[i][{ {1},{},{} }]:clone(), 270)
                seq[{ {seq_index},2,{},{} }] = image.rotate(tmp_seq[i][{ {2},{},{} }]:clone(), 270)
                seq[{ {seq_index},3,{},{} }] = image.rotate(tmp_seq[i][{ {3},{},{} }]:clone(), 270)
                seq[{ {seq_index},4,{},{} }] = image.rotate(tmp_seq[i][{ {4},{},{} }]:clone(), 270)
            end
            seq_index = seq_index + 1
        end
    end
end
--------------------------------------------------------------------------------
-- Scale and preprocess the data
--------------------------------------------------------------------------------
seq = seq:float():div(255)

-- Preprocess the dat
for i=1,seq:size(1) do
    local mean_1 = seq[{ {i},1,{},{} }]:mean()
    local mean_2 = seq[{ {i},2,{},{} }]:mean()
    local mean_3 = seq[{ {i},3,{},{} }]:mean()
    local mean_4 = seq[{ {i},4,{},{} }]:mean()
    seq[{ {i},1,{},{} }]:add(-mean_1)
    seq[{ {i},2,{},{} }]:add(-mean_2)
    seq[{ {i},3,{},{} }]:add(-mean_3)
    seq[{ {i},4,{},{} }]:add(-mean_4)
end
train_tripplets = torch.floor((seq:size(1)*(1-opt.val_p))/128)*128
val_tripplets = seq:size(1)-train_tripplets
log.info( 'Train tripplets: ' .. train_tripplets )
log.info( 'Val tripplets: ' .. val_tripplets )

--------------------------------------------------------------------------------
-- Load model
--------------------------------------------------------------------------------
log.info('Loading network ...')
net = torch.load(opt.net)
cudnn.convert(net, cudnn)
net:cuda()
-- Set criterion
crit=nn.DistanceRatioCriterion2()
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
    shuffle = torch.randperm(train_tripplets)
    nbatches = train_tripplets/opt.batchsize

    for k=1,nbatches-1 do
        xlua.progress(k+1, nbatches)

        s = shuffle[{ {k*opt.batchsize,k*opt.batchsize+opt.batchsize} }]
        for i=1,opt.batchsize do
            w[i] = seq[{ {s[i]},{1},{},{} }]:clone():cuda() -- RGB True
            x[i] = seq[{ {s[i]},{2},{},{} }]:clone():cuda() -- RGB False
            y[i] = seq[{ {s[i]},{3},{},{} }]:clone():cuda() -- NIR True
            z[i] = seq[{ {s[i]},{4},{},{} }]:clone():cuda() -- RGB False
        end

        local feval = function(x_param)
            if x_param ~= parameters then
                parameters:copy(x_param)
            end
            gradParameters:zero()
            inputs = {w,x,y,z}
            local outputs = net:forward(inputs)
            local f = crit:forward(outputs, 1)
            Gerr = Gerr+f
            local df_do = crit:backward(outputs)
            net:backward(inputs, df_do)

            -- For logging and online display
            iter = iter or 1
            loss_curve_logger:add{['loss_curve']=f}
            iter = iter + 1

            return f,gradParameters
        end
        optim.sgd(feval, parameters, optimState)
   end
    print('')

    -- Save network
    net_layer = net:get(1):get(1):clone():float()
    cudnn.convert(net_layer, nn)
    net_layer = net_layer:clearState()
end

-------------------------------------------------------------------------------
-- Validation
---------------------------------------------------------------------------------
-- Create pairs
seq_val = torch.Tensor(val_tripplets, 2, 32, 32):float()
seq_val_labels = torch.ones(val_tripplets):float()
for i=1, val_tripplets do
    if torch.rand(1)[1] > 0.5 then
        seq_val[{ {i},{1},{},{} }] =  seq[{ {train_tripplets+i},{1},{},{} }]:clone()
        seq_val[{ {i},{2},{},{} }] =  seq[{ {train_tripplets+i},{2},{},{} }]:clone()
    else
        seq_val[{ {i},{1},{},{} }] =  seq[{ {train_tripplets+i},{1},{},{} }]:clone()
        seq_val[{ {i},{2},{},{} }] =  seq[{ {train_tripplets+i},{4},{},{} }]:clone()
        seq_val_labels[i]=0
    end
end

function validation()
    val_net = net:get(1):get(1):clone()
    scores = torch.Tensor(val_tripplets):float()

    for i, val_input in ipairs(seq_val:split(opt.batchsize)) do
        val_input = val_input:cuda()
        xlua.progress(i, val_tripplets/opt.batchsize)
        local score_idx_low = (i-1) * opt.batchsize +1
        local score_idx_high = score_idx_low + val_input:size(1) - 1
        local des_1 = val_net:forward( val_input[{{},{1},{},{}}] ):clone():float()
        local des_2 = val_net:forward( val_input[{{},{2},{},{}}] ):clone():float()
        scores[{ {score_idx_low,score_idx_high} }] = -1*torch.norm(des_1-des_2, 2, 2)
    end
    val_error = metrics.error_rate_at_95recall(seq_val_labels, scores, true)
    if val_error < best_val then
        log.info('Storing best new net...')
        net_layer = net:get(1):get(1):clone():float()
        cudnn.convert(net_layer, nn)
        net_layer = net_layer:clearState()
        torch.save(paths.concat(opt.save,'best_net.t7'),net_layer:clearState())
    end
    log.info('Error at 95% Recall:' .. val_error)
end

-------------------------------------------------------------------------------
--Logger and run
---------------------------------------------------------------------------------
--Prepare logger
loss_logger = optim.Logger(paths.concat(opt.save,'loss_logger.log'))
loss_curve_logger = optim.Logger(paths.concat(opt.save,'loss_curve_logger.log'))
accuracy_logger = optim.Logger(paths.concat(opt.save,'accuracy_logger.log'))

-- Prepare online plooting
loss_table = {}
val_table = {}
epoch = 0
best_val = 1000000000
patience = 0

while epoch < opt.max_iter do
    train()
    validation()

    --Early stop
    if val_error < best_val then
        best_val = val_error
        patience = 0
        log.info('Best val at epoch .. ' .. epoch)
    else
        patience = patience + 1
        log.info('patience is running out ...' .. patience)
    end

    if patience >= opt.patience then
        log.info('Early stopping')
        break
    end
end

loss_logger:style{['loss']='-'}
loss_curve_logger:style{['loss_curve']='-'}
accuracy_logger:style{['val']='-'}
loss_logger:plot()
loss_curve_logger:plot()
accuracy_logger:plot()
