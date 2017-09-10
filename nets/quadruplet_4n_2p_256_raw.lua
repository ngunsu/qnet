nn = require 'nn'

require 'torch'
torch.manualSeed(0)
-------------------------------------------------------------------------------
-- Feature Networks
-------------------------------------------------------------------------------

-- Setup the features network
f_network_g1 = nn.Sequential()
f_network_g1:add(nn.SpatialConvolution(1, 32, 7, 7))
f_network_g1:add(nn.Tanh(true))
f_network_g1:add(nn.SpatialMaxPooling(2,2,2,2))
f_network_g1:add(nn.SpatialConvolution(32, 64, 6, 6))
f_network_g1:add(nn.Tanh(true))
f_network_g1:add(nn.View(64*8*8))
f_network_g1:add(nn.Linear(64*8*8, 256))
f_network_g1:add(nn.Tanh(true))

--  Clone the network to form quadruplets
f_network_i1 = f_network_g1:clone('weight', 'bias','gradWeight','gradBias')
f_network_g2 = f_network_g1:clone('weight', 'bias','gradWeight','gradBias')
f_network_i2 = f_network_g1:clone('weight', 'bias','gradWeight','gradBias')

-- add them to a parallel table
prl = nn.ParallelTable()
prl:add(f_network_g1)
prl:add(f_network_i1)
prl:add(f_network_g2)
prl:add(f_network_i2)

mlp= nn.Sequential()
mlp:add(prl)

-------------------------------------------------------------------------------
-- Compute distances (RGB_POS, NIR_POS, RGB_POS2, NIR_POS2)
-------------------------------------------------------------------------------
--
-- get feature distances
cc = nn.ConcatTable()

-- neg case = g1 g2
cnn_n_g1_g2 = nn.Sequential()
cnn_n_g1_g2_dist = nn.ConcatTable()
cnn_n_g1_g2_dist:add(nn.SelectTable(1))
cnn_n_g1_g2_dist:add(nn.SelectTable(3))
cnn_n_g1_g2:add(cnn_n_g1_g2_dist)
cnn_n_g1_g2:add(nn.PairwiseDistance(2))
cnn_n_g1_g2:add(nn.View(128,1))
cc:add(cnn_n_g1_g2)

-- neg case = g1 i2
cnn_n_g1_i2 = nn.Sequential()
cnn_n_g1_i2_dist = nn.ConcatTable()
cnn_n_g1_i2_dist:add(nn.SelectTable(1))
cnn_n_g1_i2_dist:add(nn.SelectTable(4))
cnn_n_g1_i2:add(cnn_n_g1_i2_dist)
cnn_n_g1_i2:add(nn.PairwiseDistance(2))
cnn_n_g1_i2:add(nn.View(128,1))
cc:add(cnn_n_g1_i2)


-- neg case = i1 g2
cnn_n_i1_g2 = nn.Sequential()
cnn_n_i1_g2_dist = nn.ConcatTable()
cnn_n_i1_g2_dist:add(nn.SelectTable(2))
cnn_n_i1_g2_dist:add(nn.SelectTable(3))
cnn_n_i1_g2:add(cnn_n_i1_g2_dist)
cnn_n_i1_g2:add(nn.PairwiseDistance(2))
cnn_n_i1_g2:add(nn.View(128,1))
cc:add(cnn_n_i1_g2)

-- neg case = i1 i2
cnn_n_i1_i2 = nn.Sequential()
cnn_n_i1_i2_dist = nn.ConcatTable()
cnn_n_i1_i2_dist:add(nn.SelectTable(2))
cnn_n_i1_i2_dist:add(nn.SelectTable(4))
cnn_n_i1_i2:add(cnn_n_i1_i2_dist)
cnn_n_i1_i2:add(nn.PairwiseDistance(2))
cnn_n_i1_i2:add(nn.View(128,1))
cc:add(cnn_n_i1_i2)


-- pos case = g1 i1
cnn_p_g1_i1 = nn.Sequential()
cnn_p_g1_i1_dist = nn.ConcatTable()
cnn_p_g1_i1_dist:add(nn.SelectTable(1))
cnn_p_g1_i1_dist:add(nn.SelectTable(2))
cnn_p_g1_i1:add(cnn_p_g1_i1_dist)
cnn_p_g1_i1:add(nn.PairwiseDistance(2))
cnn_p_g1_i1:add(nn.View(128,1))
cc:add(cnn_p_g1_i1)

-- pos2 case = g2 i2
cnn_p_g2_i2 = nn.Sequential()
cnn_p_g2_i2_dist = nn.ConcatTable()
cnn_p_g2_i2_dist:add(nn.SelectTable(3))
cnn_p_g2_i2_dist:add(nn.SelectTable(4))
cnn_p_g2_i2:add(cnn_p_g2_i2_dist)
cnn_p_g2_i2:add(nn.PairwiseDistance(2))
cnn_p_g2_i2:add(nn.View(128,1))
cc:add(cnn_p_g2_i2)

mlp:add(cc)

-------------------------------------------------------------------------------
-- Last layer
-------------------------------------------------------------------------------

last_layer = nn.ConcatTable()

-- select min negative distance inside the triplet
mined_neg = nn.Sequential()
mining_layer = nn.ConcatTable()
mining_layer:add(nn.SelectTable(1))
mining_layer:add(nn.SelectTable(2))
mining_layer:add(nn.SelectTable(3))
mining_layer:add(nn.SelectTable(4))
mined_neg:add(mining_layer)
mined_neg:add(nn.JoinTable(2))
mined_neg:add(nn.Min(2))
mined_neg:add(nn.View(128,1))
last_layer:add(mined_neg)

-- select min negative distance inside the triplet
mined_pos = nn.Sequential()
mining_pos_layer = nn.ConcatTable()
mining_pos_layer:add(nn.SelectTable(5))
mining_pos_layer:add(nn.SelectTable(6))
mined_pos:add(mining_pos_layer)
mined_pos:add(nn.JoinTable(2))
mined_pos:add(nn.Max(2))
mined_pos:add(nn.View(128,1))
last_layer:add(mined_pos)


mlp:add(last_layer)

mlp:add(nn.JoinTable(2))

torch.save('qnet.t7', mlp:float())
