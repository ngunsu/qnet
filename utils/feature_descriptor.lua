require 'torch'
require 'nn'

local FeatureDescriptor = torch.class('FeatureDescriptor')

function FeatureDescriptor:__init(model, model_path)
    self.model = model
    self.model_path = model_path
end

function FeatureDescriptor:load_model()
    full_net = torch.load(self.model_path)
    if model == 'quadruplet' then
        self.net = full_net.desc
        self.mean = full_net.mean
        self.std = full_net.std
    end
    print(self.net)
end

function FeatureDescriptor:compute(patches)
    des = 1
    return des
end

