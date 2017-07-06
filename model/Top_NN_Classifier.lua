
--[[
Taken the hidden-unit values of the hidden units from RNN modules, This Top_Net performs the following operations:
]]--

require 'nn'

local Top_NN = {}

function Top_NN.net(loader, opt, class_n)
  local net = nn.Sequential()
  local p = nn.ParallelTable()
  if opt.if_original_feature == 1 then
    p:add(nn.Reshape(loader.feature_dim, -1))
    p:add(nn.Reshape(-1, 1))
    net:add(p)
    net:add(nn.MM())
    net:add(nn.Reshape(loader.feature_dim))
    net:add(nn.Linear(loader.feature_dim, 64))
  else
    local r1 = nn.Sequential()
    r1:add(nn.JoinTable(2))
    r1:add(nn.Reshape(2*opt.rnn_size, -1))
    p:add(r1)
    p:add(nn.Reshape(-1, 1))
    net:add(p)
    net:add(nn.MM())
    net:add(nn.Reshape(2*opt.rnn_size))
    net:add(nn.Linear(2*opt.rnn_size, 64))
  end

  net:add(nn.ReLU(true)) 
  net:add(nn.Linear(64, class_n))
  net:add(nn.LogSoftMax())
  
  return net
end

return Top_NN