
--[[
Taken the hidden-unit values of the hidden units from RNN modules, This Top_Net performs the following operations:
]]--

require 'nn'
local RNN = require 'model.my_RNN'
local GRU = require 'model.my_GRU'
local model_utils = require 'util/model_utils'

local Top_RNN = {}

function Top_RNN.net(loader, opt)
  assert( opt.if_original_feature == 1, 'opt.if_original_feature == 0')

  local mul_net = nn.Sequential()
  local p = nn.ParallelTable()
  p:add(nn.Identity())
  local r1 = nn.Sequential()
  r1:add(nn.Replicate(loader.feature_dim, 2))
  p:add(r1)
  mul_net:add(p)
  mul_net:add(nn.CMulTable())
  Top_RNN.mul_net = mul_net

  local class_n = loader.class_size
  local rnn
  if opt.top_c == 'rnn' then
    rnn = RNN.rnn(loader.feature_dim, opt.top_lstm_size, opt.top_num_layers, opt.dropout)
  elseif opt.top_c == 'gru' then
    rnn = GRU.gru(loader.feature_dim, opt.top_lstm_size, opt.top_num_layers, opt.dropout)
  end
  -- the initial state of the cell/hidden states
  local rnn_init_state = {}
  for L=1,opt.top_num_layers do
    local h_init = torch.zeros(1, opt.top_lstm_size)
    if opt.gpuid >=0 then h_init = h_init:cuda() end
    table.insert(rnn_init_state, h_init:clone())
  end
  Top_RNN.rnn_init_state = rnn_init_state
  local rnn_params_flat, rnn_grad_params_flat = rnn:getParameters()
  Top_RNN.rnn = rnn

  local top_c = nn.Sequential()
  top_c:add(nn.Linear(opt.top_lstm_size, class_n))
  top_c:add(nn.LogSoftMax())
  Top_RNN.top_c = top_c
  Top_RNN.params_size = mul_net:getParameters():nElement() + rnn_params_flat:nElement() + top_c:getParameters():nElement()
  print('number of parameters in the top lstm model: ' .. Top_RNN.params_size)
end

function Top_RNN.init_params(opt)

end

function Top_RNN.clone_model(loader) 
  print('cloning rnn')
  local clones_rnn = model_utils.clone_many_times(Top_RNN.rnn, loader.max_time_series_length)
  print('cloning ' .. loader.max_time_series_length ..  ' rnns for each time series finished! ')
  Top_RNN.clones_rnn = clones_rnn
end

function Top_RNN.forward(x, attention_weight, opt, flag)
  local mul_net = Top_RNN.mul_net
  local rnn_init_state = Top_RNN.rnn_init_state
  local clones_rnn = Top_RNN.clones_rnn
  local x_length = x:size(2)
  local top_c = Top_RNN.top_c
  
  -- forward for mul_net
  local weighted_input = mul_net:forward({x, attention_weight})
  Top_RNN.weighted_input = weighted_input
  
  local init_state_global = clone_list(rnn_init_state)
    -- perform forward for forward rnn
  local rnn_input = weighted_input
  local rnn_state = {[0] = init_state_global}
  local hidden_z_value = nil  -- the value of rnn1 hidden unit in the last time step 
  -- we don't set the opt.seq_length, instead, we use the current length of the time series
  for t=1,x_length do
    if flag == 'test' then
      clones_rnn[t]:evaluate() -- make sure we are in correct mode (this is cheap, sets flag)
    else
      clones_rnn[t]:training() -- make sure we are in correct mode (this is cheap, sets flag)
    end
    local lst = clones_rnn[t]:forward{rnn_input:narrow(2, t, 1):t(), unpack(rnn_state[t-1])}
    rnn_state[t] = {}
    for i=1,#rnn_init_state do table.insert(rnn_state[t], lst[i]) end -- extract the state, without output
    -- last element is the output of the current time step: the hidden value after dropout  
    if t== x_length then  
      hidden_z_value = lst[#lst]
    end
  end
  Top_RNN.rnn_state = rnn_state
  Top_RNN.hidden_z_value = hidden_z_value
  local output_v = top_c:forward(hidden_z_value)
  
  return output_v
end

function Top_RNN.backward(x, attention_weight, opt, gradout, loader)
  local rnn_init_state = Top_RNN.rnn_init_state
  local clones_rnn = Top_RNN.clones_rnn
  local rnn_state = Top_RNN.rnn_state
  local top_c = Top_RNN.top_c
  local mul_net = Top_RNN.mul_net
  local hidden_z_value = Top_RNN.hidden_z_value
  local x_length = x:size(2)
  local drnn_x = torch.DoubleTensor(loader.feature_dim, x_length):zero()
  local top_c_grad = top_c:backward(hidden_z_value, gradout)
  local dzeros = torch.zeros(opt.top_lstm_size)
  
  local rnn_input = Top_RNN.weighted_input
  -- backward for rnn and birnn
  local drnn_state = {[x_length] = clone_list(rnn_init_state, true)} -- true also zeros the clones
  -- perform back propagation through time (BPTT)
  for t = x_length,1,-1 do
    local doutput_t
    if t == x_length then
      doutput_t = top_c_grad
    else
      doutput_t = dzeros
    end
    table.insert(drnn_state[t], doutput_t)
    local dlst = clones_rnn[t]:backward({rnn_input:narrow(2, t, 1):t(), unpack(rnn_state[t-1])}, drnn_state[t])
    drnn_state[t-1] = {}
    for k,v in pairs(dlst) do
      if k == 1 then -- k == 1 is gradient on x, which we dont need
        -- note we do k-1 because first item is dembeddings, and then follow the 
        -- derivatives of the state, starting at index 2. I know...
        drnn_x:select(2, t):copy(v)
      else
        drnn_state[t-1][k-1] = v
      end
    end
  end
  
  -- backward for mul_net
  local grad_mul_net = mul_net:backward({x, attention_weight}, drnn_x)
  
  return grad_mul_net

end


return Top_RNN



