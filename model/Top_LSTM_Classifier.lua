
--[[
Taken the hidden-unit values of the hidden units from RNN modules, This Top_Net performs the following operations:
]]--

require 'nn'
local LSTM = require 'model.my_LSTM'
local RNN = require 'model.my_RNN'
local GRU = require 'model.my_GRU'
local model_utils = require 'util/model_utils'

local Top_LSTM = {}

function Top_LSTM.net(loader, opt)
  assert( opt.if_original_feature == 1, 'opt.if_original_feature == 0')

  local mul_net = nn.Sequential()
  local p = nn.ParallelTable()
  p:add(nn.Identity())
  local r1 = nn.Sequential()
  r1:add(nn.Replicate(loader.feature_dim, 2))
  p:add(r1)
  mul_net:add(p)
  mul_net:add(nn.CMulTable())
  Top_LSTM.mul_net = mul_net

  local class_n = loader.class_size
  local top_lstm = nil
  if opt.top_c == 'rnn' then
    top_lstm = RNN.rnn(loader.feature_dim, opt.top_lstm_size, opt.top_num_layers, opt.dropout)
  elseif opt.top_c == 'gru' then
    top_lstm = GRU.gru(loader.feature_dim, opt.top_lstm_size, opt.top_num_layers, opt.dropout)
  elseif opt.top_c == 'lstm' then
    top_lstm = LSTM.lstm(loader.feature_dim, opt.top_lstm_size, opt.top_num_layers, opt.dropout)
  end
  local top_bilstm = nil
  if opt.top_bidirection == 1 then
    if opt.top_c == 'rnn' then
      top_bilstm = RNN.rnn(loader.feature_dim, opt.top_lstm_size, opt.top_num_layers, opt.dropout)
    elseif opt.top_c == 'gru' then
      top_bilstm = GRU.gru(loader.feature_dim, opt.top_lstm_size, opt.top_num_layers, opt.dropout)
    elseif opt.top_c == 'lstm' then
      top_bilstm = LSTM.lstm(loader.feature_dim, opt.top_lstm_size, opt.top_num_layers, opt.dropout)
    end
  end
  -- the initial state of the cell/hidden states
  local rnn_init_state = {}
  for L=1,opt.top_num_layers do
    local h_init = torch.zeros(1, opt.top_lstm_size)
    if opt.gpuid >=0 then h_init = h_init:cuda() end
    table.insert(rnn_init_state, h_init:clone())
    if opt.top_c == 'lstm' then
      table.insert(rnn_init_state, h_init:clone())
    end
  end
  Top_LSTM.rnn_init_state = rnn_init_state
  local rnn_params_flat, rnn_grad_params_flat = top_lstm:getParameters()
  Top_LSTM.rnn = top_lstm
  Top_LSTM.birnn = top_bilstm

  local top_c = nn.Sequential()
  if opt.top_bidirection == 0 then
    top_c:add(nn.Linear(opt.top_lstm_size, class_n))
  else
    top_c:add(nn.Linear(2*opt.top_lstm_size, class_n))
  end
  top_c:add(nn.LogSoftMax())
  Top_LSTM.top_c = top_c
  if opt.top_bidirection == 0 then
    Top_LSTM.params_size = mul_net:getParameters():nElement() + rnn_params_flat:nElement() + top_c:getParameters():nElement()
  else
    Top_LSTM.params_size = mul_net:getParameters():nElement() + 2*rnn_params_flat:nElement() + top_c:getParameters():nElement()
  end
  print('number of parameters in the top lstm model: ' .. Top_LSTM.params_size)
end

function Top_LSTM.init_params(opt)
  -- initialize the LSTM forget gates with slightly higher biases to encourage remembering in the beginning
  if opt.top_c == 'lstm' then
    for layer_idx = 1, opt.top_num_layers do
      for _,node in ipairs(Top_LSTM.rnn.forwardnodes) do
        if node.data.annotations.name == "i2h_" .. layer_idx then
          print('setting forget gate biases to 1 in LSTM layer ' .. layer_idx)
          -- the gates are, in order, i,f,o,g, so f is the 2nd block of weights
          node.data.module.bias[{{opt.top_lstm_size+1, 2*opt.top_lstm_size}}]:fill(1.0)
        end
      end
    end
  end
  if opt.top_bidirection == 1 then
    if opt.top_c == 'lstm' then
      for layer_idx = 1, opt.top_num_layers do
        for _,node in ipairs(Top_LSTM.birnn.forwardnodes) do
          if node.data.annotations.name == "i2h_" .. layer_idx then
            print('setting forget gate biases to 1 in LSTM layer ' .. layer_idx)
            -- the gates are, in order, i,f,o,g, so f is the 2nd block of weights
            node.data.module.bias[{{opt.top_lstm_size+1, 2*opt.top_lstm_size}}]:fill(1.0)
          end
        end
      end
    end 
  end 
end

function Top_LSTM.clone_model(loader, opt) 
  print('cloning rnn')
  local clones_rnn = model_utils.clone_many_times(Top_LSTM.rnn, loader.max_time_series_length)
  print('cloning ' .. loader.max_time_series_length ..  ' rnns for each time series finished! ')
  Top_LSTM.clones_rnn = clones_rnn
  if opt.bidirection == 1 then
    print('cloning birnn')
    local clones_birnn = model_utils.clone_many_times(Top_LSTM.birnn, loader.max_time_series_length)
    print('cloning ' .. loader.max_time_series_length ..  ' birnns for each time series finished! ')
    Top_LSTM.clones_birnn = clones_birnn
  end
end

function Top_LSTM.forward(x, attention_weight, opt, flag)
  local mul_net = Top_LSTM.mul_net
  local rnn_init_state = Top_LSTM.rnn_init_state
  local clones_rnn = Top_LSTM.clones_rnn
  local clones_birnn = Top_LSTM.clones_birnn
  local x_length = x:size(2)
  local top_c = Top_LSTM.top_c
  
  -- forward for mul_net
  local weighted_input = mul_net:forward({x, attention_weight})
  Top_LSTM.weighted_input = weighted_input
  
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
  Top_LSTM.rnn_state = rnn_state

  -- perform the forward pass for birnn: in the other direction
  if opt.top_bidirection == 1 then
    local birnn_state, bihidden_z_value
    local rnn_input = weighted_input
    local birnn_state = {[x_length+1] = init_state_global}
    bihidden_z_value = nil  -- the value of rnn1 hidden unit in the last time step 
    -- we don't set the opt.seq_length, instead, we use the current length of the time series
    for t=x_length, 1, -1 do
      if flag == 'test' then
        clones_birnn[t]:evaluate() -- make sure we are in correct mode (this is cheap, sets flag)
      else
        clones_birnn[t]:training() -- make sure we are in correct mode (this is cheap, sets flag)
      end
      local lst = clones_birnn[t]:forward{rnn_input:narrow(2, t, 1):t(), unpack(birnn_state[t+1])}
      birnn_state[t] = {}
      for i=1,#rnn_init_state do table.insert(birnn_state[t], lst[i]) end -- extract the state, without output
      -- last element is the output of the current time step: the hidden value after dropout
      if t == 1 then
        bihidden_z_value = lst[#lst]
      end
    end
    Top_LSTM.birnn_state = birnn_state
    -- concatenate the output of forward and backward LSTM
    hidden_z_value = torch.cat(hidden_z_value, bihidden_z_value, 2)
  end
  
  Top_LSTM.hidden_z_value = hidden_z_value
  local output_v = top_c:forward(hidden_z_value)
   
  return output_v
end

function Top_LSTM.backward(x, attention_weight, opt, gradout, loader)
  local rnn_init_state = Top_LSTM.rnn_init_state
  local clones_rnn = Top_LSTM.clones_rnn
  local clones_birnn = Top_LSTM.clones_birnn
  local rnn_state = Top_LSTM.rnn_state
  local birnn_state = Top_LSTM.birnn_state
  local top_c = Top_LSTM.top_c
  local mul_net = Top_LSTM.mul_net
  local hidden_z_value = Top_LSTM.hidden_z_value
  local x_length = x:size(2)
  local drnn_x = torch.DoubleTensor(loader.feature_dim, x_length):zero()
  local bidrnn_x = torch.DoubleTensor(loader.feature_dim, x_length):zero()
  local top_c_grad = top_c:backward(hidden_z_value, gradout)
  local dzeros = torch.zeros(opt.top_lstm_size)
  local grad_net1, grad_net2
  if opt.top_bidirection == 1 then
    grad_net1 = top_c_grad:sub(1, -1, 1, opt.top_lstm_size)
    grad_net2 = top_c_grad:sub(1, -1, opt.top_lstm_size+1, -1)
  else
    grad_net1 = top_c_grad
  end
  local rnn_input = Top_LSTM.weighted_input
  -- backward for rnn and birnn
  local drnn_state = {[x_length] = clone_list(rnn_init_state, true)} -- true also zeros the clones
  -- perform back propagation through time (BPTT)
  for t = x_length,1,-1 do
    local doutput_t
    if t == x_length then
      doutput_t = grad_net1
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
  
  if opt.top_bidirection == 1 then
    local bidrnn_state = {[1] = clone_list(rnn_init_state, true)} -- true also zeros the clones
    -- perform back propagation through time (BPTT)
    for t = 1, x_length do
      local doutput_t
      if t == 1 then
        doutput_t = grad_net2
      else
        doutput_t = dzeros
      end

      table.insert(bidrnn_state[t], doutput_t)
      local dlst = clones_birnn[t]:backward({rnn_input:narrow(2, t, 1):t(), unpack(birnn_state[t+1])}, bidrnn_state[t])
      bidrnn_state[t+1] = {}
      for k,v in pairs(dlst) do
        if k == 1 then -- k == 1 is gradient on x, which we dont need
          -- note we do k-1 because first item is dembeddings, and then follow the 
          -- derivatives of the state, starting at index 2. I know...
          bidrnn_x:select(2, t):copy(v)
        else
          bidrnn_state[t+1][k-1] = v
        end
      end
    end
  end

  if opt.top_bidirection then
    drnn_x:add(bidrnn_x)
  end
  
  -- backward for mul_net
  local grad_mul_net = mul_net:backward({x, attention_weight}, drnn_x)
  
  return grad_mul_net

end


return Top_LSTM



