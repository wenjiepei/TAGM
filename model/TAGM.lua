
local model_utils = require 'util/model_utils'

local TAGM = {}

local function TAGM_model(input_size, rnn_size, n, dropout)
  dropout = dropout or 0 

  -- there will be n+2 inputs
  local inputs = {}
  table.insert(inputs, nn.Identity()()) -- x
  table.insert(inputs, nn.Identity()()) -- attention_weight == input_gate == 1-forget_gate
  for L = 1,n do
    -- since we don't have output gate, hence we prev_c = prev_h
    table.insert(inputs, nn.Identity()()) -- prev_h[L]
  end

  local x, input_size_L
  local outputs = {}
  local in_gate = inputs[2]
  local forget_gate = nn.AddConstant(1.0)(nn.MulConstant(-1.0)(in_gate))
  for L = 1,n do
    -- c,h from previos timesteps
    local prev_h = inputs[L+2]
    -- the input to this layer
    if L == 1 then 
      --      x = OneHot(input_size)(inputs[1])
      x = inputs[1]
      input_size_L = input_size
    else 
      x = outputs[L-1] 
      if dropout > 0 then x = nn.Dropout(dropout)(x) end -- apply dropout, if any
      input_size_L = rnn_size
    end
    -- evaluate the input sums at once for efficiency
    local i2h = nn.Linear(input_size_L, rnn_size)(x):annotate{name='i2h_'..L}
    local h2h = nn.Linear(rnn_size, rnn_size)(prev_h):annotate{name='h2h_'..L}
    local in_transform = nn.ReLU()(nn.CAddTable()({i2h, h2h}))
    -- decode the gates

    -- perform the LSTM update
    local next_h           = nn.CAddTable()({
      nn.CMulTable()({forget_gate, prev_h}),
      nn.CMulTable()({in_gate, in_transform})
    })
    table.insert(outputs, next_h)
  end

  -- set up the decoder
  local top_h = outputs[#outputs]
  local hidden_v = nn.Dropout(dropout)(top_h)
  --- for the normal LSTM
  --  local proj = nn.Linear(rnn_size, input_size)(top_h):annotate{name='decoder'}
  --  local logsoft = nn.LogSoftMax()(proj)
  --  table.insert(outputs, logsoft)
  --- in our case, we only use the last time step, hence we don't perform classification inside the LSTM model 
  table.insert(outputs, hidden_v)
  return nn.gModule(inputs, outputs)

end

function TAGM.model(loader, opt)

  local pre_m = nn.Replicate(opt.top_lstm_size, 1)
  TAGM.pre_m = pre_m
  local class_n = loader.class_size
  local top_lstm_variant, top_bilstm_variant
  top_lstm_variant = TAGM_model(loader.feature_dim, opt.top_lstm_size, opt.top_num_layers, opt.dropout)
  if opt.top_bidirection == 1 then
    top_bilstm_variant = TAGM_model(loader.feature_dim, opt.top_lstm_size, opt.top_num_layers, opt.dropout)
  end
  -- the initial state of the cell/hidden states
  local rnn_init_state = {}
  for L=1,opt.top_num_layers do
    local h_init = torch.zeros(1, opt.top_lstm_size)
    if opt.gpuid >=0 then h_init = h_init:cuda() end
    table.insert(rnn_init_state, h_init:clone())
  end
  TAGM.rnn_init_state = rnn_init_state
  local rnn_params_flat, rnn_grad_params_flat = top_lstm_variant:getParameters()
  TAGM.lstm = top_lstm_variant
  TAGM.bilstm = top_bilstm_variant

  local top_c = nn.Sequential()
  if opt.top_bidirection == 0 then
    top_c:add(nn.Linear(opt.top_lstm_size, class_n))
  else
    top_c:add(nn.Linear(2*opt.top_lstm_size, class_n))
  end
  top_c:add(nn.LogSoftMax())
  TAGM.top_c = top_c
  if opt.top_bidirection == 0 then
    TAGM.params_size = rnn_params_flat:nElement() + top_c:getParameters():nElement()
  else
    TAGM.params_size = 2*rnn_params_flat:nElement() + top_c:getParameters():nElement()
  end
  print('number of parameters in the top lstm model: ' .. TAGM.params_size)
end

function TAGM.clone_model(loader, opt) 
  print('cloning rnn')
  local clones_rnn = model_utils.clone_many_times(TAGM.lstm, loader.max_time_series_length)
  print('cloning ' .. loader.max_time_series_length ..  ' rnns for each time series finished! ')
  TAGM.clones_rnn = clones_rnn
  if opt.top_bidirection == 1 then
    print('cloning bidirectional rnn')
    local clones_birnn = model_utils.clone_many_times(TAGM.bilstm, loader.max_time_series_length)
    print('cloning ' .. loader.max_time_series_length ..  ' birnns for each time series finished! ')
    TAGM.clones_birnn = clones_birnn
  end
end

function TAGM.init_params(opt)

end

function TAGM.forward(x, attention_weight, opt, flag)
  local pre_m = TAGM.pre_m
  local lstm = TAGM.lstm
  local top_c = TAGM.top_c
  local rnn_init_state = TAGM.rnn_init_state
  local clones_rnn = TAGM.clones_rnn
  local clones_birnn = TAGM.clones_birnn

  -- forward of pre_m
  local pre_out = pre_m:forward(attention_weight)
  -- forward of lstm
  local x_length = x:size(2)
  local init_state_global = clone_list(rnn_init_state)
  local rnn_input = x
  local rnn_state = {[0] = init_state_global}
  local hidden_z_value = nil  -- the value of rnn1 hidden unit in the last time step 
  -- we don't set the opt.seq_length, instead, we use the current length of the time series
  for t=1,x_length do
    if flag == 'test' then
      clones_rnn[t]:evaluate() -- make sure we are in correct mode (this is cheap, sets flag)
    else
      clones_rnn[t]:training() -- make sure we are in correct mode (this is cheap, sets flag)
    end
    local lst = clones_rnn[t]:forward{rnn_input:narrow(2, t, 1):squeeze(), pre_out:narrow(2, t, 1):squeeze(), unpack(rnn_state[t-1])}
    rnn_state[t] = {}
    for i=1,#rnn_init_state do table.insert(rnn_state[t], lst[i]) end -- extract the state, without output
    -- last element is the output of the current time step: the hidden value after dropout  
    if t== x_length then  
      hidden_z_value = lst[#lst]
    end
  end
  TAGM.rnn_state = rnn_state
  
  -- forward of bilstm
  if opt.top_bidirection == 1 then
    local birnn_state = {[x_length+1] = init_state_global}
    local bihidden_z_value = nil  -- the value of rnn1 hidden unit in the last time step 
    -- we don't set the opt.seq_length, instead, we use the current length of the time series
    for t=x_length, 1, -1 do
      if flag == 'test' then
        clones_birnn[t]:evaluate() -- make sure we are in correct mode (this is cheap, sets flag)
      else
        clones_birnn[t]:training() -- make sure we are in correct mode (this is cheap, sets flag)
      end
      local lst = clones_birnn[t]:forward{rnn_input:narrow(2, t, 1):squeeze(), pre_out:narrow(2, t, 1):squeeze(), unpack(birnn_state[t+1])}
      birnn_state[t] = {}
      for i=1,#rnn_init_state do table.insert(birnn_state[t], lst[i]) end -- extract the state, without output
      -- last element is the output of the current time step: the hidden value after dropout  
      if t== 1 then  
        bihidden_z_value = lst[#lst]
      end
    end
    TAGM.birnn_state = birnn_state
    -- concatenate the output of forward and backward LSTM
    hidden_z_value = torch.cat(hidden_z_value, bihidden_z_value, 1)
  end
  
  TAGM.hidden_z_value = hidden_z_value
  TAGM.pre_out = pre_out
  local output_v = top_c:forward(hidden_z_value)
  
  return output_v
  
end

function TAGM.backward(x, attention_weight, opt, gradout, loader)
  local pre_m = TAGM.pre_m
  local lstm = TAGM.lstm
  local top_c = TAGM.top_c
  local rnn_init_state = TAGM.rnn_init_state
  local clones_rnn = TAGM.clones_rnn
  local clones_birnn = TAGM.clones_birnn
  local rnn_state = TAGM.rnn_state
  local birnn_state = TAGM.birnn_state
  local x_length = x:size(2)

  local hidden_z_value = TAGM.hidden_z_value
  local drnn_pre = torch.DoubleTensor(opt.top_lstm_size, x_length):zero()
  local bidrnn_pre = torch.DoubleTensor(opt.top_lstm_size, x_length):zero()
  local rnn_input = x
  local pre_out = TAGM.pre_out
  local top_c_grad = top_c:backward(hidden_z_value, gradout)
  local grad_net1, grad_net2
  if opt.top_bidirection == 1 then
    grad_net1 = top_c_grad:sub(1, opt.top_lstm_size)
    grad_net2 = top_c_grad:sub(opt.top_lstm_size+1, -1)
  else
    grad_net1 = top_c_grad
  end
  local dzeros = torch.zeros(opt.top_lstm_size)
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
    local dlst = clones_rnn[t]:backward({rnn_input:narrow(2, t, 1):squeeze(), pre_out:narrow(2, t, 1):squeeze(), 
      unpack(rnn_state[t-1])}, drnn_state[t])
    drnn_state[t-1] = {}
    for k,v in pairs(dlst) do
      if k == 2 then -- k == 1 is gradient on x, which we dont need
        -- note we do k-1 because first item is dembeddings, and then follow the 
        -- derivatives of the state, starting at index 2. I know...
        drnn_pre:select(2, t):copy(v)
      elseif k>2 then
        drnn_state[t-1][k-2] = v
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
      local dlst = clones_birnn[t]:backward({rnn_input:narrow(2, t, 1):squeeze(), pre_out:narrow(2, t, 1):squeeze(), 
        unpack(birnn_state[t+1])}, bidrnn_state[t])
      bidrnn_state[t+1] = {}
      for k,v in pairs(dlst) do
        if k == 2 then -- k == 1 is gradient on x, which we dont need
          -- note we do k-1 because first item is dembeddings, and then follow the 
          -- derivatives of the state, starting at index 2. I know...
          bidrnn_pre:select(2, t):copy(v)
        elseif k>2 then
          bidrnn_state[t+1][k-2] = v
        end
      end
    end
    drnn_pre:add(bidrnn_pre)
  end
  
  -- backward for mul_net
  local grad_mul_net = pre_m:backward(attention_weight, drnn_pre)
  
  return grad_mul_net

end

return TAGM
