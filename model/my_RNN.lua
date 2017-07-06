
local RNN = {}

function RNN.rnn(input_size, rnn_size, n, dropout)
  
  -- there are n+1 inputs (hiddens on each layer and x)
  local inputs = {}
  table.insert(inputs, nn.Identity()()) -- x
  for L = 1,n do
    table.insert(inputs, nn.Identity()()) -- prev_h[L]

  end

  local x, input_size_L
  local outputs = {}
  for L = 1,n do
    
    local prev_h = inputs[L+1]
    if L == 1 then 
      x = inputs[1]
      input_size_L = input_size
    else 
      x = outputs[(L-1)] 
      if dropout > 0 then x = nn.Dropout(dropout)(x) end -- apply dropout, if any
      input_size_L = rnn_size
    end

    -- RNN tick
    local i2h = nn.Linear(input_size_L, rnn_size)(x)
    local h2h = nn.Linear(rnn_size, rnn_size)(prev_h)
    local next_h = nn.SoftPlus(5)(nn.CAddTable(){i2h, h2h}) -- a smooth approximation, whose derivative is just logistic function
--    local next_h = nn.ReLU()(nn.CAddTable(){i2h, h2h}) -- original ReLU algorithm

    table.insert(outputs, next_h)
  end
-- set up the decoder
-- In our case, we just directly use the value of the corresponding hidden unit as the output 
  local top_h = outputs[#outputs]
  local hidden_v = nn.Dropout(dropout)(top_h)
  table.insert(outputs, hidden_v)
  return nn.gModule(inputs, outputs)
end

return RNN
