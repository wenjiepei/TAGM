
local matio = require 'matio'
local path = require 'pl.path'

local mat_to_t7 = {}

function mat_to_t7.mat_to_t7_data(input_mat_path, input_dir, out_dir)
  print('loading data: '.. input_mat_path)
  local data = matio.load(input_mat_path)
  -- get the file name
  local filename = string.sub(input_mat_path, input_dir:len()+2, input_mat_path:len()-4)
  local output_path = path.join(out_dir, filename)
  local output = string.format(output_path .. '.t7')
  torch.save(output, data)

end

return mat_to_t7
