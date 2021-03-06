
require 'torch'
require 'lfs'
require 'image'
local path = require 'pl.path'

local original_data = {}
local data_dir = '../../data/arabic/window_2.t7'
--local data_dir = '../../data/arabic_voice/arabic_voice_window_3_ifperm_1.t7'
local data = torch.load(data_dir)
local labels = data.new_labels:squeeze()
local data_size = labels:size(1)
original_data.class_size = data.new_labels:max() 
print('data size: ', data_size)
print('class number: ', original_data.class_size)

local function get_statistical_info()
  -- get the statistical information about the feature info
  local max_length = 0
  local max_feat_v = data.new_X[1]:max(2)
  local min_feat_v = data.new_X[1]:min(2)
  local mean_feat_v = torch.zeros(min_feat_v:size())
  local length_tensor = torch.zeros(data_size)
  for i = 1, data_size do
    length_tensor[i] = data.new_X[i]:size(2)
    local temp = data.new_X[i]:max(2)
    max_feat_v = torch.cmax(max_feat_v, temp)
    temp = data.new_X[i]:min(2)
    min_feat_v = torch.cmin(min_feat_v, temp)
    temp = data.new_X[i]:sum(2)
    mean_feat_v = mean_feat_v + temp
  end
  original_data.mean_feat_v = mean_feat_v / torch.sum(length_tensor)
  print(original_data.mean_feat_v)
  original_data.length_tensor = length_tensor
  max_length = torch.max(length_tensor)
  local std_length = torch.std(length_tensor)
  local mean_length = torch.mean(length_tensor)
  print('The max length of the time series in this data set: ', max_length)
  print('The mean length of the time series in this data set: ', mean_length)
  print('The std variance of the time series in this data set: ', std_length) 
  original_data.max_time_series_length = max_length
  original_data.feature_dim = data.new_X[1]:size(1) 
  print('feature dimension: ', original_data.feature_dim)
  original_data.max_feat_v = max_feat_v
  original_data.min_feat_v = min_feat_v
--  print('max_feat_v:')
--  print(max_feat_v)
--  print('min_feat_v:')
--  print(min_feat_v)
end
get_statistical_info()


---visualize the one sample as a image
local function visualize_feat(sample)
  local img = sample:contiguous():view(1, sample:size(1), sample:size(2))
  local img = image.scale(img,'*5', 'simple')
  image.display({image=img})
end

--local indexs = 880*4
--for i = 1, 5 do
--  local index = indexs + i*20
--  visualize_feat(data.new_X[index])
--end

--- append each time series to the max length with the random value between [minv, maxv]
local train_ratio = 0.625
local validation_ratio = 0.125
local test_ratio = 0.25
local nTrain = train_ratio * data_size
local nValidation = validation_ratio * data_size
local nTest = test_ratio * data_size
local each_class_s = data_size / original_data.class_size
-- convert to BDHW format
local train_X = torch.DoubleTensor(nTrain, 1, original_data.feature_dim, original_data.max_time_series_length)
local validation_X = torch.DoubleTensor(nValidation, 1, original_data.feature_dim, original_data.max_time_series_length)
local test_X = torch.DoubleTensor(nTest, 1, original_data.feature_dim, original_data.max_time_series_length)
local train_T = torch.DoubleTensor(nTrain)
local validation_T = torch.DoubleTensor(nValidation)
local test_T = torch.DoubleTensor(nTest)
print('training size: ', nTrain)
print('validation size: ', nValidation)
print('test size: ', nTest)

local function append_feat(if_noise)
  local max_feat_mat = original_data.max_feat_v:repeatTensor(1, original_data.max_time_series_length)
  local min_feat_mat = original_data.min_feat_v:repeatTensor(1, original_data.max_time_series_length)
  local mean_feat_mat = original_data.mean_feat_v:repeatTensor(1, original_data.max_time_series_length)
  local dist = max_feat_mat - min_feat_mat
  local train_count = 0 
  local validation_count = 0
  local test_count = 0
  for i = 1, data_size do
    local append_len = original_data.max_time_series_length - data.new_X[i]:size(2)
    local new_feat = nil
    if if_noise > 0 and append_len > 0 then
      local rand_feat = torch.rand(original_data.feature_dim, append_len)
      rand_feat:cmul(torch.mul(dist:sub(1, -1, 1, append_len), 0.5))
      rand_feat:add(torch.add(min_feat_mat:sub(1, -1, 1, append_len), torch.mul(dist:sub(1, -1, 1, append_len), 0.25)))
--      rand_feat:zero()
--      rand_feat:add(max_feat_mat:sub(1, -1, 1, append_len)*0.5)
      local randp = math.floor(torch.uniform(0.2, 0.8)*append_len)
      new_feat = torch.cat({rand_feat:sub(1, -1, 1, randp), data.new_X[i], rand_feat:sub(1, -1, randp+1, -1)}, 2)
    else
      new_feat = data.new_X[i]  
    end
    
--    visualize_feat(data.new_X[i])
--    visualize_feat(new_feat)
    if i % each_class_s <= each_class_s*train_ratio and i % each_class_s > 0 then
      train_count = train_count + 1
      train_X[train_count] = new_feat
      train_T[train_count] = labels[i]
--      print(i, '  ', train_count)
    elseif  i % each_class_s <= each_class_s*(1-test_ratio) and i % each_class_s > 0 then
      validation_count = validation_count + 1
      validation_X[validation_count] = new_feat
      validation_T[validation_count] = labels[i]
    else
      test_count = test_count + 1
      test_X[test_count] = new_feat
      test_T[test_count] = labels[i]
    end
--    if i>5 then
--    image.display({image = train_X:narrow(1,i-5,5)})
--    end
  end
  
--  print(train_count, '  ', validation_count, '  ', test_count )
end
local if_noise = 1
append_feat(if_noise)
--print(train_X:size())
--- save the data
local pa = string.sub(data_dir, 1, data_dir:len()-3)
local output = nil
if if_noise > 0 then
  output = string.format(pa .. '_append_noise.t7')
else
  output = string.format(pa .. '_append_clean.t7')
end
torch.save(output, {train_X = train_X, validation_X = validation_X, 
test_X = test_X, train_T = train_T, validation_T = validation_T, test_T = test_T})

