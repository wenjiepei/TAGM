local path = require 'pl.path'
local table_operation = require 'util.table_operation'

local data_loader = {}
data_loader.__index = data_loader


function data_loader.create(opt)
  local self = {}
  setmetatable(self, data_loader)
  self:load_data_by_index_t7(opt)
  
  self.batch_ix = 1
  self.rand_order = torch.randperm(self.nTrain)
  print('data load done. ')
  print('before collectgarbage: ', collectgarbage("count"))
  collectgarbage()
  print('after collectgarbage: ', collectgarbage("count"))
  --  os.exit()
  return self
end

-- Return the tensor of index of the data with batch size 
function data_loader:get_next_train_batch(batch_size)
  self.previous_batch_ix = self.batch_ix
  local set_size = self.nTrain;
  local rtn_index = torch.zeros(batch_size)
  for i = 1, batch_size do
    local temp_ind = i + self.batch_ix - 1
    if temp_ind > set_size then -- cycle around to beginning
      temp_ind = temp_ind - set_size
    end
    rtn_index[i] = self.rand_order[temp_ind]
  end
  self.batch_ix = self.batch_ix + batch_size;
  -- cycle around to beginning
  if self.batch_ix >set_size then
    self.batch_ix = 1
    -- randomize the order of the training set
    self.rand_order = torch.randperm(self.nTrain)
  end
  return rtn_index;
end

function data_loader:load_data_by_index_t7(opt)
  print('loading ', opt.data_set, ' data: ')
  local data_dir = nil
  local data = nil
  if opt.data_set == 'arabic_append_noise' then
    data_dir = '../../data/arabic/window_2_append_noise.t7'
    data = torch.load(data_dir)
  elseif opt.data_set == 'SST_2' or opt.data_set == 'SST_5' then
    data_dir = '../../data/SST/SST.t7'
    data = torch.load(data_dir)
    data.train_X = {}
    data.validation_X = {}
    data.test_X = {}
    self.train_sentences = {}
    self.validation_sentences = {}
    self.test_sentences = {}
    local train_ind, validation_ind, test_ind, sentence_labels_C
    if opt.data_set == 'SST_2' then
      train_ind = data.train_2C_ind
      validation_ind = data.validation_2C_ind
      test_ind = data.test_2C_ind 
      sentence_labels_C = data.sentence_labels_2C
    else -- SST_5
      train_ind = data.train_5C_ind
      validation_ind = data.validation_5C_ind
      test_ind = data.test_5C_ind 
      sentence_labels_C = data.sentence_labels_5C
    end
    local max_l = 0
    for k = 1, train_ind:size(1) do
      local d = train_ind[k]:squeeze()
      data.train_X[#data.train_X+1] = data.dat_X[d]:t()
      max_l = math.max(max_l, data.dat_X[d]:size(1))
      self.train_sentences[#self.train_sentences+1] = data.sentences[d]
    end
    for k = 1, validation_ind:size(1) do
      local d = validation_ind[k]:squeeze()
      data.validation_X[#data.validation_X+1] = data.dat_X[d]:t()
      max_l = math.max(max_l, data.dat_X[d]:size(1))
      self.validation_sentences[#self.validation_sentences+1] = data.sentences[d]
    end
    for k = 1, test_ind:size(1) do
      local d = test_ind[k]:squeeze()
      data.test_X[#data.test_X+1] = data.dat_X[d]:t()
      max_l = math.max(max_l, data.dat_X[d]:size(1))
      self.test_sentences[#self.test_sentences+1] = data.sentences[d]
    end
    data.train_T = sentence_labels_C:squeeze():index(1, train_ind:squeeze():long())
    data.validation_T = sentence_labels_C:squeeze():index(1, validation_ind:squeeze():long())
    data.test_T = sentence_labels_C:squeeze():index(1, test_ind:squeeze():long())
    self.train_score = data.sentence_labels:squeeze():index(1, train_ind:squeeze():long())
    self.validation_score = data.sentence_labels:squeeze():index(1, validation_ind:squeeze():long())
    self.test_score = data.sentence_labels:squeeze():index(1, test_ind:squeeze():long())
    data.max_len = max_l
  elseif opt.data_set == 'SST_2_train_all' or opt.data_set == 'SST_5_train_all' then
    data_dir = '../../data/SST/SST.t7'
    local data_dir1 = '../../data/SST/train_all_X_T1.t7'
    local data_dir2 = '../../data/SST/train_all_X_T2.t7'
    data = torch.load(data_dir)
    local data1 = torch.load(data_dir1)
    local data2 = torch.load(data_dir2)
    data.train_X = {}
    data.validation_X = {}
    data.test_X = {}
    self.train_sentences = {}
    self.validation_sentences = {}
    self.test_sentences = {}
    local validation_ind, test_ind, sentence_labels_C
    local train_T1 = data1.train_all_T1:squeeze()
    local train_T2 = data2.train_all_T2:squeeze()
    local train_T = torch.zeros(train_T1:nElement()+train_T2:nElement())
    if opt.data_set == 'SST_2_train_all' then
      local ind = 0
      for i = 1, train_T1:nElement() do
        if train_T1[i] >3 then
          ind = ind + 1 
          train_T[ind] = 2
          data.train_X[ind] = data1.train_all_X1[i]:t()
        elseif train_T1[i] < 3 then
          ind = ind + 1 
          train_T[ind] = 1
          data.train_X[ind] = data1.train_all_X1[i]:t()
        end
      end
      for i = 1, train_T2:nElement() do
        if train_T2[i] >3 then
          ind = ind + 1 
          train_T[ind] = 2
          data.train_X[ind] = data2.train_all_X2[i]:t()
        elseif train_T2[i] < 3 then
          ind = ind + 1 
          train_T[ind] = 1
          data.train_X[ind] = data2.train_all_X2[i]:t()
        end
      end
      data.train_T = train_T:sub(1, ind):clone()
      validation_ind = data.validation_2C_ind
      test_ind = data.test_2C_ind 
      sentence_labels_C = data.sentence_labels_2C
    else -- SST_5
      for i = 1, train_T1:nElement() do
        train_T[i] = train_T1[i]
        data.train_X[i] = data1.train_all_X1[i]:t()
    end
    local ind = train_T1:nElement()
    for i = 1, train_T2:nElement() do
      train_T[i+ind] = train_T2[i]
      data.train_X[i+ind] = data2.train_all_X2[i]:t()
    end
    data.train_T = train_T:clone()
    validation_ind = data.validation_5C_ind
    test_ind = data.test_5C_ind 
    sentence_labels_C = data.sentence_labels_5C
    end
    local max_l = 0
    for k = 1, validation_ind:size(1) do
      local d = validation_ind[k]:squeeze()
      data.validation_X[#data.validation_X+1] = data.dat_X[d]:t()
      max_l = math.max(max_l, data.dat_X[d]:size(1))
      self.validation_sentences[#self.validation_sentences+1] = data.sentences[d]
    end
    for k = 1, test_ind:size(1) do
      local d = test_ind[k]:squeeze()
      data.test_X[#data.test_X+1] = data.dat_X[d]:t()
      max_l = math.max(max_l, data.dat_X[d]:size(1))
      self.test_sentences[#self.test_sentences+1] = data.sentences[d]
    end
    data.validation_T = sentence_labels_C:squeeze():index(1, validation_ind:squeeze():long())
    data.test_T = sentence_labels_C:squeeze():index(1, test_ind:squeeze():long())
    self.validation_score = data.sentence_labels:squeeze():index(1, validation_ind:squeeze():long())
    self.test_score = data.sentence_labels:squeeze():index(1, test_ind:squeeze():long())
    data.max_len = max_l
  else
    error('no such feature data!')
  end

  self.train_X = data.train_X
  self.validation_X = data.validation_X
  self.test_X = data.test_X
  self.train_T = data.train_T
  self.validation_T = data.validation_T
  self.test_T = data.test_T
  self.nTrain = self.train_T:size(1)
  self.nValidation = self.validation_T:size(1)
  self.nTest = self.test_T:size(1)
  local data_size = self.train_T:nElement()+self.validation_T:nElement()+self.test_T:nElement()
  self.class_size = data.train_T:max() 
  print('class number: ', self.class_size)

  print('training size: ', self.train_T:size(1))
  print('validation size: ', self.validation_T:size(1))
  print('test size: ', self.test_T:size(1))
  if opt.data_set == 'SST_2' or opt.data_set == 'SST_5' or opt.data_set == 'SST_2_train_all' 
    or opt.data_set == 'SST_5_train_all' then
    self.max_time_series_length = data.max_len
  else
    self.max_time_series_length = self.train_X:size(4)
  end
  print('The max length of the time series in this data set: ', self.max_time_series_length)
  if opt.data_set == 'SST_2' or opt.data_set == 'SST_5' or opt.data_set == 'SST_2_train_all' 
    or opt.data_set == 'SST_5_train_all' then
    self.feature_dim = data.train_X[1]:size(1)
  else
    self.feature_dim = self.train_X:size(3)
  end
  print('feature dimension: ', self.feature_dim)
end

return data_loader
