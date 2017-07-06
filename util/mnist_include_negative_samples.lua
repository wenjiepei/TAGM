

require 'torch'
require 'image'

local data_dir = '../../data/mnist/mnist_distort.t7'
local data = torch.load(data_dir)
print('train size: ', data.train_X:size(), data.train_T:size())
print('train size: ', data.test_X:size(), data.test_T:size())


local train_negative_size = data.train_X:size(1) / 10 * 2
local test_negative_size = data.test_X:size(1) / 10 * 2
local train_negative_empty_samples = torch.zeros(train_negative_size, 1, data.train_X:size(3), data.train_X:size(4))
local test_negative_empty_samples = torch.zeros(test_negative_size, 1, data.train_X:size(3), data.train_X:size(4))
print(train_negative_empty_samples:size())
print(test_negative_empty_samples:size())
local v = data.train_X[1][1][1][1]
--train_negative_empty_samples:add(v)
--test_negative_empty_samples:add(v)
train_negative_empty_samples:uniform(-1, 1)
test_negative_empty_samples:uniform(-1, 1)


data.train_X = torch.cat(data.train_X, train_negative_empty_samples, 1)
local lab = torch.zeros(train_negative_size):add(11):byte()
data.train_T = torch.cat(data.train_T, lab, 1)
local randp = torch.randperm(data.train_X:size(1))
data.train_X = data.train_X:index(1, randp:long())
data.train_T = data.train_T:index(1, randp:long())

data.test_X = torch.cat(data.test_X, test_negative_empty_samples, 1)
local lab = torch.zeros(test_negative_size):add(11):byte()
data.test_T = torch.cat(data.test_T, lab, 1)
local randp = torch.randperm(data.test_X:size(1))
data.test_X = data.test_X:index(1, randp:long())
data.test_T = data.test_T:index(1, randp:long())

data.validation_X = data.test_X
data.validation_T = data.test_T

local output = string.format('../../data/mnist/mnist_distort_negative.t7')
torch.save(output, {train_X = data.train_X, validation_X = data.validation_X, 
  test_X = data.test_X, train_T = data.train_T, validation_T = data.validation_T, test_T = data.test_T})
print('done!')