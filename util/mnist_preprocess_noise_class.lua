
require 'torch'
require 'image'

local data_dir = '../../data/mnist/mnist_distort.t7'
local data = torch.load(data_dir)
print('train size: ', data.train_X:size(), data.train_T:size())
print('train size: ', data.test_X:size(), data.test_T:size())

---visualize the one sample as a image
local function visualize_feat(sample, w)
  local img = sample:contiguous():view(1, sample:size(1), sample:size(2))
  local img = image.scale(img,'*3', 'simple')
  w = image.display({image=img, win = w})
  print(sample:size(2))
  return w
end

--image.display({image = data.train_X:narrow(1,5,5)})
--print(data.train_T:narrow(1,5,5))

local interest_class = 5

local train_interest_ind={}
local train_noise_ind={}
local test_interest_ind={}
local test_noise_ind={}
for i = 1, data.train_X:size(1) do
  if data.train_T[i] <= 5 then
    train_interest_ind[#train_interest_ind+1] = i
  else
    train_noise_ind[#train_noise_ind+1] = i
  end
end

for i = 1, data.test_X:size(1) do
  if data.test_T[i] <= 5 then
    test_interest_ind[#test_interest_ind+1] = i
  else
    test_noise_ind[#test_noise_ind+1] = i
  end
end
train_interest_ind = torch.LongTensor(train_interest_ind)
train_noise_ind = torch.LongTensor(train_noise_ind)
test_interest_ind = torch.LongTensor(test_interest_ind)
test_noise_ind = torch.LongTensor(test_noise_ind)
local train_X = data.train_X:index(1, train_interest_ind)
local train_noise_X = data.train_X:index(1, train_noise_ind)
local train_T = data.train_T:index(1, train_interest_ind)
local test_X = data.test_X:index(1, test_interest_ind)
local test_noise_X = data.test_X:index(1, test_noise_ind)
local test_T = data.test_T:index(1, test_interest_ind)
print(train_X:size())

local train_X_new = torch.Tensor(train_X:size(1), train_X:size(2), train_X:size(3), 3*train_X:size(4))
local test_X_new = torch.Tensor(test_X:size(1), test_X:size(2), test_X:size(3), 3*test_X:size(4))
local function append_feat_with_other_classes()  
  local randi = torch.rand(2*train_T:size(1)) * train_noise_ind:size(1)
  local randp = torch.rand(train_T:size(1)) * 3
  for i = 1, train_T:size(1) do
    local new_feat = nil
    local rand1 = math.ceil(randi[2*i-1])
    local rand2 = math.ceil(randi[2*i])
    local randpi = math.ceil(randp[i])
    if rand1 == rand2 then
      rand1 = rand2+1
    end
    if randpi == 1 then
      new_feat = torch.cat({train_X[i],train_noise_X[rand1], train_noise_X[rand2]}, 3)
    elseif randpi == 2 then
      new_feat = torch.cat({train_noise_X[rand1], train_X[i], train_noise_X[rand2]}, 3)
    else
      new_feat = torch.cat({train_noise_X[rand1], train_noise_X[rand2], train_X[i]}, 3)
    end
    train_X_new[i] = new_feat
  end
  
  randi = torch.rand(2*test_T:size(1)) * test_noise_ind:size(1)
  randp = torch.rand(test_T:size(1)) * 3
  for i = 1, test_T:size(1) do
    local new_feat = nil
    local rand1 = math.ceil(randi[2*i-1])
    local rand2 = math.ceil(randi[2*i])
    local ranpi = math.ceil(randp[i])
    if rand1 == rand2 then
      rand1 = rand2+1
    end
    if ranpi == 1 then
      new_feat = torch.cat({test_X[i],test_noise_X[rand1], test_noise_X[rand2]}, 3)
    elseif ranpi == 2 then
      new_feat = torch.cat({test_noise_X[rand1], test_X[i], test_noise_X[rand2]}, 3)
    else
      new_feat = torch.cat({test_noise_X[rand1], test_noise_X[rand2], test_X[i]}, 3)
    end
    test_X_new[i] = new_feat
  end
  
end
append_feat_with_other_classes()

image.display({image = train_X_new:narrow(1,5,5)})
print(train_T:narrow(1,5,5))

local data = {}
data.train_X = train_X_new
data.train_T = train_T
data.test_X = test_X_new
data.test_T = test_T
data.validation_X = test_X_new
data.validation_T = test_T
local output = string.format('../../data/mnist/mnist_distort_noise.t7')
torch.save(output, {train_X = data.train_X, validation_X = data.validation_X, 
  test_X = data.test_X, train_T = data.train_T, validation_T = data.validation_T, test_T = data.test_T})
print('done!')








