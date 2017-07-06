
require 'image'

local path = require 'pl.path'
local AUC_EER = require 'util/my_AUC_EER_calculation'
require 'util.misc'
local data_loader = require 'util.data_loader'
local model_utils = require 'util.model_utils'
local define_my_model = require 'model.define_my_model'
local table_operation = require 'util/table_operation'

local evaluate_process = {}

--- preprocessing helper function
local function prepro(opt, x)
  if opt.gpuid >= 0 and opt.opencl == 0 then -- ship the input arrays to GPU
    x = x:float():cuda()
  end
  return x
end

local function image_display_batch(model, opt, x, set_class, batch_ind, attention_weight)
  local weight_images
  if opt.if_attention == 0 then
    weight_images = attention_weight
    weight_images:resize(weight_images:size(1), 1, weight_images:size(2), weight_images:size(3))
  else
    weight_images = torch.zeros(x:size())
    for i = 1, attention_weight:size(1) do
      for j = 1, attention_weight:size(2) do
        local v = math.ceil(attention_weight[i][j] * x:size(3))
        local b = math.min(x:size(3)-v+1, x:size(3))
        weight_images[i][1]:sub(b, -1, j, j):fill(1)
      end
    end
  end
  local function image_scale(samples)
    local rtn_images = {}
    for k = 1, samples:size(1) do
      local img 
      if samples:dim() == 4 then
        img = samples[k]:contiguous():view(1, samples:size(3), samples:size(4))
      else
        img = samples[k]:contiguous():view(1, samples:size(2), samples:size(3))
      end
      rtn_images[#rtn_images+1] = image.scale(img,'*2', 'simple')
    end
    return rtn_images
  end
  local ss = math.min(16, x:size(1))
  local weight_output = image_scale(weight_images:narrow(1,1,ss))
  local original = image_scale(x:narrow(1,1,ss))

  opt.w1=image.display({image=weight_output, nrow=4, legend='Attention weights', win = opt.w1})
  opt.w2=image.display({image=original, nrow=4, legend='Inputs', win = opt.w2})
  if opt.if_direct_test_from_scratch then
    -- top image is original image while bottom image is the segmented image
    local function image_align(original_image, stn_image)
      local img_h = original_image:size(2) + stn_image:size(2)
      local img_w = math.max(original_image:size(3), stn_image:size(3))
      local img = torch.ones(1, img_h+10, img_w):fill(0.5)
      local larg = math.max(original_image:max(), stn_image:max())
      local smal = math.min(original_image:min(), stn_image:min())
      img:sub(1, 1, 1, original_image:size(2), 1, original_image:size(3)):copy(original_image:csub(smal):div(larg-smal))
      img:sub(1, 1, original_image:size(2)+11, original_image:size(2)+stn_image:size(2)+10, 
        1, stn_image:size(3)):copy(stn_image:csub(smal):div(larg-smal))
      return img
    end
    if not path.exists(path.join(opt.current_result_dir, 'image')) then lfs.mkdir(path.join(opt.current_result_dir, 'image')) end
    for i = 1, weight_images:size(1) do
      local ind = (batch_ind-1) * opt.batch_size + i
      weight_images[i]:mul(x[i]:max()-x[i]:min()):add(x[i]:min())
      local img = image_align(x[i], weight_images[i])
      img = image.scale(img, '*2', 'simple')
      image.save(opt.current_result_dir .. '/image/' .. set_class .. ind .. '.png', img)
    end
  end
  --    sys.sleep(1000)

end

local function save_SST_weight(set_name, all_weights, predictions, opt, loader)
  if not path.exists(path.join(opt.current_result_dir, 'weights')) then lfs.mkdir(path.join(opt.current_result_dir, 'weights')) end
  local temp_file = io.open(string.format( '%s/weights/%s_weight.txt', opt.current_result_dir, set_name), 'w')
  local nn, sentences, scores
  if set_name == 'train' then
    nn = loader.nTrain
    sentences = loader.train_sentences
    scores = loader.train_score
  elseif set_name == 'validation' then
    nn = loader.nValidation
    sentences = loader.validation_sentences
    scores = loader.validation_score
  else
    nn = loader.nTest
    sentences = loader.test_sentences
    scores = loader.test_score
  end
  for i = 1, nn do
    local sen = sentences[i]
    local sen_len = 0
    for word in string.gmatch(sen,"%S+") do
      sen_len = sen_len + 1
    end
    if sen_len ~= all_weights[i]:nElement() then
      print(i .. '   error: in-equal length: sens_len: '.. sen_len .. '  weights: ' .. all_weights[i]:nElement())
      print(sen)
    else
      temp_file:write(string.format('%-10d, scores: %-10f  prediction: %-3d \n', i, scores[i], predictions[i]))
      temp_file:write(sen, '\n')
      local k = 1
      local w_s = ""
      for word in string.gmatch(sen,"%S+") do
--        -- one way
--        temp_file:write(string.format('%20s %20f\n', word,  all_weights[i][k]))
        -- the other way
        if word:len() >= 5 then
          temp_file:write(string.format('%s ', word))
          w_s = w_s .. string.format('%1.2f', all_weights[i][k])
          for m = 1, word:len()-3 do
            w_s = w_s .. ' '
          end
        else
          temp_file:write(string.format('%-6s', word))
          w_s = w_s .. string.format('%1.2f  ', all_weights[i][k])
        end
        k = k+1
      end
      temp_file:write(string.format('\n%s\n', w_s))
      temp_file:write('\n\n')
    end
  end
  temp_file:close()


end

--- inference one sample
local function inference(model, x, true_y, opt)
  -- decode the model and parameters
  local attention = model.attention
  local top_net = model.top_net
  local criterion = model.criterion 
  local params_flat = model.params_flat
  local x_length = x:size(2)

  -- perform the forward pass for attention model
  local attention_weights, hidden_z_value
  if opt.if_attention == 1 then
    attention_weights, hidden_z_value = attention.forward(x, opt, 'test')
  else
    attention_weights = torch.ones(1, x_length)
  end
  -- perform the forward for the top-net module
  local net_output = nil
  if opt.top_c == 'NN' then
    if opt.if_original_feature == 1 then
      net_output = top_net:forward({x, attention_weights})
    else
      net_output = top_net:forward({hidden_z_value, attention_weights})
    end
  elseif opt.top_c == 'lstm' or opt.top_c == 'rnn' or opt.top_c == 'gru' or opt.top_c == 'TAGM' then
    net_output = top_net.forward(x, attention_weights, opt, 'test')
  else
    error('no such top classifier!')
  end
  --compute the loss
  --  local current_loss = criterion:forward(net_output, torch.Tensor({true_y})) -- for batch_size == 1
  local current_loss = criterion:forward(net_output, true_y)
  local _, pred_label = net_output:squeeze():max(1)

  if opt.if_attention == 0 and opt.top_c == 'lstm' then
    attention_weights:resize(1, opt.top_lstm_size, x_length)
    local max_v = torch.max(attention_weights)
    attention_weights:div(max_v)
  end

  return current_loss, pred_label:squeeze(), attention_weights
end

--- input @data_set is a data sequence (table of Tensor) to be evaluated
local function evaluation_set_performance(opt, model, data_sequence, true_labels, if_test, set_name, loader)
  local total_loss_avg = 0
  local accuracy = 0
  local data_size = true_labels:size(1)
  local batch_size = opt.batch_size
  local temp_idx = 1
  local cc = 1
  local all_attention_weights = {}
  local predictions = torch.zeros(data_size)
  for i = 1, data_size do 
    local x, true_y
    x = data_sequence[i]
    if x:dim()==3 then
      x = x:view(x:size(2), x:size(3))
    end
    true_y = true_labels[i]
    x = prepro(opt, x)
    if opt.gpuid >= 0 and opt.opencl == 0 then
      true_y = true_y:float():cuda()
    end
    local temp_loss, predict_label, attention_weights = inference(model, x, true_y, opt)
    all_attention_weights[#all_attention_weights+1] = attention_weights:clone()
    total_loss_avg = temp_loss + total_loss_avg
    if predict_label == true_y then
      accuracy = accuracy + 1
      predictions[i] = 1
    end
    if i % 200 == 0 then
      print(i, 'finished!')
    end
    --    if opt.if_direct_test_from_scratch or (cc==1) then
    ----      image_display_batch(model, opt, x, set_name, cc, attention_weights)
    --    end
  end
  total_loss_avg = total_loss_avg / data_size
  accuracy = accuracy / data_size * 100
  if opt.if_direct_test_from_scratch and opt.top_c == 'TAGM' then
    if opt.data_set:sub(1,3) == 'SST' then
      if not (opt.data_set:len() > 5 and set_name == 'train') then
        save_SST_weight(set_name, all_attention_weights, predictions, opt, loader)
      end
    end
  end
  return total_loss_avg, accuracy
end

--- evaluate the data set
function evaluate_process.evaluate_set(set_name, opt, loader, model, if_plot)
  print('start to evaluate the whole ' .. set_name .. ' set...')
  local timer = torch.Timer()
  local time_s = timer:time().real
  if not if_plot then
    if_plot = false
  end
  local total_loss_avg = nil
  local accuracy = nil
  if set_name == 'train' then
    total_loss_avg, accuracy = evaluation_set_performance(opt, model,
      loader.train_X,loader.train_T, false, set_name, loader)
    --      image_display(model, opt, loader.train_X, 'train')
  elseif set_name == 'validation' then
    total_loss_avg, accuracy = evaluation_set_performance(opt, model,
      loader.validation_X,loader.validation_T, false, set_name, loader)
    --      image_display(model, opt, loader.validation_X, 'validation')
  elseif set_name == 'test' then
    total_loss_avg, accuracy = evaluation_set_performance(opt, model,
      loader.test_X,loader.test_T, true, set_name, loader)
    --      image_display(model, opt, loader.test_X, 'test')
  else
    error('there is no such set name!')
  end 
  local time_e = timer:time().real
  print('total average loss of ' .. set_name .. ' set:', total_loss_avg)
  print('accuracy: ', accuracy)
  print('elapsed time for evaluating the ' .. set_name .. ' set:', time_e - time_s)
  return total_loss_avg, accuracy
end

--- load the data and the trained model from the check point and evaluate the model
function evaluate_process.evaluate_from_scratch(opt, if_train_validation)

  ------------------- create the data loader class ----------
  local loader = data_loader.create(opt)
  local feature_dim = loader.feature_dim
  local do_random_init = true

  ------------------ begin to define the whole model --------------------------
  local model = define_my_model.define_model(opt, loader, true)
  define_my_model.load_model(opt,model, false)
  local if_plot = false
  ------------------- create the data loader class ----------
  print('evaluate the model from scratch...')
  local train_loss, train_accuracy = nil
  local validation_loss, validation_accuracy = nil
  if if_train_validation then 
    train_loss, train_accuracy = evaluate_process.evaluate_set('train', opt, loader, model, false)
    validation_loss, validation_accuracy = evaluate_process.evaluate_set('validation', opt, loader, model, false)
  end
  local test_loss, test_accuracy = evaluate_process.evaluate_set('test', opt, loader, model, true)

  local temp_file = io.open(string.format('%s/%s_results_GPU_%d_dropout_%1.2f.txt', 
    opt.current_result_dir, opt.opt_method, opt.gpuid, opt.dropout), "a")
  temp_file:write(string.format('similarity measurement results \n'))
  if if_train_validation then
    temp_file:write(string.format('train set loss = %6.8f, train accuracy= %6.8f\n', 
      train_loss, train_accuracy ))
    temp_file:write(string.format('validation set loss = %6.8f, validation accuracy = %6.8f\n', 
      validation_loss, validation_accuracy ))
  end
  temp_file:write(string.format('test set loss = %6.8f, test accuracy = %6.8f\n', 
    test_loss, test_accuracy ))

  if if_train_validation then
    return train_accuracy, validation_accuracy, test_accuracy
  else
    return test_accuracy
  end
end

--- for the gradient check
function evaluate_process.grad_check(model, x, true_y, opt)
  -- decode the model and parameters
  if opt.if_attention == 0 then
    model.attention.params_size = 1
  end
  
  local attention_params_flat = model.params_flat:sub(1, model.attention.params_size)
--  local attention_top_params_flat = model.params_flat:sub(model.attention.params_size-opt.rnn_size*2, model.attention.params_size)
--  local attention_grad_top_params_flat = model.grad_params_flat:sub(model.attention.params_size-opt.rnn_size*2, model.attention.params_size)
  local attention_grad_params_flat = model.grad_params_flat:sub(1, model.attention.params_size)
  local top_net_params_flat = model.params_flat:sub(model.attention.params_size+1,  -1)
  local top_net_grad_flat = model.grad_params_flat:sub(model.attention.params_size+1, -1)
  local total_params = model.params_size
  local function calculate_loss()
    local current_loss = inference(model, x, true_y, opt)
    return current_loss
  end  

  local function gradient_compare(params, grad_params)
    local check_number = math.min(200, params:nElement())
    local loss_minus_delta, loss_add_delta, grad_def
    if opt.gpuid >= 0 then
      loss_minus_delta = torch.CudaTensor(check_number)
      loss_add_delta = torch.CudaTensor(check_number)
      grad_def = torch.CudaTensor(check_number)
    else
      loss_minus_delta = torch.DoubleTensor(check_number)
      loss_add_delta = torch.DoubleTensor(check_number)
      grad_def = torch.DoubleTensor(check_number)    
    end
    local params_backup = params:clone()
    local rand_ind = torch.randperm(params:nElement())
    rand_ind = rand_ind:sub(1, check_number)
    for k = 3, 8 do
      local delta = 1 / torch.pow(1e1, k)
      print('delta:', delta)
      for i = 1, check_number do
        local ind = rand_ind[i]
        params[ind] = params[ind] - delta
        loss_minus_delta[i] = calculate_loss() 
        params[ind] = params[ind] + 2*delta
        loss_add_delta[i] = calculate_loss()
        local gradt = (loss_add_delta[i] - loss_minus_delta[i]) / (2*delta)
        grad_def[i] = gradt
        params[ind] = params[ind] - delta -- retore the parameters
        if i % 100 ==0 then
          print(i, 'processed!')
        end
      end
      params:copy(params_backup) -- retore the parameters
      local grad_model = grad_params:index(1, rand_ind:long())
      local if_print = true
      local threshold = 1e-4
      local inaccuracy_num = 0
      local reversed_direction = 0
      assert(grad_def:nElement()==grad_model:nElement())
      local relative_diff = torch.zeros(grad_def:nElement())
      relative_diff = torch.abs(grad_def - grad_model)
      relative_diff:cdiv(torch.cmax(torch.abs(grad_def), torch.abs(grad_model)))
      for i = 1, grad_def:nElement() do
        if if_print then
          print(string.format('index: %4d, rand_index: %4d, relative_diff: %6.5f,  gradient_def: %6.25f,  grad_model: %6.25f',
            i, rand_ind[i], relative_diff[i], grad_def[i], grad_model[i]))
        end
        if relative_diff[i] > threshold then
          if math.max(math.abs(grad_def[i]), math.abs(grad_model[i])) > 1e-8 then
            inaccuracy_num = inaccuracy_num + 1
          end   
        end
      end
      for i = 1, grad_def:nElement() do
        if grad_def[i] * grad_model[i] < 0 then
          if if_print then
            print(string.format('index: %4d, relative_diff: %6.5f,  gradient_def: %6.10f,  grad_params: %6.10f',
              i, relative_diff[i], grad_def[i], grad_model[i]))
          end
          reversed_direction = reversed_direction + 1
        end
      end

      print('there are', inaccuracy_num, 'inaccuracy gradients.')
      print('there are', reversed_direction, 'reversed directions.')
    end
  end


--     check rnn params
  gradient_compare(attention_params_flat, attention_grad_params_flat)  
--          gradient_compare(attention_top_params_flat, attention_grad_top_params_flat)  
--  --  --   check top_net params
--          gradient_compare(top_net_params_flat, top_net_grad_flat)


end

return evaluate_process

