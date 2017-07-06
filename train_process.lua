-- internal library
local optim = require 'optim'
local path = require 'pl.path'

-- local library
local RNN = require 'model.my_RNN'
local model_utils = require 'util.model_utils'
local data_loader = require 'util.data_loader'
require 'util.misc'
local Top_Net = require 'model/Top_NN_Classifier'
local evaluate_process = require 'evaluate_process'
local table_operation = require 'util/table_operation'
local define_my_model = require 'model/define_my_model'

local train_process = {}

--- process one batch to get the gradients for optimization and update the parameters 
-- return the loss value of one minibatch of samples
local function feval(opt, loader, model, rmsprop_para, iter_count)
  -- decode the model and parameters, 
  -- since it is just the reference to the same memory location, hence it is not time-consuming.

  local attention = model.attention
  local top_net = model.top_net
  local criterion = model.criterion 
  local params_flat = model.params_flat
  local grad_params_flat = model.grad_params_flat
  local params_grad_all_batches = model.params_grad_all_batches

  ---------------------------- get minibatch --------------------------
  ---------------------------------------------------------------------

  local data_index = loader:get_next_train_batch(opt.batch_size)
  local loss_total = 0
  params_grad_all_batches:zero()

  -- Process the batch of samples one by one, since different sample contains different length of time series, 
  -- hence it's not convenient to handle them together
  for batch = 1, opt.batch_size do
    local current_data_index = data_index[batch]
    local x = loader.train_X[current_data_index]
    local true_y = loader.train_T[current_data_index]
    if x:dim() == 3 and x:size(1) == 1 then
      x = x:view(x:size(2), x:size(3))
    elseif x:dim() > 3 then
      error('x:dim > 3')
    end
    local x_length = x:size(2)
    
    ---------------------- forward pass of the whole model -------------------  
    --------------------------------------------------------------------------  

    -- perform the forward pass for attention model
    local attention_weights, hidden_z_value
    if opt.if_attention == 1 then
      attention_weights, hidden_z_value = attention.forward(x, opt, 'training')
    else
      attention_weights = torch.ones(x_length)
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
      net_output = top_net.forward(x, attention_weights, opt, 'training')
    else
      error('no such top classifier!')
    end
    --compute the loss
    --  local current_loss = criterion:forward(net_output, torch.Tensor({true_y})) -- for batch_size == 1
    local current_loss = criterion:forward(net_output, true_y)
    loss_total = loss_total + current_loss


    ---------------------- backward pass of the whole model ---------------------
    -----------------------------------------------------------------------------

    -- peform the backprop on the top_net
    grad_params_flat:zero()
    local grad_net = nil
    if opt.top_c == 'NN' then
      if opt.if_original_feature ==1 then
        grad_net = top_net:backward({x, attention_weights}, criterion:backward(net_output, true_y))
        if opt.if_attention == 1 then
          attention.backward(opt, hidden_z_value, grad_net[2], x)
        end
      else
        grad_net = top_net:backward({hidden_z_value, attention_weights}, criterion:backward(net_output, true_y))
        if opt.if_attention == 1 then
          attention.backward(opt, hidden_z_value, grad_net[2], x, grad_net[1])
        end
      end
    elseif opt.top_c == 'lstm' or opt.top_c == 'rnn' or opt.top_c == 'gru' then
      grad_net = top_net.backward(x, attention_weights, opt, criterion:backward(net_output, true_y), loader)
      if opt.if_attention == 1 then
        attention.backward(opt, hidden_z_value, grad_net[2], x)
      end
    elseif opt.top_c == 'TAGM' then
      grad_net = top_net.backward(x, attention_weights, opt, criterion:backward(net_output, true_y), loader)
      if opt.if_attention == 1 then
        attention.backward(opt, hidden_z_value, grad_net, x)
      end
    else
      error('no such classifier!')
    end

    params_grad_all_batches:add(grad_params_flat)

    -- for gradient check
    if opt.check_gradient and iter_count==1 then
      evaluate_process.grad_check(model, x, true_y, opt)
      print('\n')
      os.exit()
    end
  end
  loss_total = loss_total / opt.batch_size
  -- udpate all the parameters
  params_grad_all_batches:div(opt.batch_size)
  params_grad_all_batches:clamp(-opt.grad_clip, opt.grad_clip)
  if opt.opt_method == 'rmsprop' then
    local function feval_rmsprop(p)
      return loss_total, params_grad_all_batches
    end
    optim.rmsprop(feval_rmsprop, params_flat, rmsprop_para.config)
  elseif opt.opt_method == 'gd' then -- 'gd' simple direct minibatch gradient descent
    params_flat:add(-opt.learning_rate, params_grad_all_batches)
  else
    error("there is no such optimization option!")  
  end

  return loss_total
end

--- major function 
function train_process.train(opt)

  ------------------- create the data loader class ----------
  -----------------------------------------------------------

  local loader = data_loader.create(opt)
  local do_random_init = true

  ------------------ begin to define the whole model --------------------------
  -----------------------------------------------------------------------------
  local model = {}
  model, opt = define_my_model.define_model(opt, loader)

  --------------- start optimization here -------------------------
  -----------------------------------------------------------------
  -- for rmsprop
  local rmsprop_config = {learningRate = opt.learning_rate, alpha = opt.decay_rate}
  local rmsprop_state = {}
  local rmsprop_para = {config = rmsprop_config, state = rmsprop_state}

  local iterations = math.floor(opt.max_epochs * loader.nTrain / opt.batch_size)
  local iterations_per_epoch = math.floor(loader.nTrain / opt.batch_size)
  local train_losses = torch.zeros(iterations)
  local timer = torch.Timer()
  local time_s = timer:time().real
  local epoch = 0
  local better_times_total = 0
  local better_times_decay = 0
  local current_best_acc = 0
  for i = 1, iterations do
    if epoch > opt.max_epochs then break end
    if i>opt.max_iterations then break end
    epoch = i / loader.nTrain * opt.batch_size
    local time_ss = timer:time().real
    -- optimize one batch of training samples
    train_losses[i] = feval(opt, loader, model, rmsprop_para, i)
    local time_ee = timer:time().real
    local time_current_iteration = time_ee - time_ss
    if i % 10 == 0 then collectgarbage() end

    -- handle early stopping if things are going really bad
    local function isnan(x) return x ~= x end
    if isnan(train_losses[i]) then
      print('loss is NaN.  This usually indicates a bug.' .. 
        'Please check the issues page for existing issues, or create a new issue, ' .. 
        'if none exist.  Ideally, please state: your operating system, 32-bit/64-bit, your blas version, cpu/cuda/cl?')
      break -- halt
    end
    -- check if the loss value blows up
    local function is_blowup(loss_v)
      if loss_v > opt.blowup_threshold then
        print('loss is exploding, aborting:', loss_v)
        return true
      else 
        return false
      end
    end
    if is_blowup(train_losses[i]) then
      break
    end

    if i % opt.print_every == 0 then
      print(string.format("%d/%d (epoch %.3f), train_loss = %6.8f, time/batch = %.4fs", 
        i, iterations, epoch, train_losses[i], time_current_iteration))
    end

    if i * opt.batch_size % opt.evaluate_every == 0 then
      local temp_sum_loss = torch.sum(train_losses:sub(i - opt.evaluate_every/opt.batch_size+1, i))
      local temp_mean_loss = temp_sum_loss / opt.evaluate_every * opt.batch_size
      print(string.format('average loss in the last %d iterations = %6.8f', opt.evaluate_every, temp_mean_loss))
      print('learning rate: ', opt.learning_rate)

      local whole_validation_loss,  validation_acc = nil
      if opt.validation_size == 0 then
        local whole_train_loss, train_acc = evaluate_process.evaluate_set('train', opt, loader, model)
        whole_validation_loss = whole_train_loss
        validation_acc = train_acc
      else
        whole_validation_loss,  validation_acc = evaluate_process.evaluate_set('validation', opt, loader, model)
      end
      local whole_test_loss, test_acc 
      if opt.if_output_step_test_error then
        whole_test_loss, test_acc = evaluate_process.evaluate_set('test', opt, loader, model)
      end
      local time_e = timer:time().real
      print(string.format('elasped time in the last %d iterations: %.4fs,    total elasped time: %.4fs', 
        opt.evaluate_every, time_e-time_s, time_e))
      if validation_acc > current_best_acc then
        current_best_acc = validation_acc
        better_times_total = 0
        better_times_decay = 0
        --- save the current trained best model
        define_my_model.save_model(opt, model)
        if validation_acc == 0 then
          break
        end
      else
        better_times_total = better_times_total + 1
        better_times_decay = better_times_decay + 1
        if better_times_total >= opt.stop_iteration_threshold then
          print(string.format('no more better result in %d iterations! hence stop the optimization!', 
            opt.stop_iteration_threshold))
          break
        elseif better_times_decay >= opt.decay_threshold then
          print(string.format('no more better result in %d iterations! hence decay the learning rate', 
            opt.decay_threshold))
          local decay_factor = opt.learning_rate_decay
          rmsprop_config.learningRate = rmsprop_config.learningRate * decay_factor -- decay it
          opt.learning_rate = rmsprop_config.learningRate -- update the learning rate in opt
          print('decayed learning rate by a factor ' .. decay_factor .. ' to ' .. rmsprop_config.learningRate)
          better_times_decay = 0 
          -- back to the currently optimized point
          print('back to the currently best optimized point...')
          model = define_my_model.load_model(opt, model, false)
        end
      end     
      print('better times: ', better_times_total, '\n\n')
      -- save to log file
      local temp_file = nil
      if i  == 1 and not opt.if_init_from_check_point then
        temp_file = io.open(string.format('%s/%s_results_GPU_%d_dropout_%1.2f.txt',
          opt.current_result_dir, opt.opt_method, opt.gpuid, opt.dropout), "w")
      else
        temp_file = io.open(string.format('%s/%s_results_GPU_%d_dropout_%1.2f.txt', 
          opt.current_result_dir, opt.opt_method, opt.gpuid, opt.dropout), "a")
      end
      temp_file:write('better times: ', better_times_total, '\n')
      temp_file:write('learning rate: ', opt.learning_rate, '\n')
      temp_file:write(string.format("%d/%d (epoch %.3f) \n", i, iterations, epoch))
      temp_file:write(string.format('average loss in the last %d (%5d -- %5d) iterations = %6.8f \n', 
        opt.evaluate_every/opt.batch_size, i-opt.evaluate_every/opt.batch_size+1, i, temp_mean_loss))
      --      temp_file:write(string.format('train set loss = %6.8f, train age mean absolute error= %6.8f\n', 
      --       whole_train_loss, differ_avg_train ))
      temp_file:write(string.format('validation set loss = %6.8f, validation accuracy= %6.8f\n', 
        whole_validation_loss, validation_acc ))
      if opt.if_output_step_test_error then
        temp_file:write(string.format('test set loss = %6.8f, test accuracy = %6.8f\n', 
          whole_test_loss, test_acc ))
      end
      temp_file:write(string.format('elasped time in the last %d iterations: %.4fs,    total elasped time: %.4fs\n', 
        opt.evaluate_every, time_e-time_s, time_e))
      temp_file:write(string.format('\n'))
      temp_file:close()
      time_s = time_e
    end
  end
  local time_e = timer:time().real
  print('total elapsed time:', time_e)
end

return train_process
    
    