--[[

This is the implementation of the Temporal Attention-Gated Model (TAGM) for robust sequence classification.

Copyright (c) 2016 Wenjie Pei
Delft University of Technology 

]]--

require 'torch'
require 'nn'
require 'nngraph'
require 'lfs'
local path = require 'pl.path'

local train_process = require 'train_process'
local evaluate_process = require 'evaluate_process'

local cmd = torch.CmdLine()
cmd:text()
cmd:text('TAGM for sequence classification.')
cmd:text()
cmd:text('Options')
--- data
cmd:option('-data_dir','../../data/','data directory.')
cmd:option('-fold_dir','../../data') -- the fold name used in the experiments
cmd:option('-data_set', 'arabic_append_noise') -- options: 'arabic_voice', 'arabic_append_noise', 'SST_2', 'SST_5', 'SST_2_train_all' , 'SST_5_train_all' 
cmd:option('-result_dir','result','result directory.')

--- model params
cmd:option('-att_rnn_size', 64, 'size of LSTM internal state (for attention model)')
cmd:option('-if_attention', 1, 'if use attention model')
cmd:option('-attention_sig_w', 3, 'to validate the learning rate for attention weights, options: {1, 3, 5}, normally 3 is a good option.')
cmd:option('-attention_num_layers', 1, 'number of layers in the attention model')
cmd:option('-att_model', 'rnn', 'lstm or rnn, we use rnn in the paper')

--- optimization
cmd:option('-opt_method', 'rmsprop', 'the optimization method with options: 1. "rmsprop"  2. "gd" (exact gradient descent)')
cmd:option('-learning_rate',8e-3,'learning rate')
cmd:option('-learning_rate_decay',0.5,'learning rate decay')  
cmd:option('-learning_rate_decay_after',0,'in number of epochs, when to start decaying the learning rate')
cmd:option('-decay_rate',0.95,'decay rate for rmsprop')
cmd:option('-dropout',0.0,'dropout for regularization, used after each RNN hidden layer. 0 = no dropout')
cmd:option('-batch_size',32,'number of sequences to train on for gradient descent each time')
cmd:option('-max_epochs',200,'number of full passes through the training data')
cmd:option('-max_iterations',100000,'max iterations to run')                                              
cmd:option('-grad_clip',10,'clip gradients at this value')
cmd:option('-blowup_threshold',1e4,'the blowup threshold')
cmd:option('-check_gradient', false, 'whether to check the gradient value') 
cmd:option('-do_random_init', true, 'whether to initialize the parameters manually') 
cmd:option('-stop_iteration_threshold', 24,
       'if better than the later @ iterations , then stop the optimization')
cmd:option('-decay_threshold', 6, 'if better than the later @ iterations , then decay the learning rate')
cmd:option('-if_init_from_check_point', false, 'initialize network parameters from checkpoint at this path')
cmd:option('-if_direct_test_from_scratch', false)
cmd:option('-if_output_step_test_error', true)

--- for top_classifier
cmd:option('-top_c', 'TAGM', 'the top classifier: NN, or lstm, or rnn, or gru, or TAGM')
cmd:option('-if_original_feature', 1, 'if use original feature orelse hidden value for attention model')
cmd:option('-top_lstm_size', 64, 'size of LSTM internal state (for top lstm model)')
cmd:option('-top_num_layers', 1, 'number of layers in the top LSTM')
cmd:option('-top_bidirection', 0, 'if use bidirection on top toc_c')

--- bookkeeping
cmd:option('-seed',123,'torch manual random number generator seed')
cmd:option('-print_every',1,'how many steps/minibatches between printing out the loss')
cmd:option('-evaluate_every',3072,'how many samples between evaluate the whole data set')
cmd:option('-savefile','current_model','filename to autosave the checkpont to. Will be inside checkpoint_dir/')
cmd:option('-disp_image', 1, 'if display image from the stn output')
cmd:option('-w1', 1, 'for disp_image window')
cmd:option('-w2', 1, 'for disp_image window')

--- GPU/CPU
-- currently, only supports CUDA
cmd:option('-gpuid',-1,'which gpu to use. -1 = use CPU')
cmd:text()

-- parse input params                                                                                                                                                                                                                                                                                                                                                                                                                                                   
local opt = cmd:parse(arg)

if opt.data_set == 'arabic_append_noise' then
  opt.batch_size = 32
  opt.evaluate_every = 3072
  if opt.top_c == 'TAGM' then
    opt.learning_rate = 2e-3
  else
    opt.learning_rate = 1e-2
  end
  elseif opt.data_set == 'SST_2' or opt.data_set == 'SST_5' then
  opt.batch_size = 32
  opt.evaluate_every = 1600
  if opt.top_c == 'TAGM' then
    opt.learning_rate = 8e-3
  else
    opt.learning_rate = 1e-2
  end
  elseif opt.data_set == 'SST_2_train_all' or opt.data_set == 'SST_5_train_all' then
  opt.batch_size = 64
  opt.evaluate_every = 6400
  if opt.top_c == 'TAGM' then
    opt.learning_rate = 8e-3
  else
    opt.learning_rate = 1e-2
  end
end

-- decrease the learning rate proportionally
local factor = 32 / opt.top_lstm_size
opt.learning_rate = opt.learning_rate * factor

if opt.gpuid < 0 then                                                                                                   
  print('Perform calculation by CPU using the optimization method: ' .. opt.opt_method)                                         
else
  print('Perform calculation by GPU with OpenCL using the optimization method: ' .. opt.opt_method)
end
print('the model type is: ', opt.model)

torch.manualSeed(opt.seed)

-- about disp_image
if opt.disp_image then
  opt.w1 = nil
  opt.w2 = nil
end

  --------- initialize cunn/cutorch for training on the GPU and fall back to CPU gracefully ------
  ------------------------------------------------------------------------------------------------
  if opt.gpuid >= 0 then
    local ok, cunn = pcall(require, 'cunn')
    local ok2, cutorch = pcall(require, 'cutorch')
    if not ok then print('package cunn not found!') end
    if not ok2 then print('package cutorch not found!') end
    if ok and ok2 then
      print('using CUDA on GPU ' .. opt.gpuid .. '...')
      cutorch.setDevice(opt.gpuid + 1) -- note +1 to make it 0 indexed! sigh lua
      cutorch.manualSeed(opt.seed)
    else
      print('If cutorch and cunn are installed, your CUDA toolkit may be improperly configured.')
      print('Check your CUDA toolkit installation, rebuild cutorch and cunn, and try again.')
      print('Falling back on CPU mode')
      opt.gpuid = -1 -- overwrite user setting
    end
  end

-- make sure that output directory exists
if not path.exists(opt.result_dir) then lfs.mkdir(opt.result_dir) end
local current_result_dir = path.join(opt.result_dir, opt.data_set)
if not path.exists(current_result_dir) then lfs.mkdir(current_result_dir) end
current_result_dir = path.join(current_result_dir, 'if-use-attention_' .. opt.if_attention ..
  '_attention_' .. opt.att_model .. '_top-c_' .. opt.top_c)
if not path.exists(current_result_dir) then lfs.mkdir(current_result_dir) end
current_result_dir = path.join(current_result_dir, 'att-rnn-size_' .. opt.att_rnn_size .. '_att-layers_' .. opt.attention_num_layers .. 
'_top-c-rnn-size_' .. opt.top_lstm_size .. 
  '_sigW_' .. opt.attention_sig_w .. '_top-layers_' .. opt.top_num_layers .. '_top-bidirection_' .. opt.top_bidirection)
if not path.exists(current_result_dir) then lfs.mkdir(current_result_dir) end
opt.current_result_dir = current_result_dir  

if opt.if_direct_test_from_scratch then
  evaluate_process.evaluate_from_scratch(opt, true)
else
  -- begin to train the model
  print('Begin to train the model...')
  train_process.train(opt)
  print("Training Done!")
  torch.manualSeed(opt.seed)
  opt.if_direct_test_from_scratch = true
  evaluate_process.evaluate_from_scratch(opt, true)
end
