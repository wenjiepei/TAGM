
--[[

This is the code to calculate the AUC and EER(equal error rate).
temporarily this code only supoorts two-class problem.

Copyright (c) 2015 Wenjie Pei
Delft University of Technology 

]]--

require 'torch'
require 'gnuplot'

local AUC_EER = {}

local function plot_AUC(fpr, tpr)
  if not torch.isTensor(fpr) then
    fpr = torch.Tensor(fpr)
  end
  if not torch.isTensor(tpr) then
    tpr = torch.Tensor(tpr)
  end
  
  gnuplot.figure()
  gnuplot.title('AUC curve')
  gnuplot.plot(fpr, tpr, '-')
  gnuplot.xlabel('False positive rate')
  gnuplot.ylabel('True positive rate')
  gnuplot.plotflush()
end

local function plot_EER()


end

---return the AUC value
--input 'true_labels': the true_label vector (a torch vector) for samples
--input 'scores': the score for each sample (a torch vector)
--input 'pos_class': the positive class label (a numerical value)
function AUC_EER.calculate_AUC(true_labels, scores, pos_class)
  assert(true_labels:size(1) == scores:size(1), 'error: true_labels and scores have different size. ' )
  local sample_size = true_labels:size(1)
  local sorted_scores, ind = torch.sort(scores)
  local sorted_labels = true_labels:index(1, torch.LongTensor(ind)) 
  local positive_n = torch.sum(true_labels:eq(pos_class)) -- positive number
  local negative_n = sample_size - positive_n 
  local auc_v = 0

  --- maybe there is better way to do it in torch.
  local function filter_value(v, iv)
    local tp_n = torch.sum(sorted_labels:sub(iv, sample_size):eq(pos_class))
    local fp_n =  sample_size - iv + 1 - tp_n
    local tp_rate = tp_n / positive_n
    local fp_rate = fp_n / negative_n
    return fp_rate, tp_rate
  end

  local previous_v = -1
  local previous_fpr = 0
  local previous_tpr = 0
  local fpr_all = {}
  local tpr_all = {}
  for i = 1, sample_size do
    local v = sorted_scores[i]
    assert(previous_v <= v, 'error: previous_v > v')
    if v ~= previous_v then
      local fpr, tpr = filter_value(v, i)
      fpr_all[#fpr_all+1] = fpr
      tpr_all[#tpr_all+1] = tpr
      if i > 1 then
        if previous_fpr == fpr then
          assert(tpr <= previous_tpr, 'error: tpr <= previous_tpr')
--          tpr = previous_tpr
        else
          local area = (previous_fpr-fpr) * (tpr + previous_tpr) / 2.0 
          auc_v = auc_v + area
        end
      end
      previous_fpr = fpr
      previous_tpr = tpr
      previous_v = v
    end
  end
--  print('fpr:' ,'tpr:')
--  for i = 1, #fpr_all do
--    print(fpr_all[i], tpr_all[i])
--  end
  
  return auc_v, fpr_all, tpr_all
end

--- return the EER value and AUC value
function AUC_EER.calculate_EER_AUC(true_labels, scores, pos_class, if_plot_auc)
  local auc_v, fpr_all, tpr_all = AUC_EER.calculate_AUC(true_labels, scores, pos_class)
  if if_plot_auc then
    plot_AUC(fpr_all, tpr_all)
  end
  local FRR = -(torch.Tensor(tpr_all) - 1)
  local FAR = torch.Tensor(fpr_all)
  local EER = 0
  local temp = torch.sum(FRR:le(FAR))
  if temp < FRR:size(1) then
    if (FAR[temp]-FRR[temp])<= (FRR[temp+1]-FAR[temp+1]) then
      EER=(FAR[temp]+FRR[temp])/2
    else
      EER=(FRR[temp+1]+FAR[temp+1])/2
    end
  else
    EER=(FAR[temp]+FRR[temp])/2
  end
  return EER, auc_v
end 

-- for test
local function try_experiment()
  local scores = torch.Tensor({0.5, 0.6, 0.4, 0.3, 0.7, 0.2, 0.1, 0.9})
  local true_labels = torch.Tensor({1, 1, 0, 1, 1, 0, 0, 0})
--  local scores = torch.Tensor({2, 2, 3, 4})
--  local true_labels = torch.Tensor({1, 2, 2, 1})
  local pos_class = 1
--  local auc_v = AUC_EER.calculate_AUC(true_labels,scores,pos_class)
  local eer_v, new_auc = AUC_EER.calculate_EER_AUC(true_labels,scores,pos_class, 1)
--  print('auc_v: ', auc_v)
  print('EER_v: ', eer_v, 'new_auc: ', new_auc)
end

--try_experiment()

return AUC_EER
