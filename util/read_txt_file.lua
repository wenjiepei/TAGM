
-----
-- the code is just a sample, not correct by syntax
local function read_txt_file()

  local index_file = path.join(opt.fold_dir, 'fold_all_' .. i .. '.txt')
  local indexf = io.open(index_file)
  while true do
    local line = indexf:read()
    if line == nil then break end
    --      print(line)
    if line:len()>0 and line:byte(1)>47 and line:byte(1)<58 then
      if i == opt.test_fold_index then -- for test data
        test_name_str[#test_name_str+1] = line
        --        elseif i == opt.validation_fold_index then -- for validation data
        --          validation_name_str[#validation_name_str+1] = line
      else
        train_name_str[#train_name_str+1] = line
      end
    end
  end
end
