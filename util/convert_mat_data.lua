
local convert_mat_t7 = require 'util/convert_mat_to_t7_data'
local path = require 'pl.path'


local function dirLookup(dir)
  local all_files = {}
   local p = io.popen('find "'..dir..'" -type f')  --Open directory look for files, save data in p. By giving '-type f' as parameter, it returns all files.     
   for file in p:lines() do                         --Loop through all files
       print(file)
       all_files[#all_files+1] = file       
   end
   return all_files
end


local function convert_TiSSiLe_mat_to_t7()
--  local data_name = {'frontal_mean_lbp_125x105_7x5'}
  local data_name = {'frontal_mean_warped_cnn'}
--    local data_name = {'frontal_normalized_lmsDyn_49_wrt_eye'}
--  local data_name = {'frontal_normalized_lms_wrt_eye'}
  for i = 1, #data_name do
    local pathd = path.join('../../data', data_name[i], 'mat_format' )
    local all_files = dirLookup(pathd)
    local out_dir = path.join('../../data', data_name[i], 't7_format' )
    if not path.exists(out_dir) then lfs.mkdir(out_dir) end
    for i = 1, #all_files do
      convert_mat_t7.mat_to_t7_data(all_files[i], pathd, out_dir)
    end
  end

end

convert_TiSSiLe_mat_to_t7()