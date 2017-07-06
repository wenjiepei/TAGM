
--[[
Some basic table operations by Wenjie Pei

]]


local table_operation = {}

function table_operation.subrange(t, first, last)
  local sub = {}
--  print(#t)
  for i=first,last do
    sub[#sub + 1] = t[i]
  end
  return sub
end

--function table_operation.copyTable(st)  
--    local tab = {}  
--    for k, v in pairs(st or {}) do  
--        if type(v) ~= "table" then  
--            tab[k] = v  
--        else  
--            tab[k] = table_operation.copyTable(v)  
--        end  
--    end  
--    return tab  
--end

function table_operation.shallowCopy(orig)
    local orig_type = type(orig)
    local copy
    if orig_type == 'table' then
        copy = {}
        for orig_key, orig_value in pairs(orig) do
            copy[orig_key] = orig_value
        end
    else -- number, string, boolean, etc
        copy = orig
    end
    return copy
end

function table_operation.deepCopy(orig)
    local orig_type = type(orig)
    local copy
    if orig_type == 'table' then
        copy = {}
        for orig_key, orig_value in next, orig, nil do
            copy[table_operation.deepCopy(orig_key)] = table_operation.deepCopy(orig_value)
        end
        setmetatable(copy, table_operation.deepCopy(getmetatable(orig)))
    else -- number, string, boolean, etc
        copy = orig
    end
    return copy
end 

--- split the input table t into two separate tables: 
-- return table1 with the elements with index1 in t
-- return table 2 with the left elements in t 
-- index: torch.Tensor
function table_operation.splitTable(t, index)
  local sorted_index = torch.sort(index)
  local table1 = {}
  local table2 = {}
  local ci = 1
  for k, v in pairs(t) do
    if ci <= sorted_index:nElement() and k == sorted_index[ci] then
      table1[#table1+1] = v
      ci = ci + 1
    else
      table2[#table2+1] = v
    end
  end
  return table1, table2
end

--- return the subtable with the index in the table 't'
function table_operation.subTable(t, index)
  local rtn_t = {}
  local ci = 1
  for i = 1, index:nElement() do
    rtn_t[#rtn_t+1] = t[index[i]]
  end

  return rtn_t
end

function try()
  local t = {a = 1, b = 2}
  local c = table_operation.shallowCopy(t)
  for k, v in pairs(c) do
    print(k, v)
  end
  
  local tt = {t = t}
  local ttt = {tt =tt, t = t}
  c = table_operation.deepCopy(ttt)
  local function display_table(t)
    for k, v in pairs(t) do
      if type(v) ~= 'table' then
        print(k, v)
      else
        print(k)
        display_table(v)
      end
    end
  end
  display_table(c)
end

--try()

return table_operation