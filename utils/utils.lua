json = require 'json'

function table_to_file(path, t)
    -- Serialize table
    local t_s = json.encode(t)

    -- Save
    local f = io.open(path, 'w')
    f:write(t_s)
    f:close()
end
