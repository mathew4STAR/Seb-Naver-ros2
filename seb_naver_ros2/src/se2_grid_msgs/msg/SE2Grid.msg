SE2GridInfo info

# layer names.
string[] layers

# if has SO(2) info.
bool[] has_so2

# Grid map data.
std_msgs/Float32MultiArray[] data

# Row start index (default 0).
uint16 start_index_row

# Column start index (default 0).
uint16 start_index_col
