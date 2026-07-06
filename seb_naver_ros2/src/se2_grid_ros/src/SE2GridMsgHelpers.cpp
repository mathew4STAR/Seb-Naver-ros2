#include "se2_grid_ros/SE2GridMsgHelpers.hpp"

#include <map>
#include <string>

namespace se2_grid {

int nDimensions()
{
  return 2;
}

std::map<StorageIndices, std::string> storageIndexNames = {
    {StorageIndices::Column, "column_index"},
    {StorageIndices::Row, "row_index"}
};

} /* namespace */
