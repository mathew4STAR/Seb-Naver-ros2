#include "terrain_analyzer/TerrainAnalyzer.hpp"
#include <rclcpp/rclcpp.hpp>

using namespace terrain_analyzer;

int main( int argc, char * argv[] )
{ 
    rclcpp::init(argc, argv);
    auto node = std::make_shared<rclcpp::Node>("terrain_analyzer_node");

    TerrainAnalyzer terrain_analyzer;
    terrain_analyzer.init(node);

    rclcpp::spin(node);
    rclcpp::shutdown();

    return 0;
}
