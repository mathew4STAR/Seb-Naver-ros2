#include <pluginlib/class_list_macros.hpp>
#include "terrain_analyzer/post_processors/post_processors.hpp"

PLUGINLIB_EXPORT_CLASS(terrain_analyzer::MeanInRadiusFilter, terrain_analyzer::PostProcessor<se2_grid::SE2Grid>)
PLUGINLIB_EXPORT_CLASS(terrain_analyzer::InpaintProcessor, terrain_analyzer::PostProcessor<se2_grid::SE2Grid>)
PLUGINLIB_EXPORT_CLASS(terrain_analyzer::MathProcessor, terrain_analyzer::PostProcessor<se2_grid::SE2Grid>)
PLUGINLIB_EXPORT_CLASS(terrain_analyzer::CurvatureComputer, terrain_analyzer::PostProcessor<se2_grid::SE2Grid>)
PLUGINLIB_EXPORT_CLASS(terrain_analyzer::NormalComputer, terrain_analyzer::PostProcessor<se2_grid::SE2Grid>)
PLUGINLIB_EXPORT_CLASS(terrain_analyzer::SignedDistanceField, terrain_analyzer::PostProcessor<se2_grid::SE2Grid>)
