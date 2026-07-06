#pragma once

#include <typeinfo>
#include <pluginlib/class_loader.hpp>
#include <memory>
#include <sstream>
#include <vector>
#include <map>
#include <rclcpp/rclcpp.hpp>
#include <yaml-cpp/yaml.h>

namespace terrain_analyzer
{
    template<typename T>
    class PostProcessor
    {
    public:
        PostProcessor() : configured_(false) {};

        virtual ~PostProcessor() {};

        bool configure(const YAML::Node& config)
        {
            if (configured_)
            {
                // Note: assuming rclcpp::get_logger("PostProcessor") for these base classes,
                // or we could pass the logger/node in. For simplicity, we use a static logger.
                RCLCPP_WARN(rclcpp::get_logger("PostProcessor"), "PostProcessor %s of type %s already being reconfigured", post_processor_name_.c_str(), post_processor_type_.c_str());
            }
            configured_ = false;
            bool retval = true;

            retval = retval && loadConfiguration(config);
            retval = retval && configure();
            configured_ = retval;
            return retval;
        }

        virtual bool process(const T& data_in, T& data_out) = 0;

        std::string getType() { return post_processor_type_; }
        const std::string& getName() const { return post_processor_name_; }

    protected:
        virtual bool configure() = 0;

        bool getParam(const std::string& name, std::string& value) const
        {
            if (!params_[name]) return false;
            try { value = params_[name].as<std::string>(); return true; } catch (...) { return false; }
        }

        bool getParam(const std::string& name, bool& value) const
        {
            if (!params_[name]) return false;
            try { value = params_[name].as<bool>(); return true; } catch (...) { return false; }
        }

        bool getParam(const std::string& name, double& value) const
        {
            if (!params_[name]) return false;
            try { value = params_[name].as<double>(); return true; } catch (...) { return false; }
        }

        bool getParam(const std::string& name, int& value) const
        {
            if (!params_[name]) return false;
            try { value = params_[name].as<int>(); return true; } catch (...) { return false; }
        }

        bool getParam(const std::string& name, unsigned int& value) const
        {
            if (!params_[name]) return false;
            try { 
                int signed_value = params_[name].as<int>();
                if (signed_value < 0) return false;
                value = signed_value;
                return true;
            } catch (...) { return false; }
        }

        bool getParam(const std::string& name, std::vector<double>& value) const
        {
            if (!params_[name] || !params_[name].IsSequence()) return false;
            value.clear();
            try {
                for (auto it = params_[name].begin(); it != params_[name].end(); ++it) {
                    value.push_back(it->as<double>());
                }
                return true;
            } catch (...) { return false; }
        }

        bool getParam(const std::string& name, std::vector<std::string>& value) const
        {
            if (!params_[name] || !params_[name].IsSequence()) return false;
            value.clear();
            try {
                for (auto it = params_[name].begin(); it != params_[name].end(); ++it) {
                    value.push_back(it->as<std::string>());
                }
                return true;
            } catch (...) { return false; }
        }

        bool getParam(const std::string& name, YAML::Node& value) const
        {
            if (!params_[name]) return false;
            value = params_[name];
            return true;
        }
        
        std::string post_processor_name_;
        std::string post_processor_type_;
        bool configured_;
        YAML::Node params_;

    private:
        bool setNameAndType(const YAML::Node& config)
        {
            if (!config["name"])
            {
                RCLCPP_ERROR(rclcpp::get_logger("PostProcessor"), "PostProcessor didn't have name defined");
                return false;
            }
            std::string name = config["name"].as<std::string>();

            if (!config["type"])
            {
                RCLCPP_ERROR(rclcpp::get_logger("PostProcessor"), "PostProcessor %s didn't have type defined", name.c_str());
                return false;
            }
            std::string type = config["type"].as<std::string>();

            post_processor_name_ = name;
            post_processor_type_ = type;
            RCLCPP_DEBUG(rclcpp::get_logger("PostProcessor"), "Configuring PostProcessor of Type: %s with name %s", type.c_str(), name.c_str());
            return true;
        }

    protected:
        bool loadConfiguration(const YAML::Node& config)
        {
            if (!config.IsMap())
            {
                RCLCPP_ERROR(rclcpp::get_logger("PostProcessor"), "A PostProcessor configuration must be a map with fields name, type, and params");
                return false;
            } 

            if (!setNameAndType(config))
            {
                return false;
            }

            if (config["params"])
            {
                YAML::Node params = config["params"];
                if (!params.IsMap())
                {
                    RCLCPP_ERROR(rclcpp::get_logger("PostProcessor"), "params must be a map");
                    return false;
                }
                params_ = params;
            }
            return true;    
        }
    };

    template <typename T>
    class PostProcessorChain
    {
    private:
        pluginlib::ClassLoader<terrain_analyzer::PostProcessor<T>> loader_;
        std::vector<std::shared_ptr<terrain_analyzer::PostProcessor<T>>> reference_pointers_;
        T buffer0_;
        T buffer1_;
        bool configured_;

    public:
        PostProcessorChain(std::string data_type) : 
            loader_("terrain_analyzer", std::string("terrain_analyzer::PostProcessor<") + data_type + std::string(">")), 
            configured_(false)
        {
            std::string lib_string = "";
            std::vector<std::string> libs = loader_.getDeclaredClasses();
            for (unsigned int i = 0 ; i < libs.size(); i ++)
            {
                lib_string = lib_string + std::string(", ") + libs[i];
            }    
            RCLCPP_DEBUG(rclcpp::get_logger("PostProcessorChain"), "In PostProcessorChain ClassLoader found the following libs: %s", lib_string.c_str());
        }

        ~PostProcessorChain()
        {
            clear();
        }

        bool configure(const std::string& param_file_path, const std::string& param_namespace)
        {
            try {
                YAML::Node config = YAML::LoadFile(param_file_path);
                
                // Navigate to terrain_analyzer_node -> ros__parameters -> param_namespace
                if (config["terrain_analyzer_node"] && 
                    config["terrain_analyzer_node"]["ros__parameters"] &&
                    config["terrain_analyzer_node"]["ros__parameters"][param_namespace]) 
                {
                    YAML::Node chain_config = config["terrain_analyzer_node"]["ros__parameters"][param_namespace];
                    return this->configure(chain_config);
                } else {
                    RCLCPP_DEBUG(rclcpp::get_logger("PostProcessorChain"), "Could not load the chain configuration from namespace %s. Assuming empty.", param_namespace.c_str());
                    configured_ = true;
                    return true;
                }
            } catch (const YAML::Exception& e) {
                RCLCPP_ERROR(rclcpp::get_logger("PostProcessorChain"), "Error parsing YAML file %s: %s", param_file_path.c_str(), e.what());
                return false;
            }
        }

        bool process(const T& data_in, T& data_out)
        {
            unsigned int list_size = reference_pointers_.size();
            bool result = true;
            if (list_size == 0)
            {
                data_out = data_in;
                result = true;
            }
            else if (list_size == 1)
                result = reference_pointers_[0]->process(data_in, data_out);
            else if (list_size == 2)
            {
                result = reference_pointers_[0]->process(data_in, buffer0_);
                if (result == false) return false;
                result = result && reference_pointers_[1]->process(buffer0_, data_out);
            }
            else
            {
                result = reference_pointers_[0]->process(data_in, buffer0_);
                if (result == false) return false;
                for (unsigned int i = 1; i < reference_pointers_.size() - 1; i++)
                {
                    if (i % 2 == 1)
                        result = result && reference_pointers_[i]->process(buffer0_, buffer1_);
                    else
                        result = result && reference_pointers_[i]->process(buffer1_, buffer0_);
                    
                    if (result == false) return false;
                }
                if (list_size % 2 == 1)
                    result = result && reference_pointers_.back()->process(buffer1_, data_out);
                else
                    result = result && reference_pointers_.back()->process(buffer0_, data_out);
            }
            return result;
        }

        bool clear() 
        {
            configured_ = false;
            reference_pointers_.clear();
            return true;
        }

        bool configure(const YAML::Node& config)
        {
            if (!config.IsSequence())
            {
                RCLCPP_ERROR(rclcpp::get_logger("PostProcessorChain"), "The PostProcessor chain specification must be a list.");
                return false;
            }

            for (std::size_t i = 0; i < config.size(); ++i)
            {
                if(!config[i].IsMap())
                {
                    RCLCPP_ERROR(rclcpp::get_logger("PostProcessorChain"), "PostProcessors must be specified as maps.");
                    return false;
                }
                else if (!config[i]["type"])
                {
                    RCLCPP_ERROR(rclcpp::get_logger("PostProcessorChain"), "Could not add a PostProcessor because no type was given");
                    return false;
                }
                else if (!config[i]["name"])
                {
                    RCLCPP_ERROR(rclcpp::get_logger("PostProcessorChain"), "Could not add a PostProcessor because no name was given");
                    return false;
                }
                else
                {
                    for (std::size_t j = i + 1; j < config.size(); ++j)
                    {
                        if(!config[j].IsMap())
                        {
                            RCLCPP_ERROR(rclcpp::get_logger("PostProcessorChain"), "PostProcessors must be specified as maps.");
                            return false;
                        }

                        if(!config[j]["name"])
                        {
                            RCLCPP_ERROR(rclcpp::get_logger("PostProcessorChain"), "PostProcessors names must be strings.");
                            return false;
                        }

                        std::string namei = config[i]["name"].as<std::string>();
                        std::string namej = config[j]["name"].as<std::string>();
                        if (namei == namej)
                        {
                            RCLCPP_ERROR(rclcpp::get_logger("PostProcessorChain"), "A PostProcessor with the name %s already exists", namei.c_str());
                            return false;
                        }
                    }

                    std::string type = config[i]["type"].as<std::string>();
                    std::string name = config[i]["name"].as<std::string>();

                    if (type.find("/") == std::string::npos)
                    {
                        RCLCPP_ERROR(rclcpp::get_logger("PostProcessorChain"), "Bad PostProcessor type %s. Filter type must be of form <package_name>/<filter_name>", type.c_str());
                        return false;
                    }

                    std::vector<std::string> libs = loader_.getDeclaredClasses();
                    bool found = false;
                    for (auto it = libs.begin(); it != libs.end(); ++it)
                    {
                        if (*it == type)
                        {
                            found = true;
                            break;
                        }
                    }
                    if (!found)
                    {
                        RCLCPP_ERROR(rclcpp::get_logger("PostProcessorChain"), "Couldn't find PostProcessor of type %s", type.c_str());
                        return false;
                    }
                }

                std::string type = config[i]["type"].as<std::string>();
                std::string name = config[i]["name"].as<std::string>();

                std::shared_ptr<terrain_analyzer::PostProcessor<T>> p = loader_.createSharedInstance(type);
                if (p.get() == NULL)
                {
                    RCLCPP_ERROR(rclcpp::get_logger("PostProcessorChain"), "Failed to allocate PostProcessor %s of type %s", name.c_str(), type.c_str());
                    return false;
                }

                bool configured = p->configure(config[i]);
                if (!configured)
                {
                    RCLCPP_ERROR(rclcpp::get_logger("PostProcessorChain"), "Failed to configure PostProcessor %s of type %s", name.c_str(), type.c_str());
                    return false;
                }

                reference_pointers_.push_back(p);
                RCLCPP_DEBUG(rclcpp::get_logger("PostProcessorChain"), "Configured %s:%s PostProcessor", type.c_str(), name.c_str());
            }

            configured_ = true;
            return true;
        }
    };
}
