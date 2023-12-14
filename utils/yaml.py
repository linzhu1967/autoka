from omegaconf import OmegaConf



def load_config_yaml(config_path):
    # # transfer yaml data into dict
    # with open(config_path, "r") as f:
    #     all_configs = yaml.load(f, Loader=yaml.FullLoader)
    # return all_configs

    conf = OmegaConf.load(config_path)
    # Output is identical to the YAML file
    # print(OmegaConf.to_yaml(conf))
    return OmegaConf.to_container(conf)