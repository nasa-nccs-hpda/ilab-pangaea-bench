# import yaml
from omegaconf import OmegaConf

defaults = {
    "dataset": {"auto_download": True, "ignore_index": -1},
    "encoder": {}
}


def build_config(config_type: str, params: dict, cfg: dict = {}):
    """
    Build a YAML configuration from user-defined parameters and stock dataset.

    This function creates or updates a configuration dictionary based on the
    provided parameters and an optional existing configuration.

    Args:
        config_type (str): Type of configuration, corresponding to the
            Pangaea config directory name.
        params (dict): User-defined parameters from the notebook.
        cfg (dict, optional): Existing configuration to build upon.
            Defaults to an empty dictionary.

    Returns:
        cfg_updated: The built or updated config dict.
        yaml_filename = Filename cfg was saved to.
    """
    # Add default values to user-defined params
    cfg_updated = OmegaConf.merge(cfg, params)

    # Automatically generate filename from _target_ cfg field
    target = (
        params.get("_target_") or
        cfg.get("_target_") or
        ""
    )
    base_filename = target.split(".")[-1].lower()
    yaml_filename = f"{base_filename}.yaml"

    # Save to file
    with open(yaml_filename, "w", encoding="utf-8") as file:
        OmegaConf.save(cfg_updated, file)
    print(f"Config saved to {yaml_filename}")

    return cfg_updated, yaml_filename
