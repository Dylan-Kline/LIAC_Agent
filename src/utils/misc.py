def update_data_root(cfg, root):
    cfg.root = root
    for key, value in cfg.items():
        if isinstance(value, dict) and "root" in value:
            cfg[key]["root"] = root