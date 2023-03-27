# __all__ = ['acquistion', 'dataset', 'model', 'strategy']

import yaml
with open('utils/config.yaml', 'r') as file:
    config = yaml.safe_load(file)
    file.close()