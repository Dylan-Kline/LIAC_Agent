from mmengine.registry import Registry

DATASET = Registry('data', 
                   locations=['src.data'])
MEMORY = Registry('memory',
                  locations=['src.memory'])
PROVIDER = Registry('provider',
                    locations=['src.provider'])
PROMPT = Registry('prompt',
                  locations=['src.prompt'])
ENVIRONMENT = Registry('environment',
                       locations=['src.environment'])
PLOTS = Registry('plot',
                 locations=['plotting'])