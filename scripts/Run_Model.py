from src.Envirnonment import make_anti_correlated_2good
from src.Model import run_single


parameters={
                    'time_steps': 300,
                    'n_agents': 100,
                    'env_size': (50, 50),
                    'resource_fns': make_anti_correlated_2good(sigma=0.25),
                    'max_resource': 4,
                    'regrowth_rate': 0.05,
                    'metabolic_rate_mean': 1.0,
                    'preferences_mean': [0.2, 0.8],
                    'vision': 6,
                    'bundle_mode': "uniform",
                    'reproduction_enabled': False,
                    'record': True,
                    'live': False
               }


run_single(parameters=parameters)