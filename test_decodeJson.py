from AnalyticalModel import *
from os.path import join

load_param_path = join("data", "MTMR_31519", "real", "gc-MTMR-31519.json")
model = MTM_MLSE4POL()
model.decode_json_file(load_param_path)