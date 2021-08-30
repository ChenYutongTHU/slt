import os
import sys

def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)


this_dir = os.path.dirname(__file__)
add_path(os.path.join(this_dir, 'models_3d', 'I3D'))
add_path(os.path.join(this_dir, 'models_3d', 'CoCLR'))
add_path(os.path.join(this_dir, 'models_3d', 'CoCLR','dataset'))
add_path(os.path.join(this_dir, 'models_3d', 'S3D_HowTo100M'))
add_path(os.path.join(this_dir, 'models_3d'))
add_path(os.path.join(this_dir))
# add_path(os.path.join(this_dir, '..', '..'))
# # add_path(os.path.join(this_dir, '..', '..', 'common'))
# # add_path(os.path.join(this_dir, '..', '..', 'common_pytorch'))
# add_path(os.path.join(this_dir, '..', '..', 'slt'))
# add_path(os.path.join(this_dir, '..', '..', 'WLASL/code/I3D'))
# add_path(os.path.join(this_dir, '..', '..', 'CoCLR'))
# add_path(os.path.join(this_dir, '..', '..', 'S3D_HowTo100M'))
# add_path(os.path.join(this_dir))

