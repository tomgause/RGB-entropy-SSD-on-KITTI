# config.py
import os.path

# gets home dir cross platform
home = os.path.expanduser("~")
ddir = os.path.join(home,"data\\KITTI\\")

KITTIroot = ddir # path to KITTI root div

# NO IDEA WHAT THIS SHOULD BE, COPIED 512 DATA.
# EDITS HERE HAVE SIGNIFICANT CONSEQUENCES.
# TODO: figure this out.
v = {
        'feature_maps' : [64, 32, 16, 8, 4, 2, 1],

        'min_dim' : 512,

        'steps' : [8, 16, 32, 64, 128, 256, 512],

        'min_sizes' : [20, 51, 133, 215, 296, 378, 460],

        'max_sizes' : [51, 133, 215, 296, 378, 460, 542],

        'aspect_ratios' : [[2], [2, 3], [2, 3], [2, 3], [2, 3], [2], [2]],

        'variance' : [0.1, 0.2],

        'clip' : True,

        'name' : 'v0_192x624',
}