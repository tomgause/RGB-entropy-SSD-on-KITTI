# config.py
import os.path

# gets home dir cross platform
home = os.path.expanduser("~")
ddir = os.path.join(home,"data/KITTI/")

KITTIroot = ddir # path to KITTI root div

# gets home dir cross platform
home = os.path.expanduser("~")
ddir = os.path.join(home,"data\\kitti_single\\")

kitti_single_root = ddir # path to KITTI root div

ddir = os.path.join(home,"data\\kitti_single_2\\")

kitti_single_root_2 = ddir # path to KITTI root div

# NO IDEA WHAT THIS SHOULD BE, COPIED 512 DATA.
# EDITS HERE HAVE SIGNIFICANT CONSEQUENCES.
# TODO: figure this out.
v = {
        'feature_maps' : [(48, 160), (48, 160), (24, 80), (24, 80), (12, 40), (6, 20)],

        'min_dim' : [1280, 384],

        'steps' : [8, 16, 32, 64, 100, 300],

        'min_sizes' : [26.88, 57.6, 115.2, 172.8, 230.4, 288],

        'max_sizes' : [57.6, 115.2, 172.8, 230.4, 288, 345.6],

        # 'aspect_ratios' : [[2, .5],
        #                     [2, .5, 3, 1./3],
        #                     [2, .5, 3, 1./3],
        #                     [2, .5, 3, 1./3],
        #                     [2, .5],
        #                     [2, .5]],

        'aspect_ratios' : [[2, 3, .5, 1./3, 2./3],
                        [2, 3, .5, 1./3, 2./3],
                        [2, 3, .5, 1./3, 2./3],
                        [2, 3, .5, 1./3, 2./3],
                        [2, 3, .5, 1./3, 2./3],
                        [2, 3, .5, 1./3, 2./3]],

        'variance' : [0.1, 0.2],

        'clip' : True,

        'name' : 'v0_1280x384',
}
# aspect_ratios: 1.0
# aspect_ratios: 2.0
# aspect_ratios: 0.5
# aspect_ratios: 3.0
# aspect_ratios: 0.3333

#[[1], [2], [2, 3], [2, 3], [2, 3], [2], [2]],

# [[2, .5],
# [2, .5, 3, 1./3],
# [2, .5, 3, 1./3],
# [2, .5, 3, 1./3],
# [2, .5],
# [2, .5]]