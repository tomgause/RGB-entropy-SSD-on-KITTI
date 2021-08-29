import torch
from math import sqrt as sqrt
import numpy as np
from itertools import product as product

class PriorBox(object):
    """Compute priorbox coordinates in center-offset form for each source
    feature map.
    Note:
    This 'layer' has changed between versions of the original SSD
    paper, so we include both versions, but note v is the most tested and most
    recent version of the paper.

    """
    def __init__(self, cfg):
        super(PriorBox, self).__init__()
        # self.type = cfg.name
        self.image_size = [384, 1280]#cfg['min_dim']
        # number of priors for feature map location (either 4 or 6)
        self.num_priors = len(cfg['aspect_ratios'])
        self.variance = cfg['variance'] or [0.1]
        self.feature_maps = cfg['feature_maps']
        self.sizes = cfg['anchor_sizes']
        self.steps = cfg['steps']
        self.aspect_ratios = cfg['aspect_ratios']
        self.clip = cfg['clip']
        self.version = cfg['name']
        for v in self.variance:
            if v <= 0:
                raise ValueError('Variances must be greater than 0')

    def forward(self):
        layers_anchors = []
        # Calculate default anchor boxes for all layers
        for k, f in enumerate(self.feature_maps):
            # Compute position grid
            y, x = np.mgrid[0:f[0], 0:f[1]]
            y = (y.astype(float) + 0.5) / f[0]
            x = (x.astype(float) + 0.5) / f[1]

            # Expand dims to support easy broadcasting
            y = np.expand_dims(y, axis=-1).tolist()
            x = np.expand_dims(x, axis=-1).tolist()

            # Compute relative height and width
            num_anchors = len(self.aspect_ratios[k]) + len(self.sizes)
            h = np.zeros((num_anchors, ), dtype=float).tolist()
            w = np.zeros((num_anchors, ), dtype=float).tolist()

            # Add anchor box with ratio=1
            h = self.sizes[k][0] / f[0]
            w = self.sizes[k][0] / f[1]

            di = 1
            if len(self.sizes) > 1:
                h = sqrt(self.sizes[k][0] * self.sizes[k][1]) / f[0]
                w = sqrt(self.sizes[k][0] * self.sizes[k][1]) / f[1]

            for i, r in enumerate(self.aspect_ratios[k]):
                h = self.sizes[k][0] / f[0] / sqrt(r)
                w = self.sizes[k][0] / f[1] / sqrt(r)

            anchor_bboxes = [y, x, h, w]
            layers_anchors.append(anchor_bboxes)

        # back to torch land
        print(type(layers_anchors))
        print(type(layers_anchors[0]))
        print(type(layers_anchors[0][0]))
        print(type(layers_anchors[0][0][0]))
        print(type(layers_anchors[0][0][0][0]))
        print(type(layers_anchors[0][0][0][0][0]))
        output = torch.Tensor(layers_anchors).view(-1, 4)
        return output


            # for i in range(f[1]):
            #     for j in range(f[0]):
            #         f_k_x = self.image_size[1] / self.steps[k]
            #         f_k_y = self.image_size[0] / self.steps[k]
            #
            #         # unit center x,y
            #         cx = (i + 0.5) / f_k_x
            #         cy = (j + 0.5) / f_k_y
            #
            #         # aspect_ratio: 1
            #         # rel size: min_size
            #         s_k_x = self.min_sizes[k]/self.image_size[1]
            #         s_k_y = self.min_sizes[k]/self.image_size[0]
            #         mean += [cx, cy, s_k_x, s_k_y]
            #
            #         # aspect_ratio: 1
            #         # rel size: sqrt(s_k * s_(k+1))
            #         s_k_prime_x = sqrt(s_k_x * (self.max_sizes[k]/self.image_size[1]))
            #         s_k_prime_y = sqrt(s_k_y * (self.max_sizes[k] / self.image_size[0]))
            #         mean += [cx, cy, s_k_prime_x, s_k_prime_y]
            #
            #         # rest of aspect ratios
            #         for ar in self.aspect_ratios[k]:
            #          mean += [cx, cy, s_k_x*sqrt(ar), s_k_y/sqrt(ar)]
            #          mean += [cx, cy, s_k_x/sqrt(ar), s_k_y*sqrt(ar)]

        # TODO merge these
        # for k, f in enumerate(self.feature_maps):
        #     for i, j in product(range(f[0]), repeat=2):
        #         f_k_x = self.image_size[1] / self.steps[k]
        #         f_k_y = self.image_size[0] / self.steps[k]
        #
        #         # unit center x,y
        #         cx = (j + 0.5) / f_k_x
        #         cy = (i + 0.5) / f_k_y
        #
        #         # aspect_ratio: 1
        #         # rel size: min_size
        #         s_k_x = self.min_sizes[k]/self.image_size[1]
        #         s_k_y = self.min_sizes[k]/self.image_size[0]
        #         mean += [cx, cy, s_k_x, s_k_y]
        #
        #         # aspect_ratio: 1
        #         # rel size: sqrt(s_k * s_(k+1))
        #         s_k_prime_x = sqrt(s_k_x * (self.max_sizes[k]/self.image_size[1]))
        #         s_k_prime_y = sqrt(s_k_y * (self.max_sizes[k] / self.image_size[0]))
        #         mean += [cx, cy, s_k_prime_x, s_k_prime_y]
        #
        #         # rest of aspect ratios
        #         for ar in self.aspect_ratios[k]:
        #             mean += [cx, cy, s_k_x*sqrt(ar), s_k_y/sqrt(ar)]
        #             mean += [cx, cy, s_k_x/sqrt(ar), s_k_y*sqrt(ar)]

        # # back to torch land
        # output = torch.Tensor(mean).view(-1, 4)
        # #if self.clip:
        # #    output.clamp_(max=1, min=0)
        # return output
