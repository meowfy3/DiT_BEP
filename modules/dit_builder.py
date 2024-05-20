# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# GLIDE: https://github.com/openai/glide-text2im
# MAE: https://github.com/facebookresearch/mae/blob/main/models_mae.py
# --------------------------------------------------------

from .dit_clipped import DiT_Clipped


#################################################################################
#                                   DiT Configs                                  #
#################################################################################

def DiT_clipper_builder(**kwargs):  # change configuration (depth=num DiT blocks, h_s=length embeddings,n_h=attention )
    return DiT_Clipped(depth=4, hidden_size=512, patch_size=2, num_heads=4, **kwargs)
# you can try firstly to play with these cnfg (kwargs=key_words_arguments)
# CHANGE 'ARCHITECTURE'

DiT_models = {
    'DiT_Clipped': DiT_clipper_builder
}
