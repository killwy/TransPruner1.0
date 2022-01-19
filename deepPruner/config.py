
class resolve(object):
    def __init__(self,d):
        for key,val in d.items():
            if isinstance(val,(list,tuple)):
                setattr(self,key,[resolve(x) if isinstance(x,dict) else x for x in val])
            else:
                setattr(self,key,resolve(val) if isinstance(val, dict) else val)


config = {
    "max_disp": 192,
    "cost_aggregator_scale": 4,  # for DeepPruner-fast change this to 8.
    "mode": "training",  # for evaluation/ submission, change this to evaluation.

    # The code allows the user to change the feature extrcator to any feature extractor of their choice.
    # The only requirements of the feature extractor are:
    #     1.  For cost_aggregator_scale == 4:
    #             features at downsample-level X4 (feature_extractor_ca_level)
    #             and downsample-level X2 (feature_extractor_refinement_level) should be the output.
    #         For cost_aggregator_scale == 8:
    #             features at downsample-level X8 (feature_extractor_ca_level),
    #             downsample-level X4 (feature_extractor_refinement_level),
    #             downsample-level X2 (feature_extractor_refinement_level_1) should be the output,

    #     2.  If the feature extractor is modified, change the "feature_extractor_outplanes_*" key in the config
    #         accordingly.

    "feature_extractor_ca_level_outplanes": 32,
    "feature_extractor_refinement_level_outplanes": 32,  # for DeepPruner-fast change this to 64.
    "feature_extractor_refinement_level_1_outplanes": 32,
    "disp_window":200,
    "min_disp":0,
    "patch_match_args": {
        "sample_count": 12,
        "iteration_count": 2,
        "propagation_filter_size": 3
    },
    "temperature": 7,
    "post_CRP_sample_count": 7,
    "post_CRP_sampler_type": "uniform",  # change to patch_match for Sceneflow model.
    "hourglass_inplanes": 16,
    "training_epoches": 1040,
    "img_h_train":370,
    "img_w_tes":1226,
    "patchsize":4,
    "transformer_layer_num":5,
    "trans_vec_dim":64,
    "defualt_h":256,
    "defualt_w":512
}

config=resolve(config)
