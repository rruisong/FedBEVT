from fedbevt.data_utils.datasets.camera_only.late_fusion_dataset import CamLateFusionDataset

__all__ = {
    'CamLateFusionDataset': CamLateFusionDataset,
}

# the final range for evaluation
GT_RANGE = [-140, -40, -3, 140, 40, 1]
CAMERA_GT_RANGE = [-50, -50, -3, 50, 50, 1]
# The communication range for cavs
COM_RANGE = 70


def build_dataset(dataset_cfg, visualize=False, train=True, test=False, client=None):
    dataset_name = dataset_cfg['fusion']['core_method']
    error_message = f"{dataset_name} is not found. " \
                    f"Please add your processor file's name/" \
                    f"data_utils/datasets/init.py"
    assert dataset_name in ['LateFusionDataset',
                            'EarlyFusionDataset',                           
                            'CamLateFusionDataset',], error_message

    dataset = __all__[dataset_name](
        params=dataset_cfg,
        visualize=visualize,
        train=train,
        test=test,
        client=client
    )

    return dataset
