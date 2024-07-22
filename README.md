# embryo_binary_segmentation

[![License MIT](https://img.shields.io/pypi/l/embryo_binary_segmentation.svg?color=green)](https://github.com/GuignardLab/embryo_binary_segmentation/raw/main/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/embryo_binary_segmentation.svg?color=green)](https://pypi.org/project/embryo_binary_segmentation)
[![Python Version](https://img.shields.io/pypi/pyversions/embryo_binary_segmentation.svg?color=green)](https://python.org)
[![tests](https://github.com/GuignardLab/embryo_binary_segmentation/workflows/tests/badge.svg)](https://github.com/GuignardLab/embryo_binary_segmentation/actions)
[![codecov](https://codecov.io/gh/GuignardLab/embryo_binary_segmentation/branch/main/graph/badge.svg)](https://codecov.io/gh/GuignardLab/embryo_binary_segmentation)

CNN-based model for 3D segmentation of mouse embryo

----------------------------------

## Installation

You can install `embryo_binary_segmentation` via [pip]:

    pip install embryo_binary_segmentation



To install latest development version :

    pip install git+https://github.com/GuignardLab/embryo_binary_segmentation.git


## Data structure

> compliance with the data structure is necessary for correct data loading

**It is important that the folder with images has FUSE in its name, and the folder with masks has SEG in its name.**

Each folder must have the following structure:

- Train
    - JLM_12
        - FUSE
            - e1.tif
            - e2.tif
            - ...
        - SEG
            - e1.tif
            - e2.tif
            - ...
    - Woon_7
        - FUSE
            - e1.tif
            - e2.tif
            - ...
        - SEG
            - e1.tif
            - e2.tif
            - ...

The alphabetical order of image files and masks must match.


## Parameters

DATA_PARAMS = {
    "data_path": folder with the Train and Val datasets,
    "binarize": if you need to binarize masks before training,
    "target_size": size of cropped images,
    "patch_size": size of patches (less size, easy trainig),
    "augmentations": if you want to apply augmentations
}  

FINE_TUNING = {
    "upload_model_path": path to saved model,
    "old_steps": if you want to continue epochs count,
}


TRAINING_PARAMS = {
    "loss": loss function,
    "learning_rate": 1e-4,
    "batch_size": 8,
    "epochs": total amount,
    "save_model_path": folder where you will save best model,
    "fine_tuning":True,
    "save_each": if you want to save each 5th model weights,
}

TEST_PARAMS = {
    "data_path":"folder with test data",
    "binarize":False,
    "target_size":[64, 512, 512],
    "patch_size":[32, 512, 512],
    "batch_size": 2,
    "load_model_path": path to the model weights,
    "load_csv_path": path to the csv file with losses by epochs
}

PRED_PARAMS = {
    "data_path": folder with data for final prediction,
    "final_load_model_path": path to the model weights,
    "batch_size": don't use big number because you predict on the full size images,
    "save_pred_path": where you want to save predictions
}


## Contributing

Contributions are very welcome. Tests can be run with [tox], please ensure
the coverage at least stays the same before you submit a pull request.

## License

Distributed under the terms of the [MIT] license,
"embryo_binary_segmentation" is free and open source software

## Issues

If you encounter any problems, please [file an issue] along with a detailed description.

----------------------------------

This library was generated using [Cookiecutter] and a custom made template based on [@napari]'s [cookiecutter-napari-plugin] template.


[napari]: https://github.com/napari/napari
[Cookiecutter]: https://github.com/audreyr/cookiecutter
[@napari]: https://github.com/napari
[MIT]: http://opensource.org/licenses/MIT
[BSD-3]: http://opensource.org/licenses/BSD-3-Clause
[GNU GPL v3.0]: http://www.gnu.org/licenses/gpl-3.0.txt
[GNU LGPL v3.0]: http://www.gnu.org/licenses/lgpl-3.0.txt
[Apache Software License 2.0]: http://www.apache.org/licenses/LICENSE-2.0
[Mozilla Public License 2.0]: https://www.mozilla.org/media/MPL/2.0/index.txt
[cookiecutter-napari-plugin]: https://github.com/napari/cookiecutter-napari-plugin
[pip]: https://pypi.org/project/pip/
[PyPI]: https://pypi.org/
[tox]: https://tox.readthedocs.io/en/latest/

[file an issue]: https://github.com/GuignardLab/embryo_binary_segmentation/issues

