# CHANGELOG


## v0.3.0 (2025-03-17)

### Bug Fixes

- Fix data min and data max values in data loader
  ([`d978948`](https://github.com/ADA-research/CTRAIN/commit/d978948f5545a96341a48a422234601312360284))

- Fix HPO when no default configuration is provided.
  ([`5ef65bf`](https://github.com/ADA-research/CTRAIN/commit/5ef65bfe4378e54af43210eebfa07a7362593a55))

None is not a valid value for Categoricals.

- Fix preprocessing for tiny imagenet
  ([`dab2c3b`](https://github.com/ADA-research/CTRAIN/commit/dab2c3b1280f73338935a205ffcce5f81d43a63c))

### Features

- Add automatic download of TinyImageNet dataset
  ([`c84f6b8`](https://github.com/ADA-research/CTRAIN/commit/c84f6b83828b2607b86d79782a4e1998e3f34123))

added the automatic download to the data loader

- Add Loss Fusion for CROWN IBP
  ([`bfbd3cc`](https://github.com/ADA-research/CTRAIN/commit/bfbd3ccd7a2c54512501b93cf4ba33430faafb7e))

Technique was introduced in the auto_LiRPA paper. This reduces memory requirements substantially and
  can thus scale CROWN IBP to larger models and datasets.

- Add possibility to treat training as deterministic in HPO
  ([`6078f5b`](https://github.com/ADA-research/CTRAIN/commit/6078f5b9690486ccc7eac9569d07d3b73ed8765c))

this leads to only one training run per hyperparameter configuration which may be useful on large
  datasets and models. However, this gives an inaccurate estimate of the configuration performance.


## v0.2.1 (2025-02-19)

### Bug Fixes

- Pass gradient expansion alpha to STAPS loss calculation
  ([`06a4ae2`](https://github.com/ADA-research/CTRAIN/commit/06a4ae295db69fd717426840f4c85b6f8d7f8c22))

### Continuous Integration

- Add git pull in publish workflow
  ([`3bd02a8`](https://github.com/ADA-research/CTRAIN/commit/3bd02a8016c823802df3d7af4ba081d9681a36d4))

the commit in the previous workflow step was not present during the publish phase


## v0.2.0 (2025-02-19)

### Features

- Add checkpoint save interval
  ([`379f30e`](https://github.com/ADA-research/CTRAIN/commit/379f30e18867fbf1f944df09039ee5f54f4fca4b))

Users can now specify an interval of epochs after which a checkpoint is saved. Before that, the
  checkpoints were saved every epoch which may have been undesirable due to space constraints.


## v0.1.3 (2025-02-17)

### Bug Fixes

- Add Tiny ImageNet loader to exports of data_loaders package
  ([`3add754`](https://github.com/ADA-research/CTRAIN/commit/3add754a624db26fded1a89ce4aaad8b1faf561e))

- Fix resume_from_checkpoint functionality
  ([`015d01e`](https://github.com/ADA-research/CTRAIN/commit/015d01ec9b5fa2747aacfc8bc401f8e71c149e98))

Until now, the start_epoch was not passed correctly to the train function. In addition, we bumped
  the SMAC dependency

- Make evaluation method in model wrappers configurable
  ([`62a704d`](https://github.com/ADA-research/CTRAIN/commit/62a704da28558a83ff2415007d69762cea9480fc))

Until now, certified robustness evaluation using the `evaluate` method was carried out using the
  ADAPTIVE method. Now, users may provide a incomplete verification method to use. The default is
  still `ADAPTIVE`, i.e. the certification methods are carried out sequentially in ascending order
  of computational complexity.


## v0.1.2 (2025-02-17)

### Continuous Integration

- Fix versioning and publishing workflows
  ([`e003d74`](https://github.com/ADA-research/CTRAIN/commit/e003d74c7d07de49a0d52d11af8c4a083834d337))


## v0.1.1 (2025-02-07)
