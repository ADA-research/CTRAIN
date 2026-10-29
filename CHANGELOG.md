# CHANGELOG


## v0.7.0 (2026-10-29)

### Bug Fixes

- Align fused CROWN-IBP optimizer and module state
  ([`8061947`](https://github.com/ADA-research/CTRAIN/commit/8061947f9c919666e3f16ecd2ffc8dc1411b3cec))

- Align set-based BatchNorm training with pinned CORA semantics
  ([`3a2beac`](https://github.com/ADA-research/CTRAIN/commit/3a2beac36bace6a2a9259af65c53addcd2abd324))

- Apply GLS regularization to native model parameters
  ([`3ab3fa0`](https://github.com/ADA-research/CTRAIN/commit/3ab3fa0199ccafce8d3851475ccede90e6f8a21c))

- Bind verification caches to model and input identity
  ([`2646b41`](https://github.com/ADA-research/CTRAIN/commit/2646b41d4f3947ecf4abf6bb5bd51f643b1d6ac5))

- Differentiate set-based nominal BatchNorm statistics
  ([`2575ac2`](https://github.com/ADA-research/CTRAIN/commit/2575ac28cc76a7dc99884783897afc445bc413d5))

- Honor model class counts and odd input dimensions
  ([`9d232a5`](https://github.com/ADA-research/CTRAIN/commit/9d232a5c289f3c6388293a5f4e916e145f10d7ea))

- Preserve PGD counterexamples across attack iterations
  ([`bd64c89`](https://github.com/ADA-research/CTRAIN/commit/bd64c89946f06d26ea94f7ebe2ceb45ee0f5cde3))

- Preserve PGD dtype and handle Shi regulariser graph variants
  ([`42eca2c`](https://github.com/ADA-research/CTRAIN/commit/42eca2c36c897e3fe8ff612eef28cec973041a7d))

- Preserve scheduler state in training checkpoints
  ([`d120c6e`](https://github.com/ADA-research/CTRAIN/commit/d120c6e4d267882001a3f5954cbf3ee73c85c881))

- Repair BoTorch integration and HPO options
  ([`1d81e7a`](https://github.com/ADA-research/CTRAIN/commit/1d81e7a786e68a716fd2e0dc42cd704b24d0c772))

- Repair epsilon schedule endpoints and resumption
  ([`6783b45`](https://github.com/ADA-research/CTRAIN/commit/6783b45b261de185c6089ed082b0682b696705f7))

- Repair hybrid training bounds and propagate shared options
  ([`1eb3dfa`](https://github.com/ADA-research/CTRAIN/commit/1eb3dfafec7d38b2e38b9cafed4674bbb0b3f315))

- Share one bound graph across TAPS and STAPS blocks
  ([`4a49f17`](https://github.com/ADA-research/CTRAIN/commit/4a49f173e0d8b9b5c0d8a60a4ce5ddf326a69d5d))

- Support short HPO budgets and scheduler-aware checkpoint loading
  ([`af7e028`](https://github.com/ADA-research/CTRAIN/commit/af7e02846e48f9a39efd24d5c08a5f7264381317))

- Support SMAC initial design API across versions
  ([`2a86cbd`](https://github.com/ADA-research/CTRAIN/commit/2a86cbd741e9242b6c2b25dcfd1fdb562eb9b5b0))

- Use evaluation transforms for validation holdouts
  ([`7619629`](https://github.com/ADA-research/CTRAIN/commit/76196296db5bc5305d495ebed160e4bef54fb959))

- Validate ONNX exports against a float64 fallback reference
  ([`d2f232c`](https://github.com/ADA-research/CTRAIN/commit/d2f232c037df7866e779f2b3eca8c48ff67292c6))

- **crown-ibp**: Correct fused warmup cross entropy
  ([`891bd80`](https://github.com/ADA-research/CTRAIN/commit/891bd80552dffed8a3d1b7e2558452234b3d91f0))

- **deps**: Update auto_LiRPA compatibility pin
  ([`723cfe4`](https://github.com/ADA-research/CTRAIN/commit/723cfe4425fae6025e7687ad4531ea1e057b693b))

- **eval**: Preserve state and validate robustness metrics
  ([`ca368f8`](https://github.com/ADA-research/CTRAIN/commit/ca368f8f6aa63b4479f755abcebe641e84b232a1))

- **training**: Respect loader normalization in epsilon schedules
  ([`11634cd`](https://github.com/ADA-research/CTRAIN/commit/11634cddc5205c27c2c8ffc2194db465b634a5d3))

### Build System

- Vendor NeuralSAT with provenance and runtime dependencies
  ([`2e5ccd2`](https://github.com/ADA-research/CTRAIN/commit/2e5ccd2cb94e70d64ad19ae0ed26f2a8e672b73d))

### Chores

- **release**: Bump version to 1.0.0
  ([`4848bbb`](https://github.com/ADA-research/CTRAIN/commit/4848bbb1111f96c17e8f7e7f1565c5e42480201f))

### Documentation

- Align training and evaluation API examples
  ([`1160b86`](https://github.com/ADA-research/CTRAIN/commit/1160b8616b7fd64aa5e6f8cec1ab9ab29ea6c217))

- Refresh guides, API reference and tutorials
  ([`b2d14ba`](https://github.com/ADA-research/CTRAIN/commit/b2d14ba0daef5dc3eb6033429e2483d7e8dddf3d))

### Features

- Add certified training for general output properties
  ([`33739eb`](https://github.com/ADA-research/CTRAIN/commit/33739ebe5b8cece3c12a76515d0791951efbf6ca))

- Add CNN7 final-downsampling architecture option
  ([`bf6da65`](https://github.com/ADA-research/CTRAIN/commit/bf6da65fb5d63f9d6f417034f39ec8fe55ab171f))

- Add expressive IBP training with CC-IBP and Exp-IBP
  ([`8037ded`](https://github.com/ADA-research/CTRAIN/commit/8037dedb4bd2a66cce2898f6a470336aed34e146))

- Add Gaussian loss smoothing training
  ([`9c44a28`](https://github.com/ADA-research/CTRAIN/commit/9c44a28765e4fa2911c22cf37f6373c3935f9dc9))

- Add IBP-R certified training
  ([`670e699`](https://github.com/ADA-research/CTRAIN/commit/670e699f2622153b5505f597ac298881db7d9d49))

- Add native PGD teachers and CC-Dist training
  ([`3dfb5fb`](https://github.com/ADA-research/CTRAIN/commit/3dfb5fb847289696b011e0e7b4ec239162226cac))

- Add portable case studies and documentation notebooks
  ([`465c91e`](https://github.com/ADA-research/CTRAIN/commit/465c91edc9bbce4261f7640efe1315bdd3478cef))

- Add set-based training and LiRPA forward certification
  ([`6ce84be`](https://github.com/ADA-research/CTRAIN/commit/6ce84be0849f1f20f1de6cb8d4922cccae450099))

- Expand MNIST property experiments and HPO
  ([`5851229`](https://github.com/ADA-research/CTRAIN/commit/5851229bce05c0c40f86c1d58fdee0227f4f7e52))

- Expose AutoAttack and alpha-CROWN evaluation options
  ([`88f7065`](https://github.com/ADA-research/CTRAIN/commit/88f706578a815f6e07a5708a0c655526d96b2087))

- Extend property training with lazy datasets and warm-up
  ([`b625726`](https://github.com/ADA-research/CTRAIN/commit/b6257264ebe9239d5fa116d4e9a12c00d161d01c))

- Integrate CC-Dist into multi-objective HPO
  ([`89c0fa1`](https://github.com/ADA-research/CTRAIN/commit/89c0fa123684f2688ca2d196c6d8960d6c731ee7))

- Integrate NeuralSAT and shared complete verification screening
  ([`e2c3474`](https://github.com/ADA-research/CTRAIN/commit/e2c347429ff27a3feb31dd13fb4719471361082e))

- **train**: Select PGD and CC-Dist validation weights
  ([`60886a4`](https://github.com/ADA-research/CTRAIN/commit/60886a483929dd2ff4ac37217b603a92158b7694))

### Refactoring

- Remove case studies from library repository
  ([`6b75643`](https://github.com/ADA-research/CTRAIN/commit/6b756438fd5fed02c57b2869bac56aa99631c68d))

- Unify certified training lifecycle
  ([`0714f2b`](https://github.com/ADA-research/CTRAIN/commit/0714f2bd2315458226143633e9e404a6551b3371))

- **bound**: Rename zonotope module to set_based_zono
  ([`822789c`](https://github.com/ADA-research/CTRAIN/commit/822789cca4f35ff2f3715a2c9ee56eff0bd1dd72))

- **data**: Consolidate image loaders
  ([`d9b08d9`](https://github.com/ADA-research/CTRAIN/commit/d9b08d9ee1bbf98fe59c6b8344ed4bc23ea016e0))

- **eval**: Consolidate certification evaluation
  ([`2f17d7c`](https://github.com/ADA-research/CTRAIN/commit/2f17d7c002becf07c85fe8d076816945fe8524a7))

- **eval**: Simplify property report fields
  ([`d51d908`](https://github.com/ADA-research/CTRAIN/commit/d51d9081ea03cbef09f6561317fbebd19b9dc153))

- **train**: Remove obsolete split-network handling
  ([`0e83e36`](https://github.com/ADA-research/CTRAIN/commit/0e83e360f0a028af02682a93b62214916244de42))

- **wrappers**: Organize implementations by training type
  ([`8aa1ca8`](https://github.com/ADA-research/CTRAIN/commit/8aa1ca860b38ae71b32c7960180c9c6f3311d9f3))

### Testing

- Add self-contained library suite and CPU CI
  ([`08412dc`](https://github.com/ADA-research/CTRAIN/commit/08412dc1cfba6f961fca69cbfd6ca803efabcbc8))

- Extend regressions and configure coverage
  ([`6ae45d4`](https://github.com/ADA-research/CTRAIN/commit/6ae45d49007b96d78caad28c7d3d79b458514c77))


## v0.6.1 (2026-09-01)

### Bug Fixes

- Restore kappa in constructors of CrownIBP and ShiIBP wrappers. closes #8
  ([`e4fe7da`](https://github.com/ADA-research/CTRAIN/commit/e4fe7da695834f6d99515b7bc50e395d36c4d603))

### Chores

- **paper**: Publish Pareto-front model hub and evaluation example
  ([`7fb281a`](https://github.com/ADA-research/CTRAIN/commit/7fb281abf7deabb76bf79667fe689756d4df8234))

### Documentation

- Reference Pareto-front models and ICML paper
  ([`d5a8371`](https://github.com/ADA-research/CTRAIN/commit/d5a8371f27efcfff685701ce47f012afc66504f1))


## v0.6.0 (2026-08-13)


## v0.5.0 (2026-05-26)

### Bug Fixes

- Disable dynamo engine for ONNX export
  ([`fa6e9ec`](https://github.com/ADA-research/CTRAIN/commit/fa6e9ecd559be5be6f118a76e7a197d252a8e90f))

- Improve gradient calculation in PGD attack
  ([`c149106`](https://github.com/ADA-research/CTRAIN/commit/c149106c04dd3f3112847fd12b5c9f9c6e7ab7cd))

- Remove without_grad wrapper around pgd calls
  ([`341cf20`](https://github.com/ADA-research/CTRAIN/commit/341cf20dea1514ba48d84910bf989211b0e37d3c))

- **packaging**: Support installation without Git-hosted dependencies
  ([`fab2a6b`](https://github.com/ADA-research/CTRAIN/commit/fab2a6b94d2f3c89ed2dac4930de16dfb5455948))

### Chores

- Improve device handling in training code
  ([`ea2770a`](https://github.com/ADA-research/CTRAIN/commit/ea2770a00fb3d3d84b8a5972533d51ba535d5dde))

- Improve logging
  ([`1c983d1`](https://github.com/ADA-research/CTRAIN/commit/1c983d18d64f4ee12652f23a34be0916ba598dd1))

- Improve tensor handling in epsilon scheduler
  ([`6304677`](https://github.com/ADA-research/CTRAIN/commit/6304677e03e693e16be4a55010c64324ec6bfe56))

- **paper**: Finalize ICML 2026 paper code and artifacts
  ([`1bdd319`](https://github.com/ADA-research/CTRAIN/commit/1bdd3190f68c2577cffa438ddb55f9b5a816a287))

### Features

- Add multi-objective HPO procedure
  ([`df31b69`](https://github.com/ADA-research/CTRAIN/commit/df31b699afa82b61a43f7d7073c37c15b1723dd0))

- Add paper code for "Rethinking Evaluation Paradigms in IBP-based Certified Training" to appear at
  ICML 2026
  ([`bb3f4ae`](https://github.com/ADA-research/CTRAIN/commit/bb3f4aece2c32ebcb52f0047b8de0b27a671ca34))

- Enhance evaluation capabilities
  ([`852606b`](https://github.com/ADA-research/CTRAIN/commit/852606b5c36aefcb92414f282d6d76ba742566c2))

- **sabr**: Support separate bounds for the PGD center search
  ([`3b7a523`](https://github.com/ADA-research/CTRAIN/commit/3b7a52340cde9cdb5245ab1289620737783e829d))


## v0.4.3 (2026-05-11)

### Bug Fixes

- Add dependency on requests for dataset loading
  ([`aec3b5e`](https://github.com/ADA-research/CTRAIN/commit/aec3b5e7892e3b9690f9cfe60449a610369b510d))

- Correct random initialisation in PGD adversarial attack
  ([`0b5447e`](https://github.com/ADA-research/CTRAIN/commit/0b5447e73cf031ef43772115abd10798acdc503f))

- Ensure that gradient is removed after PGD
  ([`9d6f9e9`](https://github.com/ADA-research/CTRAIN/commit/9d6f9e955e10cb9312b4a1a71cb22be3d1171d04))

- Ensure that SABR uses small boxes for regularisation in all cases
  ([`5a17177`](https://github.com/ADA-research/CTRAIN/commit/5a17177d4180523d7caf8ffa11d3a071738bc769))

- Make train function signature compatible with torch.nn
  ([`3df510c`](https://github.com/ADA-research/CTRAIN/commit/3df510c636454f269205e9bc2272cafe9463852b))

- Remove unintended shuffling for test loaders
  ([`6b7b398`](https://github.com/ADA-research/CTRAIN/commit/6b7b398faed78e352d04966c66bd836536e24e5b))


## v0.4.2 (2025-05-05)

### Bug Fixes

- Set loss fusion model to train mode
  ([`bf57f4e`](https://github.com/ADA-research/CTRAIN/commit/bf57f4e8306b456561ae8292d82f9d62f803bbb1))

### Chores

- Remove kappa trade off
  ([`0554e3e`](https://github.com/ADA-research/CTRAIN/commit/0554e3ebebda5a843757a6a67cb11e8dd59462dc))


## v0.4.1 (2025-05-03)

### Bug Fixes

- Change input node for shi regulariser
  ([`13a3576`](https://github.com/ADA-research/CTRAIN/commit/13a35768248c17d60ab259692db0b4d7cdfbd6c0))

- Make SMAC optimisation seed adjustable
  ([`ac3d25f`](https://github.com/ADA-research/CTRAIN/commit/ac3d25f98f930afcc461b5bd1aa32ef8a5928471))


## v0.4.0 (2025-04-24)


## v0.3.1 (2025-04-01)


## v0.3.0 (2025-03-17)

### Bug Fixes

- Adjust cifar10 std values in accordance with MTL-IBP/SABR implementations
  ([`70d6cb7`](https://github.com/ADA-research/CTRAIN/commit/70d6cb7bb68fda7d6404006631ce53993d165ed6))

- Fix argument passing to SABR
  ([`ee7cd8a`](https://github.com/ADA-research/CTRAIN/commit/ee7cd8a126b4f8141922b2c20719bc54685cdd09))

- Fix Crown IBP training without loss fusion
  ([`88c9f68`](https://github.com/ADA-research/CTRAIN/commit/88c9f68ab7df9e107d88cf9a631cb4db27ab8bad))

- Fix data min and data max values in data loader
  ([`d978948`](https://github.com/ADA-research/CTRAIN/commit/d978948f5545a96341a48a422234601312360284))

- Fix HPO when no default configuration is provided.
  ([`5ef65bf`](https://github.com/ADA-research/CTRAIN/commit/5ef65bfe4378e54af43210eebfa07a7362593a55))

- Fix ONNX export for certification with abCROWN
  ([`ad4d887`](https://github.com/ADA-research/CTRAIN/commit/ad4d8871150ffd4ad461bf7696f624e7040f1831))

- Fix pgd implementation to work with torch.no_grad()
  ([`17681d4`](https://github.com/ADA-research/CTRAIN/commit/17681d4757d66394bfbfc0a0c41e2e211d179521))

- Fix preprocessing for tiny imagenet
  ([`dab2c3b`](https://github.com/ADA-research/CTRAIN/commit/dab2c3b1280f73338935a205ffcce5f81d43a63c))

- Use current eps in MTL IBP pgd epsilon adjustment
  ([`cea2582`](https://github.com/ADA-research/CTRAIN/commit/cea2582ccae23e43a35c7fb1b37ca9a8613e8af1))

### Chores

- Adjust eps logging
  ([`3d11b9a`](https://github.com/ADA-research/CTRAIN/commit/3d11b9ab24b71f61b98895309ffd12dc04a8d90c))

### Features

- Add automatic download of TinyImageNet dataset
  ([`c84f6b8`](https://github.com/ADA-research/CTRAIN/commit/c84f6b83828b2607b86d79782a4e1998e3f34123))

- Add eps factor for PGD attack to SABR.
  ([`3565d15`](https://github.com/ADA-research/CTRAIN/commit/3565d158f6cf187c7b85dd09ce4f29619e453b7a))

- Add Loss Fusion for CROWN IBP
  ([`bfbd3cc`](https://github.com/ADA-research/CTRAIN/commit/bfbd3ccd7a2c54512501b93cf4ba33430faafb7e))

- Add possibility to treat training as deterministic in HPO
  ([`6078f5b`](https://github.com/ADA-research/CTRAIN/commit/6078f5b9690486ccc7eac9569d07d3b73ed8765c))

- Add user-specifiable weights for the components of the HPO loss
  ([`6602f60`](https://github.com/ADA-research/CTRAIN/commit/6602f607b375e47c5aab2f2f43354a988b8b6cb0))

- Make abCROWN config adjustable in complete verification
  ([`ca366fc`](https://github.com/ADA-research/CTRAIN/commit/ca366fc4c7a569a5f254408555d6db54c8ba5846))


## v0.2.1 (2025-02-19)

### Bug Fixes

- Pass gradient expansion alpha to STAPS loss calculation
  ([`06a4ae2`](https://github.com/ADA-research/CTRAIN/commit/06a4ae295db69fd717426840f4c85b6f8d7f8c22))

### Continuous Integration

- Add git pull in publish workflow
  ([`3bd02a8`](https://github.com/ADA-research/CTRAIN/commit/3bd02a8016c823802df3d7af4ba081d9681a36d4))


## v0.2.0 (2025-02-19)

### Features

- Add checkpoint save interval
  ([`379f30e`](https://github.com/ADA-research/CTRAIN/commit/379f30e18867fbf1f944df09039ee5f54f4fca4b))


## v0.1.3 (2025-02-17)

### Bug Fixes

- Add Tiny ImageNet loader to exports of data_loaders package
  ([`3add754`](https://github.com/ADA-research/CTRAIN/commit/3add754a624db26fded1a89ce4aaad8b1faf561e))

- Fix resume_from_checkpoint functionality
  ([`015d01e`](https://github.com/ADA-research/CTRAIN/commit/015d01ec9b5fa2747aacfc8bc401f8e71c149e98))

- Make evaluation method in model wrappers configurable
  ([`62a704d`](https://github.com/ADA-research/CTRAIN/commit/62a704da28558a83ff2415007d69762cea9480fc))


## v0.1.2 (2025-02-17)

### Continuous Integration

- Fix versioning and publishing workflows
  ([`e003d74`](https://github.com/ADA-research/CTRAIN/commit/e003d74c7d07de49a0d52d11af8c4a083834d337))


## v0.1.1 (2025-02-07)
