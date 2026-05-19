# Overview
### Official implementation of the practical method, *CSaN*, derived from the *SaN* framework introduced in "Selection-as-Nonlinearity: Bridging Attention and Activation via a Joint Game–Decision Lens for Interpretable, Discriminative Visual Representations", a CVPR 2026 Highlight paper in the *Deep learning architectures and techniques* subject area.

### Notes

- SaN and CSaN are not intended to be limited to vision. We would be grateful if the broader community could help explore its applicability to diverse data modalities and tasks. As a preliminary example, we have also observed clear improvements from CSaN on a bioinformatics task involving structural data.

- CSaN is not a plug-and-play module, but rather a model-specific compensation scheme guided by the SaN perspective. The CSaN implementations for the Swin, ViT, and Hiera families are provided as examples that users can follow to build tailored CSaN variants for the attention models they wish to implement.



