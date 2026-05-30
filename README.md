# Overview
**Official implementation** of the practical method, ***CSaN***, derived from the ***SaN*** interpretation framework introduced in "Selection-as-Nonlinearity: Bridging Attention and Activation via a Joint Game–Decision Lens for Interpretable, Discriminative Visual Representations", a **CVPR 2026 *Highlight* paper in the *Deep learning architectures and techniques*** subject area.

[Virtual Presentation](https://cvpr.thecvf.com/virtual/2026/poster/37390)

[Paper](https://openaccess.thecvf.com/content/CVPR2026/html/Cai_Selection-as-Nonlinearity_Bridging_Attention_and_Activation_via_a_Joint_Game-Decision_Lens_CVPR_2026_paper.html)

[Youtube Short Video](https://www.youtube.com/watch?v=E1xBGERCWVQ)

### Notes

- SaN and CSaN are not intended to be limited to vision. We would be grateful if the broader community could help explore its applicability to diverse data modalities and tasks. As a preliminary example, we have also observed clear improvements from CSaN on a bioinformatics task involving structural data.

- CSaN is not a plug-and-play module, but rather a model-specific compensation scheme guided by the SaN perspective. The CSaN implementations for the Swin, ViT, and Hiera families are provided as examples that users can follow to build tailored CSaN variants for the attention models they wish to implement.

------------------------------------------------

### For ImageNet Experiments

## Install Requirements
Please see environment-full.yml

------------------------------------------------

## Training (from scratch)
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 --master_port 12345 train_dist_torch_compile.py "$@" /{path to imagenet}/ --model {model: e.g., swin_tiny_csan_v2} --epochs 300 --cooldown-epochs 10 --checkpoint-hist 1 -b 128 --grad-accum-steps 4 --opt adamw --lr 2e-3 --weight-decay 0.05 --warmup-epochs 20 --drop 0 --drop-path {by model: e.g., 0.2 for swin_tiny_xxx and 0.3 for swin_small_xxx, please see the table below for more recommended settings} --aug-repeats 3 --mixup 0.8 --cutmix 1.0 --color-jitter 0.4 --opt-eps 1e-8 --smoothing 0.1 --remode pixel --reprob 0.25 --aa rand-m9-mstd0.5-inc1 --seed 0 --warmup-lr 1e-6 --min-lr 1e-5 --sched cosine --sched-on-updates -j 8 --amp --amp-dtype bfloat16 --dist-bn reduce --torchcompile --output /{path to save checkpoints}

## Fine-tuning/Using pre-trained weights
Using --resume /{path to checkpoints and pick one}




