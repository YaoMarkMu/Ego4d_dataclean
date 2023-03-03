

## Preparation
### Install dependencies 
```bash
conda env create -f environment.yml
source activate egovlp
```

### Ego4D videos and metadata
> You can skip the source video download if pretraining is not required.
1. Follow the guideline [here](https://ego4d-data.org/docs/start-here/#cli-download), download the following to  `{PATH_TO_EGO4D}`
   - Ego4D source videos (nearly 7 TB).
   - Ego4D videos metadata `manifest.csv` and benchmark metadata, e.g., `nlq_train.json` for NLQ.
   - Create the dir `dataset` and add a soft link by `ln -s {PATH_TO_EGO4D} dataset/ego4d`.

2. For effectively pretraining, we compress videos in the following way:
   - Resize the source videos with a short size equal to 256 by script  `utils/video_resize.py`.
   - Chunk the resized videos to multiple segments (up to 600 sec) by script `utils/video_chunk.py`.

### EgoClip: an egocentric video-language pretraining dataset
- Download the EgoClip metadata from [here](https://drive.google.com/file/d/1-aaDu_Gi-Y2sQI_2rsI2D1zvQBJnHpXl/view?usp=sharing) and put it to `dataset/egoclip.csv`.

- For the usage of EgoClip, please see our dataloader `data_loader/EgoClip_EgoMCQ_dataset.py`. The data format of EgoClip is:
  ```python
  import pandas as pd
  
  metadata = pd.read_csv('dataset/egoclip_metadata.csv', sep='\t', error_bad_lines=False)
  print(metadata.shape[0])
  print(metadata.iloc[0])
  
  # Out:
  3847723                                                         # Num of clips for EgoClip
  
  clip_idx                                                     0  # the idx of clip
  video_uid                 001e3e4e-2743-47fc-8564-d5efd11f9e90  # the uid of source video
  video_dur                                           128.033333  # the duration of source video
  narration_source                              narration_pass_1  # the source of annotator
  narration_ind                                                0  # the idx of narration
  narration_time                                          3.3445  # the narration timestamp
  clip_start                                            2.967651  # the start timestamp of clip
  clip_end                                              3.721266  # the end timestamp of clip
  clip_text           #C C picks a bag of clothes from the floor  # the narration of clip
  tag_verb                                                  [93]  # the verb idx of the narration
  tag_noun                                        [192, 115, 12]  # the noun idx of the narration
  ```
  
^ The terms `tag_verb` and `tag_noun` are used for EgoNCE pretraining objective, which considers synonyms. For example, `pick`, `collect`, `gather` are all belong to the verb parent with idx 93: `take_(pick,_grab,_get)`.
The mapping dictionary can be found [here](https://drive.google.com/drive/folders/16fUv5rrZmt06Ty3QAEweDpveC-84RI9Z?usp=sharing).

### EgoMCQ: an egocentric video-language development set

- Download the EgoMCQ metadata from [here](https://drive.google.com/file/d/1-5iRYf4BCHmj4MYQYFRMY4bhsWJUN3rW/view?usp=sharing) and put it to `dataset/egomcq.json`.
- EgoMCQ is a benchmark for video-language multiple-choice questions. Given a text query, we want the model to choose the correct video clip from five candidates that sampled from two settings: `inter-video` or `intra-video`.
- For the usage of EgoMCQ, please see our dataloader `data_loader/EgoClip_EgoMCQ_dataset.py`.


## 🏋️‍️ Pretraining
This code is built on PyTorch with DistributedDataParallel (DDP). We pretrain EgoVLP on 4 nodes, each with 8 A100 GPUs (10 epochs in about two days).

- Train on EgoClip:  `python3 -m torch.distributed.launch 
  --nnodes=$HOST_NUM 
  --node_rank=$INDEX 
  --master_addr $CHIEF_IP 
  --nproc_per_node $HOST_GPU_NUM 
  --master_port 8081 
  run/train_egoclip.py --config configs/pt/egoclip.json`
  
- Test on EgoMCQ:  `python3 -m torch.distributed.launch 
  --nnodes=$HOST_NUM 
  --node_rank=$INDEX 
  --master_addr $CHIEF_IP 
  --nproc_per_node $HOST_GPU_NUM 
  --master_port 8081 
  run/train_egoclip.py --config configs/eval/egomcq.json`
  
- Monitor the EgoMCQ curve during pretraining: `tensorboard --logdir results  --bind_all`

## 🗄 Pretrained Weights
- The pretrained EgoVLP model (EgoClip w/ EgoNCE) with best performance on EgoMCQ (90.7% inter-video & 57.2% intra-video) is released in [EgoVLP_PT_BEST](https://drive.google.com/file/d/1-cP3Gcg0NGDcMZalgJ_615BQdbFIbcj7/view?usp=sharing).
- Please download and put the checkpoint under: `pretrained/`

^ This checkpoint is used for EPIC-Kitchens, NLQ, MQ, OSSC, and PNR tasks, except for Charades-Ego. Since we found that VLP (CC3M+WebVid2M, EgoClip) alway degrades significantly on Charades-Ego after the first epoch, we evaluate Charades-Ego using the first pretraining epoch weights of the pretrained model in [Pretrained Weights link](https://drive.google.com/file/d/10lRA4Fldt-c5Azh5D2Zvjwi-_YR5ve5e/view?usp=sharing).

^^ You can use the pretrained checkpoint to power other egocentric video benchmarks. :)

