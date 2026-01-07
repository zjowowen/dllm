# OneFlow (toy) examples

> These scripts require the OneFlow optional deps (Transfusion backbone + torchdiffeq, etc).
>
> Install:
>
> ```bash
> pip install -e ".[oneflow]"
> ```

## Text-only pretraining (toy)

```bash
accelerate launch --config_file scripts/accelerate_configs/ddp.yaml --num_processes 1 \
  examples/oneflow/pt_text.py \
  --output_dir "models/oneflow/text_toy" \
  --tokenizer_name_or_path "gpt2" \
  --dataset_args "Trelis/tiny-shakespeare" \
  --text_field "Text" \
  --max_length 256 \
  --max_steps 2000
```

## Mixed-modal pretraining (toy random latents)

```bash
accelerate launch --config_file scripts/accelerate_configs/ddp.yaml --num_processes 1 \
  examples/oneflow/pt_mm.py \
  --output_dir "models/oneflow/mm_toy" \
  --tokenizer_name_or_path "gpt2" \
  --max_steps 2000 \
  --image_num_tokens 64
```

## Sampling

```bash
python -u examples/oneflow/sample.py \
  --model_dir "models/oneflow/mm_toy/checkpoint-final" \
  --prompt "Once upon a time"
```

## Offline multimodal pretraining (WebDataset latents)

This is the recommended workflow for **no-network clusters**:
1) Download images+captions to WebDataset shards on a machine with internet/data access.
2) Precompute VAE latents into a new WebDataset (much smaller than raw images).
3) Copy the resulting bundle to the cluster and train from local paths only.

### 1) Precompute latents (online machine)

```bash
python -u scripts/oneflow/precompute_latents_wds.py \
  --input_shards "/path/to/wds_128/shard-*.tar" \
  --output_dir "/path/to/latents_128_bundle" \
  --image_size 128 \
  --vae_id_or_path "stabilityai/sd-vae-ft-mse" \
  --batch_size 64 --num_workers 8 \
  --maxcount 10000 \
  --tokenizer_name_or_path "gpt2" \
  --max_caption_tokens 128 \
  --write_input_ids True
```

### 2) Train from latents shards (offline cluster)

```bash
sbatch --gres=gpu:8 scripts/train.slurm.sh \
  --accelerate_config fsdp \
  --script_path examples/oneflow/pt_wds_latents.py \
  --output_dir "/path/to/checkpoints/oneflow/mm_latents_128" \
  --tokenizer_name_or_path "/path/to/latents_128_bundle/tokenizer" \
  --shards "/path/to/latents_128_bundle/wds_latents" \
  --use_precomputed_ids True \
  --max_steps 200000 \
  --per_device_train_batch_size 8 \
  --gradient_accumulation_steps 2
```

### 3) Sample and decode PNGs (offline)

```bash
python -u examples/oneflow/sample_and_decode.py \
  --model_dir "/path/to/checkpoints/oneflow/mm_latents_128/checkpoint-final" \
  --prompt "a photo of a flower <|oneflow_image|>" \
  --vae_id_or_path "/path/to/offline/sd-vae-ft-mse" \
  --output_dir "/path/to/vis/oneflow_mm" \
  --image_num_tokens 256 --latent_h 16 --latent_w 16
```

## Interactive chat

```bash
python -u examples/oneflow/chat.py \
  --model_dir "models/oneflow/mm_toy/checkpoint-final"
```


