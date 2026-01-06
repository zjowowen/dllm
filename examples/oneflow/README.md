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

## Interactive chat

```bash
python -u examples/oneflow/chat.py \
  --model_dir "models/oneflow/mm_toy/checkpoint-final"
```


