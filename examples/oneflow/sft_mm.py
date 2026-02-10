import os
from dataclasses import dataclass, field
from functools import partial

import accelerate
import transformers

import dllm
from dllm.pipelines.oneflow import OneFlowTrainer
from dllm.pipelines.oneflow.models import OneFlowConfig, OneFlowModel
from dllm.pipelines.oneflow.utils import (
    ONEFLOW_IMAGE_EOM,
    ONEFLOW_IMAGE_SOM,
    ONEFLOW_IMAGE_TOKEN,
    OneFlowCollator,
)

logger = dllm.utils.get_default_logger(__name__)


@dataclass
class ModelArguments:
    tokenizer_name_or_path: str = "gpt2"
    # Optional: initialize from an existing OneFlow checkpoint (text-only PT).
    init_model_dir: str | None = None
    dim: int = 512
    depth: int = 8
    dim_head: int = 64
    heads: int = 8
    dim_latent: int = 4


@dataclass
class DataArguments(dllm.utils.DataArguments):
    dataset_args: str = "tatsu-lab/alpaca"
    load_preprocessed_data: bool = False
    mask_prompt_loss: bool = field(default=True)


@dataclass
class TrainingArguments(OneFlowTrainer.OneFlowConfig):
    output_dir: str = None  # overwrite this
    num_train_epochs: float = 1
    learning_rate: float = 1e-4
    per_device_train_batch_size: int = 2
    per_device_eval_batch_size: int = 2
    scheduler_cls: str = field(default="LinearKappaScheduler")


def build_tokenizer(tokenizer_name_or_path: str) -> transformers.PreTrainedTokenizer:
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        tokenizer_name_or_path, padding_side="right"
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token
    if tokenizer.eos_token is None:
        tokenizer.eos_token = tokenizer.pad_token
    if tokenizer.bos_token is None:
        tokenizer.bos_token = tokenizer.pad_token

    tokenizer.add_special_tokens(
        {"additional_special_tokens": [ONEFLOW_IMAGE_TOKEN, ONEFLOW_IMAGE_SOM, ONEFLOW_IMAGE_EOM]}
    )
    return tokenizer


def sft_map_fn(row, *, tokenizer, mask_prompt_loss: bool = True) -> dict:
    """
    Map a chat-style sample (messages) into OneFlow SFT training fields.

    We prefer `tokenizer.apply_chat_template` when available (chat tokenizers),
    but also support plain tokenizers (e.g., GPT-2) by formatting as:
      prompt_text + "\n\n" + response_text
    """
    messages = row["messages"]

    def _ensure_bos(ids: list[int]) -> list[int]:
        bos = tokenizer.bos_token_id
        if bos is None:
            return ids
        if not ids:
            return [int(bos)]
        if int(ids[0]) != int(bos):
            return [int(bos)] + [int(x) for x in ids]
        return [int(x) for x in ids]

    # --- Path A: chat template tokenization (preferred) ---
    use_chat_template = bool(getattr(tokenizer, "chat_template", None))
    if use_chat_template and hasattr(tokenizer, "apply_chat_template"):
        prompt_response_tokens = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=False,
        )
        prompt_response_tokens = _ensure_bos(prompt_response_tokens)
        if mask_prompt_loss:
            prompt_tokens = tokenizer.apply_chat_template(
                messages[:-1],
                tokenize=True,
                add_generation_prompt=True,
            )
            prompt_tokens = _ensure_bos(prompt_tokens)
            return {"input_ids": prompt_response_tokens, "prompt_len": len(prompt_tokens)}
        return {"input_ids": prompt_response_tokens}

    # --- Path B: plain-text formatting (works with GPT-like tokenizers) ---
    # Expect last message is assistant response.
    if not messages or len(messages) < 2:
        raise ValueError("Expected at least 2 messages (user + assistant).")
    prompt_text = str(messages[0].get("content", "") or "").strip()
    resp_text = str(messages[-1].get("content", "") or "").strip()

    # Simple delimiter. Keep it consistent for prompt_len computation.
    prompt_with_delim = prompt_text + "\n\n"
    full_text = prompt_with_delim + resp_text

    full_ids = tokenizer.encode(full_text, add_special_tokens=False)
    full_ids = _ensure_bos(full_ids)

    if mask_prompt_loss:
        prompt_ids = tokenizer.encode(prompt_with_delim, add_special_tokens=False)
        prompt_ids = _ensure_bos(prompt_ids)
        return {"input_ids": full_ids, "prompt_len": len(prompt_ids)}
    return {"input_ids": full_ids}


def train():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    training_args.label_names = []
    training_args.remove_unused_columns = False
    dllm.utils.print_args_main(model_args, data_args, training_args)
    dllm.utils.initial_training_setup(model_args, data_args, training_args)

    tokenizer = build_tokenizer(model_args.tokenizer_name_or_path)

    with accelerate.PartialState().local_main_process_first():
        dataset = dllm.data.load_sft_dataset(
            data_args.dataset_args,
            load_preprocessed_data=data_args.load_preprocessed_data,
        )
        if not data_args.load_preprocessed_data:
            map_fn = partial(
                sft_map_fn,
                tokenizer=tokenizer,
                mask_prompt_loss=data_args.mask_prompt_loss,
            )
            dataset = dataset.map(
                map_fn,
                num_proc=data_args.num_proc,
                desc="Mapping dataset to SFT format",
            )
        dataset = dllm.utils.post_process_dataset(dataset, data_args)

    cfg = OneFlowConfig(
        vocab_size=len(tokenizer),
        bos_token_id=int(tokenizer.bos_token_id),
        eos_token_id=int(tokenizer.eos_token_id),
        pad_token_id=int(tokenizer.pad_token_id),
        unk_token_id=int(tokenizer.unk_token_id) if tokenizer.unk_token_id is not None else None,
        dim=model_args.dim,
        depth=model_args.depth,
        dim_head=model_args.dim_head,
        heads=model_args.heads,
        dim_latent=model_args.dim_latent,
    )
    if model_args.init_model_dir:
        logger.info(f"Loading OneFlowModel from: {model_args.init_model_dir}")
        model = OneFlowModel.from_pretrained(model_args.init_model_dir)
        # Keep vocab in sync with tokenizer (in case special tokens were added).
        model.resize_token_embeddings(len(tokenizer))
    else:
        model = OneFlowModel(cfg)

    accelerate.PartialState().wait_for_everyone()
    logger.info("Start OneFlow SFT (text-only baseline; extend with images as needed)...")
    trainer = OneFlowTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset["train"],
        eval_dataset=dataset.get("test", None),
        args=training_args,
        data_collator=OneFlowCollator(tokenizer=tokenizer),
        scheduler=dllm.core.schedulers.make_kappa_scheduler(training_args.scheduler_cls),
    )
    trainer.train()

    final_dir = os.path.join(training_args.output_dir, "checkpoint-final")
    os.makedirs(final_dir, exist_ok=True)
    trainer.save_model(final_dir)
    tokenizer.save_pretrained(final_dir)


if __name__ == "__main__":
    train()


