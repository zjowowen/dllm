import functools
import os
from dataclasses import dataclass, field

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

    # OneFlow model size (toy defaults)
    dim: int = 512
    depth: int = 8
    dim_head: int = 64
    heads: int = 8

    # image latent dim is unused for text-only pretraining, but kept for config completeness
    dim_latent: int = 4


@dataclass
class DataArguments(dllm.utils.DataArguments):
    dataset_args: str = "Trelis/tiny-shakespeare"
    text_field: str = "Text"
    max_length: int = 256
    streaming: bool = False
    drop_tail: bool = True
    insert_eos: bool = field(default=True)


@dataclass
class TrainingArguments(OneFlowTrainer.OneFlowConfig):
    output_dir: str = None  # overwrite this
    num_train_epochs: int = 1
    learning_rate: float = 1e-4
    per_device_train_batch_size: int = 8
    per_device_eval_batch_size: int = 8
    scheduler_cls: str = field(
        default="LinearKappaScheduler",
        metadata={
            "help": (
                "The scheduler class controlling κ(t). "
                "Available options: see `dllm/core/schedulers/kappa.py`"
            )
        },
    )


def build_tokenizer(tokenizer_name_or_path: str) -> transformers.PreTrainedTokenizer:
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        tokenizer_name_or_path, padding_side="right"
    )

    # ensure pad/eos/bos exist (mirror dllm.utils.get_tokenizer behavior)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token
    if tokenizer.eos_token is None:
        tokenizer.eos_token = tokenizer.pad_token
    if tokenizer.bos_token is None:
        tokenizer.bos_token = tokenizer.pad_token

    # add OneFlow special tokens (safe even for text-only)
    tokenizer.add_special_tokens(
        {"additional_special_tokens": [ONEFLOW_IMAGE_TOKEN, ONEFLOW_IMAGE_SOM, ONEFLOW_IMAGE_EOM]}
    )
    return tokenizer


def train():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # necessary when batch does not contain "labels" field
    training_args.label_names = []
    # necessary when batch contains customized fields
    training_args.remove_unused_columns = False
    # necessary for streaming dataset
    training_args.accelerator_config.dispatch_batches = False

    dllm.utils.print_args_main(model_args, data_args, training_args)
    dllm.utils.initial_training_setup(model_args, data_args, training_args)

    tokenizer = build_tokenizer(model_args.tokenizer_name_or_path)

    # ----- Dataset (PT-style) ------------------------------------------------------
    with accelerate.PartialState().local_main_process_first():
        dataset = dllm.data.load_pt_dataset(
            data_args.dataset_args,
            streaming=data_args.streaming,
        )

        map_fn = functools.partial(
            dllm.utils.tokenize_and_group,
            tokenizer=tokenizer,
            text_field=data_args.text_field,
            seq_length=data_args.max_length,
            insert_eos=data_args.insert_eos,
            drop_tail=data_args.drop_tail,
            add_special_tokens=False,
        )

        dataset = dataset.map(
            map_fn,
            batched=True,
            remove_columns=dataset["train"].column_names,
            **({} if data_args.streaming else {"num_proc": data_args.num_proc}),
            **({} if data_args.streaming else {"desc": "Mapping dataset to PT format"}),
        )

        # Ensure each sample starts with BOS (required for insertion slot semantics)
        bos_id = int(tokenizer.bos_token_id)

        def add_bos(row):
            ids = row["input_ids"]
            if not ids:
                return row
            if ids[0] != bos_id:
                row["input_ids"] = [bos_id] + ids
            return row

        dataset = dataset.map(
            add_bos,
            num_proc=0 if data_args.streaming else data_args.num_proc,
            desc="Prepending BOS",
        )

        if data_args.streaming:
            dataset = dataset.shuffle(seed=training_args.seed)

    # ----- Model ------------------------------------------------------------------
    cfg = OneFlowConfig(
        vocab_size=len(tokenizer),
        dim=model_args.dim,
        depth=model_args.depth,
        dim_head=model_args.dim_head,
        heads=model_args.heads,
        dim_latent=model_args.dim_latent,
    )
    model = OneFlowModel(cfg)

    # ----- Training ---------------------------------------------------------------
    accelerate.PartialState().wait_for_everyone()
    logger.info("Start OneFlow text-only training...")
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

    # Save
    final_dir = os.path.join(training_args.output_dir, "checkpoint-final")
    os.makedirs(final_dir, exist_ok=True)
    model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)


if __name__ == "__main__":
    train()


