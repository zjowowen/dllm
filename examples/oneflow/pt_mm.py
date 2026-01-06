import os
from dataclasses import dataclass, field

import accelerate
import torch
import transformers
from torch.utils.data import Dataset

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

    # latent dim
    dim_latent: int = 4


@dataclass
class DataArguments:
    # toy mixed-modal dataset
    num_samples: int = 10000
    text_len: int = 64
    image_num_tokens: int = 64


@dataclass
class TrainingArguments(OneFlowTrainer.OneFlowConfig):
    output_dir: str = None  # overwrite this
    max_steps: int = 1000
    learning_rate: float = 1e-4
    per_device_train_batch_size: int = 8
    per_device_eval_batch_size: int = 8
    eval_strategy: str = "no"
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


class ToyMMDataset(Dataset):
    def __init__(
        self,
        *,
        tokenizer: transformers.PreTrainedTokenizer,
        num_samples: int,
        text_len: int,
        image_num_tokens: int,
        dim_latent: int,
    ):
        self.tokenizer = tokenizer
        self.num_samples = int(num_samples)
        self.text_len = int(text_len)
        self.image_num_tokens = int(image_num_tokens)
        self.dim_latent = int(dim_latent)

        self.bos = int(tokenizer.bos_token_id)
        self.eos = int(tokenizer.eos_token_id)
        self.image_token = int(tokenizer.convert_tokens_to_ids(ONEFLOW_IMAGE_TOKEN))

        if tokenizer.unk_token_id is not None and self.image_token == int(tokenizer.unk_token_id):
            raise ValueError(f"Tokenizer does not recognize {ONEFLOW_IMAGE_TOKEN}")

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # random text tokens (avoid special tokens crudely by sampling in [0, vocab_size))
        vocab_size = len(self.tokenizer)
        text = torch.randint(0, vocab_size, (self.text_len,), dtype=torch.long).tolist()

        input_ids = [self.bos] + text + [self.image_token] + [self.eos]

        # random latent (acts like ground-truth Y1)
        image_latent = torch.randn((self.image_num_tokens, self.dim_latent), dtype=torch.float32)

        return {"input_ids": input_ids, "image_latent": image_latent}


def train():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    training_args.label_names = []
    training_args.remove_unused_columns = False
    training_args.accelerator_config.dispatch_batches = False

    dllm.utils.print_args_main(model_args, data_args, training_args)
    dllm.utils.initial_training_setup(model_args, data_args, training_args)

    tokenizer = build_tokenizer(model_args.tokenizer_name_or_path)

    # dataset
    dataset = ToyMMDataset(
        tokenizer=tokenizer,
        num_samples=data_args.num_samples,
        text_len=data_args.text_len,
        image_num_tokens=data_args.image_num_tokens,
        dim_latent=model_args.dim_latent,
    )

    # model
    cfg = OneFlowConfig(
        vocab_size=len(tokenizer),
        dim=model_args.dim,
        depth=model_args.depth,
        dim_head=model_args.dim_head,
        heads=model_args.heads,
        dim_latent=model_args.dim_latent,
    )
    model = OneFlowModel(cfg)

    accelerate.PartialState().wait_for_everyone()
    logger.info("Start OneFlow toy mixed-modal training...")

    trainer = OneFlowTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        eval_dataset=None,
        args=training_args,
        data_collator=OneFlowCollator(tokenizer=tokenizer),
        scheduler=dllm.core.schedulers.make_kappa_scheduler(training_args.scheduler_cls),
    )
    trainer.train()

    final_dir = os.path.join(training_args.output_dir, "checkpoint-final")
    os.makedirs(final_dir, exist_ok=True)
    model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)


if __name__ == "__main__":
    train()


