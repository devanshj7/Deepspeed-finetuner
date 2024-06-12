import deepspeed
from transformers import SchedulerType
import torch
from tqdm.auto import tqdm


class CustomTrainer:
    def __init__(self, model, args, train_dataloader, eval_dataloader):
        self.model = model
        self.args = args
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader

    def train(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.args.learning_rate)
        criterion = torch.nn.CrossEntropyLoss()

        for epoch in range(self.args.num_train_epochs):
            self.model.train()
            total_loss = 0
            for batch in tqdm(self.train_dataloader, desc=f"Epoch {epoch + 1}", disable=self.args.local_rank not in [-1, 0]):
                input_ids = batch["input_ids"].to(self.args.device)
                labels = batch["labels"].to(self.args.device)

                optimizer.zero_grad()
                outputs = self.model(input_ids, labels=labels)
                loss = criterion(outputs.logits.view(-1, outputs.logits.size(-1)), labels.view(-1))
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            avg_train_loss = total_loss / len(self.train_dataloader)
            print(f"Avg Train Loss: {avg_train_loss}")

            # Evaluation
            self.model.eval()
            eval_loss = 0
            for batch in tqdm(self.eval_dataloader, desc=f"Evaluating", disable=self.args.local_rank not in [-1, 0]):
                with torch.no_grad():
                    input_ids = batch["input_ids"].to(self.args.device)
                    labels = batch["labels"].to(self.args.device)
                    outputs = self.model(input_ids, labels=labels)
                    loss = criterion(outputs.logits.view(-1, outputs.logits.size(-1)), labels.view(-1))
                    eval_loss += loss.item()

            avg_eval_loss = eval_loss / len(self.eval_dataloader)
            print(f"Avg Eval Loss: {avg_eval_loss}")

        print("Training finished.")


def parse_args():
    parser = argparse.ArgumentParser(description="Finetune a transformers model on a causal language modeling task")
    parser.add_argument("--dataset_name", type=str, default="mlabonne/guanaco-llama2-1k", help="The name of the dataset to use (via the datasets library).")
    parser.add_argument("--dataset_config_name", type=str, default=None, help="The configuration name of the dataset to use (via the datasets library).")
    parser.add_argument("--train_file", type=str, default=None, help="A csv or a json file containing the training data.")
    parser.add_argument("--validation_file", type=str, default=None, help="A csv or a json file containing the validation data.")
    parser.add_argument("--validation_split_percentage", default=5, help="The percentage of the train set used as validation set in case there's no validation split")
    parser.add_argument("--model_name_or_path", type=str, help="Path to pretrained model or model identifier from huggingface.co/models.", default="mistralai/Mistral-7B-Instruct-v0.2")
    parser.add_argument("--config_name", type=str, default=None, help="Pretrained config name or path if not the same as model_name")
    parser.add_argument("--tokenizer_name", type=str, default=None, help="Pretrained tokenizer name or path if not the same as model_name")
    parser.add_argument("--use_slow_tokenizer", action="store_true", help="If passed, will use a slow tokenizer (not backed by the 🤗 Tokenizers library).")
    parser.add_argument("--per_device_train_batch_size", type=int, default=1, help="Batch size (per device) for the training dataloader.")
    parser.add_argument("--per_device_eval_batch_size", type=int, default=1, help="Batch size (per device) for the evaluation dataloader.")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Initial learning rate (after the potential warmup period) to use.")
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay to use.")
    parser.add_argument("--num_train_epochs", type=int, default=3, help="Total number of training epochs to perform.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--lr_scheduler_type", type=SchedulerType, default="linear", help="The scheduler type to use.")
    parser.add_argument("--max_train_steps", type=int, default=None, help="Total number of training steps to perform. If provided, overrides num_train_epochs.")
    parser.add_argument("--logging_dir", type=str, default="logs", help="[TensorBoard](https://www.tensorflow.org/tensorboard) log directory.")
    parser.add_argument("--output_dir", type=str, default="outputs", help="Where to store the final model.")
    parser.add_argument("--overwrite_output_dir", action="store_true", help="Overwrite the content of the output directory")
    parser.add_argument("--overwrite_cache", action="store_true", help="Overwrite the cached training and evaluation sets")
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument("--deepspeed_config", type=str, help="DeepSpeed config file", default=None)
    return parser.parse_args()
