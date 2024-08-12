from transformers import AutoModelForCausalLM, TrainingArguments
from datasets import load_dataset
from trl import SFTTrainer
from HFResourceScanner import scanner 

scan = scanner.Scanner(output_fmt="output.json") 

dataset = load_dataset("imdb", split="train")

model = AutoModelForCausalLM.from_pretrained("facebook/opt-350m")

trainer = SFTTrainer(
    model,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=512,
    args=TrainingArguments(output_dir="tmp_trainer", max_steps=5, fp16=True, gradient_checkpointing=True),
    callbacks=[scan]
)

scan.attach_hooks(vars().items()) #config detection


trainer.train()
