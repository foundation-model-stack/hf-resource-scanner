from transformers import AutoModelForCausalLM, TrainingArguments
from datasets import load_dataset
from trl import SFTTrainer

# from HFResourceScanner import Scanner
from HFResourceScanner import scanner 


dataset = load_dataset("imdb", split="train")

model = AutoModelForCausalLM.from_pretrained("facebook/opt-350m")

trainer = SFTTrainer(
    model,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=512,
    args=TrainingArguments(output_dir="tmp_trainer", max_steps=5),
    callbacks=[scanner.Scanner(output_fmt="output.json")]
)


scanner.modelhook(vars().items()) # Line added for config detection. To be added after defining the trainer object

trainer.train()
