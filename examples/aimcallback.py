from transformers import AutoModelForCausalLM, TrainingArguments
from datasets import load_dataset
from trl import SFTTrainer

from aim.hugging_face import AimCallback
from HFResourceScanner import Scanner

aim_callback = AimCallback(repo="/data/aim", experiment="cg-reg-extra")
def write_data(data, metadata):
    aim_callback.experiment["hf_resource_scanner_data"] = data
    aim_callback.experiment["hf_resource_scanner_metadata"] = metadata

dataset = load_dataset("imdb", split="train")
model = AutoModelForCausalLM.from_pretrained("facebook/opt-350m")

trainer = SFTTrainer(
    model,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=512,
    args=TrainingArguments(output_dir="tmp_trainer", max_steps=5),
    callbacks=[aim_callback, Scanner(output_fmt=write_data)]
)

trainer.train()
