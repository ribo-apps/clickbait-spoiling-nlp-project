import wandb
import locale
import evaluate
from evaluate import evaluator
from datasets import load_dataset
from torch.cuda import empty_cache
from transformers import AutoTokenizer
from transformers import DataCollatorForSeq2Seq
from transformers import AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer


locale.getpreferredencoding = lambda: "UTF-8"
wandb.login()


data_path = "/kuacc/users/ebostanci18/hpc_run/nlp/train_changed.json"
checkpoint = "t5-large"

tokenizer = AutoTokenizer.from_pretrained(checkpoint)


data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=checkpoint)


def preprocess_function(examples):

    inputs = examples['targetParagraphs']
    model_inputs = tokenizer(inputs, max_length=1024, truncation=True)

    labels = tokenizer(text_target=examples["spoiler"], max_length=128, truncation=True)

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

data = load_dataset("json", data_files=data_path)

tokenized_data = data.map(preprocess_function, batched=True, remove_columns=['postPlatform', 'targetParagraphs', 'targetTitle', 'targetDescription', 'targetKeywords', 'provenance', 'spoilerPositions', 'tags'])

tokenized_data = tokenized_data["train"].train_test_split(train_size=0.8, seed=42)
tokenized_data["validation"] = tokenized_data.pop("test")


model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint)


mt_metrics = evaluate.combine(
    ["bleu", "chrf", "rouge"], force_prefix=True
)

bleu = evaluate.load("bleu")
def compute_metrics(pred):
    labels_ids = pred.label_ids
    pred_ids = pred.predictions

    predictions = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)

    labels_ids[labels_ids == -100] = tokenizer.pad_token_id
    references = tokenizer.batch_decode(labels_ids, skip_special_tokens=True)

    outputs = mt_metrics.compute(predictions=predictions,
                             references=references)

    return outputs



empty_cache()
output_dir = "t5_large_1e_v2"
training_args = Seq2SeqTrainingArguments(
    output_dir=output_dir,
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    weight_decay=0.01,
    save_total_limit=3,
    num_train_epochs=1,
    predict_with_generate=True,
    fp16=False,
    #push_to_hub=True,
    report_to="wandb",
    #optim="paged_adamw_32bit",
    logging_steps=10,
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_data['train'],
    eval_dataset=tokenized_data["validation"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    #compute_metrics=compute_metrics,
)

trainer.train()

trainer.save_model(output_dir+"/last")
