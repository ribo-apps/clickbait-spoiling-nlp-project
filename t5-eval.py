import json
import locale
from tqdm import tqdm
from datasets import load_dataset
from torch.cuda import empty_cache
from transformers import AutoTokenizer
from transformers import AutoModelForSeq2SeqLM

file_path = "/kuacc/users/ebostanci18/hpc_run/nlp/t5_1e_eval.json"
checkpoint = "/kuacc/users/ebostanci18/hpc_run/nlp/t5_large_1e_v2/last"
data_path = "/kuacc/users/ebostanci18/hpc_run/nlp/validation_changed.json"

locale.getpreferredencoding = lambda: "UTF-8"

validation_data = load_dataset("json", data_files=data_path, split="train")

tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint)

model.eval()


results = {}
id = 0
for i, text in tqdm(enumerate(validation_data['targetParagraphs'][id:])):
    empty_cache()
    idx = id + i
    spoiler = validation_data["spoiler"][idx]

    try:
        input_ids = tokenizer(text, max_length=1024, return_tensors="pt").input_ids
        outputs = model.generate(input_ids)
        outputs = tokenizer.decode(outputs[0], skip_special_tokens=True)

        results[idx] = {}
        results[idx]["t5"] = outputs
        results[idx]["real_spoiler"] = spoiler
    except:
        print(f"error for idx: {idx}")

print(len(results))
print(results.keys())


with open(file_path, "w") as json_file:
    json.dump(results, json_file, indent=4)


