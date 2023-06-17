# Challenge:
https://pan.webis.de/semeval23/pan23-web/clickbait-challenge.html

[Final Report](https://github.com/ribo-apps/clickbait-spoiling-nlp-project/Final_report.pdf)

[Presentation](https://github.com/ribo-apps/clickbait-spoiling-nlp-project/spoiler_gen_presentation.pdf)
## preprocess.ipynb:</br>
We preprocessed the train and validation datasets in this script.
</br>
## openai-2shot.ipynb:</br>
Requires an OpenAI API key. We prepared our gpt baseline here. We get predictions for validation data using 2-shot.
</br>
## tf-idf.ipynb:</br>
We prepeared our TF-IDF baseline here. Both the prediction and evaluation done in this notebook.
</br>
## llama-lora.ipynb:</br>
We finetuned a LLaMA-7B model using LoRA. We also save the predictions for validation dataset in a txt file
</br>
## falcon.ipynb:</br>
We finetuned a Falcon-7B model using QLoRA. We quantized into 4 bits. We also save the predictions for validation dataset.
</br>
## roberta.ipynb:</br>
We finetuned a RoBERTa and saved the validation outputs.
</br>
## t5.py:</br>
We finetuned a T5 model and save a checkpoint.
</br>
## t5-eval.py:</br>
We make predictions for the trained T5 model, which loads from the saved checkpoint, and save the results.
</br>
## eval-scores.ipynb:</br>
We calculate the Bleu and Bert scores in this script for all save validation outputs from all models.
