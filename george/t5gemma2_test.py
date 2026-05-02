from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch


def run():
    model_name = "google/t5gemma-2-4b-4b"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)

    text = "What color is a leaf?"

    inputs = tokenizer(text, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    encoder_outputs = model.get_encoder()(**inputs)

    print(encoder_outputs.last_hidden_state.shape)  # shape of the latent representations

    outputs = model.generate(**inputs, max_new_tokens=20)

    result = tokenizer.decode(outputs[0], skip_special_tokens=True)

    print(result)


if __name__ == "__main__":
    run()
