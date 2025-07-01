import json
import torch
import torch.nn.functional as F
from transformers import BertTokenizer
from models.bert_encoder import BertEncoder
from models.rgat import RGAT
from models.scil import SCILModule
from models.decoder import NonAutoDecoder

def load_model(config, device):
    bert = BertEncoder(config["bert_model"]).to(device)
    rgat = RGAT(config["hidden_dim"], config["relation_types"]).to(device)
    scil = SCILModule(config["hidden_dim"]).to(device)
    decoder = NonAutoDecoder(config["hidden_dim"], config["num_classes"]).to(device)

    checkpoint = torch.load("outputs/checkpoints/model.pt", map_location=device)
    bert.load_state_dict(checkpoint["bert"])
    rgat.load_state_dict(checkpoint["rgat"])
    scil.load_state_dict(checkpoint["scil"])
    decoder.load_state_dict(checkpoint["decoder"])

    bert.eval()
    rgat.eval()
    scil.eval()
    decoder.eval()

    return bert, rgat, scil, decoder

def evaluate():
    with open("config.json") as f:
        config = json.load(f)
    with open("data/odia_triplet.json") as f:
        data = json.load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = BertTokenizer.from_pretrained(config["bert_model"])

    bert, rgat, scil, decoder = load_model(config, device)

    correct = [0] * 5
    total = [0] * 5

    for example in data:
        tokens = example["tokens"]
        encoded = tokenizer(" ".join(tokens), return_tensors="pt", truncation=True, padding=True)
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)

        with torch.no_grad():
            h = bert(input_ids, attention_mask)
            h_graph = rgat(h, None, None)
            proj = scil(h[:, 0, :])
            fused = torch.cat([
                proj["polarity"], proj["aspect"],
                proj["opinion"], proj["implicitness"],
                h_graph.mean(1)
            ], dim=-1)
            outputs = decoder(fused)

        pred = [torch.argmax(logit, dim=-1).item() for logit in outputs]
        gold = [
            1,  # dummy label indices (replace with label mapping logic)
            0,
            1,
            3,
            2
        ]

        for i in range(5):
            correct[i] += int(pred[i] == gold[i])
            total[i] += 1

    categories = ["Aspect", "Category", "Opinion", "Polarity", "Implicitness"]
    for i in range(5):
        acc = correct[i] / total[i] * 100
        print(f"{categories[i]} Accuracy: {acc:.2f}%")

if __name__ == "__main__":
    evaluate()
