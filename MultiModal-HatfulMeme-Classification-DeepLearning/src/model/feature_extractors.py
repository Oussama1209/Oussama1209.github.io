import torch
import torch.nn as nn
import torchvision
import timm
from torchvision.models import resnet152, ResNet152_Weights, efficientnet_b7
from transformers import (
    AutoModelForSequenceClassification,
    BertTokenizer,
    AutoTokenizer,
    AutoModel,
    BertModel,
    AutoModelForMaskedLM,
    AutoModelForImageClassification,
)


class TextFeature_extractor:

    def __init__(self, text_model, device):
        self.text_model = text_model
        self.device = device

        if self.text_model == "toxigen_hatebert":
            self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
            # Load the model without the classification layer
            self.model = AutoModelForSequenceClassification.from_pretrained(
                "tomh/toxigen_hatebert"
            )
            # self.model = AutoModel.from_pretrained("tomh/toxigen_hatebert")
        elif self.text_model == "toxigen_roberta":
            self.tokenizer = AutoTokenizer.from_pretrained("tomh/toxigen_roberta")
            # Load the model without the classification layer
            self.model = AutoModelForSequenceClassification.from_pretrained(
                "tomh/toxigen_roberta"
            )
        elif self.text_model == "bert":
            self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
            self.model = BertModel.from_pretrained("bert-base-uncased")
        elif self.text_model == "late_bert":
            self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
            self.model = AutoModel.from_pretrained("bert-base-uncased")
        elif self.text_model == "bert_finetuned":
            self.tokenizer = AutoTokenizer.from_pretrained(
                "limjiayi/bert-hateful-memes-expanded"
            )
            self.model = AutoModelForMaskedLM.from_pretrained(
                "limjiayi/bert-hateful-memes-expanded"
            )
        else:
            raise KeyError(f"{self.text_model} does not exist")

        for param in self.model.parameters():
            param.requires_grad = False

        self.model.to(self.device)

    def __call__(self, text):
        if self.text_model == "toxigen_roberta":
            tokenized_input = self.tokenizer(
                text, return_tensors="pt", padding=True
            ).to(self.device)

            # Get the hidden states from the model
            with torch.no_grad():
                outputs = self.model(**tokenized_input, output_hidden_states=True)
                hidden_states = outputs.hidden_states

                # The CLS token is the first token in the sequence in the last layer
                cls_token_representation = hidden_states[-1][:, 0, :]

            return cls_token_representation.squeeze()

        elif self.text_model == "toxigen_hatebert":
            tokenized_input = self.tokenizer(
                text, return_tensors="pt", padding=True
            ).to(self.device)

            # Get the hidden states from the model
            with torch.no_grad():
                outputs = self.model(**tokenized_input, output_hidden_states=True)
                hidden_states = outputs.hidden_states

                # The CLS token is the first token in the sequence in the last layer
                cls_token_representation = hidden_states[-1][:, 0, :]

            return cls_token_representation.squeeze()
            # tokenized_input = self.tokenizer(
            #     text, return_tensors="pt", padding=True
            # ).to(self.device)

            # # Get the hidden states from the model
            # with torch.no_grad():
            #     outputs = self.model(**tokenized_input, output_hidden_states=True)
            #     pooled_output = outputs.pooler_output
            # return pooled_output.squeeze()
        elif self.text_model == "bert":
            # encoding = self.tokenizer.batch_encode_plus(
            #     text,  # List of input texts
            #     padding=True,  # Pad to the maximum sequence length
            #     truncation=True,  # Truncate to the maximum sequence length if necessary
            #     return_tensors="pt",  # Return PyTorch tensors
            #     add_special_tokens=True,  # Add special tokens CLS and SEP
            # )
            # input_ids = encoding["input_ids"].to(self.device)
            # attention_mask = encoding["attention_mask"].to(self.device)
            # with torch.no_grad():
            #     outputs = self.model(input_ids, attention_mask=attention_mask)
            #     word_embeddings = outputs.last_hidden_state
            #     sentence_embedding = word_embeddings.mean(dim=1)
            #     return sentence_embedding
            encoded = self.tokenizer.encode_plus(
                text=text,
                add_special_tokens=True,  # add [CLS] and [SEP] tokens
                max_length=50,  # set the maximum length of a sentence
                truncation=True,  # truncate longer sentences to max_length
                padding="max_length",  # add [PAD] tokens to shorter sentences
                return_attention_mask=True,  # generate the attention mask
                return_tensors="pt",  # return encoding results as PyTorch tensors
            )
            token_ids = encoded["input_ids"].to(self.device)
            attention_mask = encoded["attention_mask"].to(self.device)

            with torch.no_grad():
                # outputs = self.model(**tokenized_input)
                last_hidden_states = self.model(
                    token_ids, attention_mask=attention_mask
                )["last_hidden_state"]
                mean_pooled_embedding = last_hidden_states.mean(axis=1)
                # zeros = torch.zeros_like(mean_pooled_embedding)
                return mean_pooled_embedding.squeeze()
                # return zeros.squeeze()
        elif self.text_model == "bert_finetuned":
            tokenized_input = self.tokenizer(
                text, return_tensors="pt", padding=True
            ).to(self.device)

            # Get the hidden states from the model
            with torch.no_grad():
                outputs = self.model(**tokenized_input, output_hidden_states=True)
                hidden_states = outputs.hidden_states

                # The CLS token is the first token in the sequence in the last layer
                cls_token_representation = hidden_states[-1][:, 0, :]

            return cls_token_representation.squeeze()
        else:
            raise KeyError(f"{self.text_model} does not exist")

class VisualFeature_extractor:
    def __init__(self, vision_model, device):
        self.device = device     
        if vision_model == "resnet152":
            self.image_dim = 224
            self.v_module = resnet152(weights=ResNet152_Weights.DEFAULT)
            self.v_module.fc = nn.Identity()
        elif vision_model == "vit_224":
            self.image_dim = 224
            self.v_module = timm.create_model("vit_base_patch16_224", pretrained=True)
            self.v_module.head = nn.Identity()

        elif vision_model == "vit_512":
            self.image_dim = 512
            self.v_module = timm.create_model(
                "vit_base_patch16_siglip_512", pretrained=True
            )
            self.v_module.head = nn.Identity()
        elif vision_model == "efficientnet_b7":

            class ExtractEmbeddings(nn.Module):
                def __init__(self, model):
                    super(ExtractEmbeddings, self).__init__()
                    # Extract all layers except the final fully connected layer
                    self.features = nn.Sequential(*list(model.children())[:-1])

                def forward(self, x):
                    x = self.features(x)
                    # Flatten the output tensor to (batch_size, num_features)
                    x = torch.flatten(x, 1)

                    return x

            model = efficientnet_b7(weights="IMAGENET1K_V1")
            self.v_module = ExtractEmbeddings(model)
        elif vision_model == "vit_finetuned":

            class ExtractEmbeddings(nn.Module):
                def __init__(self, model):
                    super(ExtractEmbeddings, self).__init__()
                    # Extract all layers except the final fully connected layer
                    self.model = model

                def forward(self, x):
                    outputs = self.model(x, output_hidden_states=True, return_dict=True)
                    hidden_states = outputs.hidden_states
                    embeddings = hidden_states[
                        -1
                    ]  # Assuming the last hidden state is the desired embedding
                    logits = embeddings[:, 0, :]
                    return logits

            self.image_dim = 224
            model = AutoModelForImageClassification.from_pretrained(
                "tommilyjones/vit-base-patch16-224-finetuned-hateful-meme-restructured",
                output_hidden_states=True,
            )
            self.v_module = ExtractEmbeddings(model)
            # self.v_module.head = nn.Identity()
        else:
            raise KeyError(f"{vision_model} does not exist")

        if vision_model == "efficientnet_b7":
            self.img_transform = torchvision.transforms.Compose(
                [
                    torchvision.transforms.Resize(256),
                    torchvision.transforms.CenterCrop(224),
                    torchvision.transforms.ToTensor(),
                    torchvision.transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                ]
            )
        else:
            self.img_transform = torchvision.transforms.Compose(
                [
                    torchvision.transforms.Resize(
                        size=(self.image_dim, self.image_dim)
                    ),
                    torchvision.transforms.ToTensor(),
                    # all torchvision models expect the same
                    # normalization mean and std
                    # https://pytorch.org/docs/stable/torchvision/models.html
                    torchvision.transforms.Normalize(
                        mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
                    ),
                ]
            )

        # for param in self.v_module.parameters():
        #     param.requires_grad = False
        self.v_module.to(self.device)
        self.v_module = self.v_module.eval()

    def __call__(self, image):
        image = self.img_transform(image).unsqueeze(0)
        with torch.no_grad():
            embeddings = self.v_module(image.to(self.device))
            return embeddings.squeeze()
