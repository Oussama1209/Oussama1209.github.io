import torch
from pathlib import Path
from hmm_dataset import *
from helpers.utils import *
from helpers.train_eval import *
from .feature_extractors import *
from .combine_modules import *

class HatefulMemesModel(torch.nn.Module):
    def __init__(self, hparams):
        super().__init__()
        for data_key in ["train_path", "dev_path", "img_dir", "model_name"]:
            if data_key not in hparams.keys():
                raise KeyError(f"{data_key} is required")

        self.hparams = hparams
        self.text_model = self.hparams.get("text_model", "toxigen_roberta")
        self.visual_model = self.hparams.get("vision_model", "resnet152")

        if self.text_model == "toxigen_hatebert":
            self.text_embedding_dim = 768
        elif self.text_model == "toxigen_roberta":
            self.text_embedding_dim = 1024
        elif self.text_model == "bert":
            self.text_embedding_dim = 768
        elif self.text_model == "bert_finetuned":
            self.text_embedding_dim = 768
        else:
            raise KeyError(f"{self.text_model} does not exist")

        self.dropout_p = self.hparams.get("dropout_p", 0.1)

        if self.visual_model == "resnet152":
            self.visual_embedding_dim = 2048
        elif self.visual_model == "vit_224":
            self.visual_embedding_dim = 768
        elif self.visual_model == "vit_512":
            self.visual_embedding_dim = 768
        elif self.visual_model == "efficientnet_b7":
            self.visual_embedding_dim = 2560
        elif self.visual_model == "vit_finetuned":
            self.visual_embedding_dim = 768
        else:
            raise KeyError(f"{self.vision_module} does not exist")

        # we want to project the output of modules into these dims
        self.text_feature_dim = self.hparams.get("text_feature_dim", 300)
        self.vision_feature_dim = self.hparams.get(
            "vision_feature_dim",
            self.text_feature_dim,
        )
        self.output_path = Path(self.hparams.get("output_path", "model_outputs"))
        self.output_path.mkdir(exist_ok=True)

        # instantiate transforms, datasets
        self.img_module = self.image_module()
        self.t_module = self.text_module()
        self.train_dataset = self.build_dataset("train_path", "train")
        self.dev_dataset = self.build_dataset("dev_path", "dev")
        self.test_dataset = self.build_dataset("test_path", "test")
        print("Dataset loaded:")
        print("train ", len(self.train_dataset))
        print("val ", len(self.dev_dataset))
        print("test ", len(self.test_dataset))

        # set up model and training
        self.model_name = self.hparams["model_name"]
        self.model = self.build_model(self.model_name)
        self.optimizer, self.scheduler = configure_optimizers(
            self.model.parameters(), self.hparams.get("lr", 0.001)
        )

    def forward(self, text, image, label=None):
        return self.model(text, image, label)

    def build_dataset(self, dataset_key, dataset_type: str):
        return HatefulMemesDataset(
            data_path=self.hparams.get(dataset_key, dataset_key),
            img_dir=self.hparams.get("img_dir"),
            dataset_type=dataset_type,
            image_transform=self.img_module,
            text_transform=self.t_module,
            # balance=True if "train" in str(dataset_key) else False,
            balance=self.hparams.get("balance", False),
            subset=self.hparams.get("subset", None),
            visual_model=self.hparams.get("vision_model"),
            text_model=self.hparams.get("text_model"),
            save_data=self.hparams.get("save_data", False),
            load_data=self.hparams.get("load_data", False),
            dataset=self.hparams.get("dataset", "hmm"),
        )

    def text_module(self):
        return TextFeature_extractor(
            self.text_model,
            self.hparams.get("device"),
        )

    def image_module(self):
        return VisualFeature_extractor(self.visual_model, self.hparams.get("device"))

    def textFeatures_project(self, identity=False):
        if identity:
            t_project = torch.nn.Identity()
        else:
            t_project = torch.nn.Linear(
                in_features=self.text_embedding_dim, out_features=self.text_feature_dim
            )
        return t_project

    def imageFeatures_project(self, identity=False):
        if identity:
            i_project = torch.nn.Identity()
        else:
            i_project = torch.nn.Linear(
                in_features=self.visual_embedding_dim, out_features=self.vision_feature_dim
            )
        return i_project
        # elif vision_model == "detr":
        #     # Load pretrained DETR model
        #     detr_model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")

        #     # Extract the backbone from the DETR model
        #     backbone = detr_model.model.backbone

        #     # Create a module that processes the image through the backbone and reduces to vision_feature_dim
        #     class DETRBackbone(nn.Module):
        #         def __init__(self, backbone, out_dim):
        #             super(DETRBackbone, self).__init__()
        #             self.backbone = backbone
        #             self.conv = nn.Conv2d(2048, out_dim, kernel_size=1)
        #             self.fc = nn.Linear(out_dim, out_dim)

        #         def forward(self, images):
        #             # Extract the feature maps from the backbone
        #             features = self.backbone(images)

        #             # Use the last feature map from the list of feature maps
        #             last_feature_map = features[-1]

        #             # Pool the features and pass through the linear layer
        #             pooled_features = F.adaptive_avg_pool2d(last_feature_map, (1, 1)).view(last_feature_map.size(0), -1)
        #             out = self.fc(pooled_features)
        #             return out

        #     v_module = DETRBackbone(backbone, self.vision_feature_dim)
        # else:
        #     raise ValueError(f"Unknown vision model: {vision_model}")
        # return v_module

    def build_model(self, model_name):
        t_project = self.textFeatures_project()
        i_project = self.imageFeatures_project()
        if model_name == "concat":
            combined_module = LanguageAndVisionConcat(
                num_classes=self.hparams.get("num_classes", 3),
                loss_fn=torch.nn.CrossEntropyLoss(),
                language_module=self.imageFeatures_project(identity=True),
                vision_module=self.imageFeatures_project(identity=True),
                # text_feature_dim=self.text_feature_dim,
                text_feature_dim=self.text_embedding_dim,
                # vision_feature_dim=self.vision_feature_dim,
                vision_feature_dim=self.visual_embedding_dim,
                fusion_output_size=self.hparams.get("fusion_output_size", 512),
                dropout_p=self.hparams.get("dropout_p", 0.5),
            )
        elif model_name == "attention_fusion":
            combined_module = LanguageAndVisionAttentionOptA(
                num_classes=self.hparams.get("num_classes", 3),
                loss_fn=torch.nn.CrossEntropyLoss(),
                language_module=t_project,
                vision_module=i_project,
                text_feature_dim=self.text_feature_dim,
                vision_feature_dim=self.vision_feature_dim,
                fusion_output_size=self.hparams.get("fusion_output_size", 512),
                dropout_p=self.hparams.get("dropout_p", 0.1),
            )
        elif model_name == "cross_attention":
            combined_module = LanguageAndVisionCrossAttention(
                num_classes=self.hparams.get("num_classes", 3),
                loss_fn=torch.nn.CrossEntropyLoss(),
                language_module=self.textFeatures_project(),
                # language_module=self.textFeatures_project(identity=True),
                vision_module=self.imageFeatures_project(),
                # vision_module=self.imageFeatures_project(identity=True),
                # text_feature_dim=self.text_embedding_dim,
                text_feature_dim=self.text_feature_dim,
                # vision_feature_dim=self.visual_embedding_dim,
                vision_feature_dim=self.vision_feature_dim,
                fusion_output_size=self.hparams.get("fusion_output_size", 512),
                dropout_p=self.hparams.get("dropout_p", 0.1),
            )
        elif model_name == "clip":
            combined_module = Clip(
                text_module=t_project,
                vision_module=i_project,
                text_feature_dim=self.text_feature_dim,
                vision_feature_dim=self.vision_feature_dim,
                num_classes=self.hparams.get("num_classes", 3),
                loss_fn=torch.nn.CrossEntropyLoss(),
            )
        else:
            raise KeyError(f"{model_name} does not exist")
        return combined_module

    def fit(self):
        set_seed(self.hparams.get("random_state", 42))
        train_dataloader = torch.utils.data.DataLoader(
            self.train_dataset,
            shuffle=True,
            batch_size=self.hparams.get("batch_size", 4),
            num_workers=self.hparams.get("num_workers", 16),
        )
        val_dataloader = torch.utils.data.DataLoader(
            self.dev_dataset,
            shuffle=False,
            batch_size=self.hparams.get("batch_size", 4),
            num_workers=self.hparams.get("num_workers", 16),
        )

        train_run, val_run, run_path, best_model_path, last_model_path = train_validate(
            self.hparams.get("device"),
            self.hparams.get("nb_epochs", 10),
            self.model,
            train_dataloader,
            val_dataloader,
            self.optimizer,
            self.scheduler,
            self.hparams.get("early_stop_patience", 4),
            self.hparams.get("early_stop_min_delta", 0.001),
            self.hparams.get("gradient_clip_val", 1),
            self.output_path,
            self.model_name,
        )
        return train_run, val_run, run_path, best_model_path, last_model_path

    def evaluate(self, run_path=None):
        test_dataloader = torch.utils.data.DataLoader(
            self.test_dataset,
            shuffle=False,
            batch_size=self.hparams.get("batch_size", 4),
            num_workers=self.hparams.get("num_workers", 16),
        )
        test_run = evaluate(self.hparams.get("device"), self.model, test_dataloader, run_path)
        return test_run
