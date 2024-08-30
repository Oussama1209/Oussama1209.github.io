import torch
import pandas as pd
import tqdm as tqdm
import os
from pathlib import Path
from PIL import Image


class HatefulMemesDataset(torch.utils.data.Dataset):
    """Uses jsonl data to preprocess and serve
    dictionary of multimodal tensors for model input.
    """

    def __init__(
        self,
        data_path,
        img_dir,
        image_transform,
        text_transform,
        visual_model: str,
        text_model: str,
        dataset_type: str,
        balance=False,
        random_state=0,
        subset=None,
        save_data: bool = False,
        load_data: bool = False,
        dataset: str = "hmm",
    ):
        self.dataset = dataset
        self.img_dir = Path(img_dir)
        self.image_transform = image_transform
        self.text_transform = text_transform
        directory = "dataloaders"
        datasets_directory_visual = os.path.join(directory, visual_model)
        datasets_directory_text = os.path.join(directory, text_model)

        if not os.path.exists(directory):
            os.makedirs(directory)

        self.samples_frame = pd.read_json(data_path, lines=True)
        # take subset samples
        if subset is not None:
            self.samples_frame = self.samples_frame.sample(subset, random_state=random_state)

        if balance:
            neg = self.samples_frame[self.samples_frame.label.eq(0)]
            pos = self.samples_frame[self.samples_frame.label.eq(1)]
            self.samples_frame = pd.concat(
                [neg.sample(pos.shape[0], random_state=random_state), pos]
            )

        self.samples_frame = self.samples_frame.reset_index(drop=True)
        # self.samples_frame.img = self.samples_frame.apply(
        #     lambda row: (img_dir / row.img), axis=1
        # )

        if dataset == "harmeme":
            # label 0: labels contrains "not harmful", 1: otherwise
            self.samples_frame["labels"] = self.samples_frame["labels"].apply(
                lambda x: 0 if "not harmful" in x else 1
            )
            # change the column name to "label"
            self.samples_frame.rename(columns={"labels": "label"}, inplace=True)
            # rename the column image name to "img"
            self.samples_frame.rename(columns={"image": "img"}, inplace=True)
            # att img/ to all element in image column
            self.samples_frame["img"] = self.samples_frame["img"].apply(
                lambda x: "img/" + x
            )
            dataset_type = "harmeme_" + dataset_type
            # re

        # Preload images and text
        self.images = []
        self.texts = []

        if load_data:
            path_images = os.path.join(datasets_directory_visual, dataset_type + ".pt")
            path_text = os.path.join(datasets_directory_text, dataset_type + ".pt")
            if not os.path.exists(path_images):
                raise Exception(f"File {path_images} does not exist")
            if not os.path.exists(path_text):
                raise Exception(f"File {path_text} does not exist")
            self.images = torch.load(path_images)
            self.texts = torch.load(path_text)
        else:
            progress_bar = tqdm.tqdm(total=len(self.samples_frame))
            for idx, row in self.samples_frame.iterrows():
                # Load and transform the image
                img_path = self.img_dir / row['img']
                image = Image.open(img_path).convert("RGB")
                image_embeddings = self.image_transform(image)
                self.images.append(image_embeddings)

                # Transform the text
                text = self.text_transform(row['text'])
                self.texts.append(text)

                progress_bar.update(1)
        if save_data:
            if not os.path.exists(datasets_directory_visual):
                os.makedirs(datasets_directory_visual)
            if not os.path.exists(datasets_directory_text):
                os.makedirs(datasets_directory_text)

            path_text = os.path.join(datasets_directory_text, dataset_type + ".pt")
            path_visual = os.path.join(datasets_directory_visual, dataset_type + ".pt")

            self.images = torch.stack(self.images)
            self.texts = torch.stack(self.texts)

            # Save the data
            torch.save(self.images, path_visual)
            torch.save(self.texts, path_text)

        print(dataset_type, "done", directory)

    def __len__(self):
        """This method is called when you do len(instance)
        for an instance of this class.
        """
        return len(self.samples_frame)

    def __getitem__(self, idx):
        """This method is called when you do instance[key]
        for an instance of this class.
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_id = self.samples_frame.loc[idx, "id"]
        image = self.images[idx]
        text = self.texts[idx]

        if "label" in self.samples_frame.columns:
            label = torch.tensor(self.samples_frame.loc[idx, "label"]).long()
            sample = {"id": img_id, "image": image, "text": text, "label": label}
        else:
            sample = {"id": img_id, "image": image, "text": text}

        return sample
