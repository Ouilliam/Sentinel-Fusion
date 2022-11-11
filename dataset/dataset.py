"""
    ./dataset.py

        This file contains the definition of the NDVI Disease Dataset which was used to train several
        models on detecting the diseases thanks to NDVI.
    
    Author: William MAILLET 
"""
import torch
from torch.utils.data import Dataset
import os 
import skimage
from torchvision import transforms
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np


class NDVIDiseaseDataset(Dataset):
    """
        The NDVI Classes Dataset, with for each image as label the presence 
        (or not) of the mildew disease.

        The dataset is composed of images and labels stored in the root directory.

        Parameters:
        -----------
        root_dir: str
            The root directory of the dataset.
        annotations: str
            The path to the csv file containing the annotations.
        transform: torchvision.transforms
            The transform to apply to the images.
    """

    def __init__(self, root_dir: str, train: bool, transform=None):
        self.root_dir = root_dir
        self.vi_dirs = ['NDVI-RAINBOW', 'NDCI', 'MOISTURE']
        self.train = train

        # Annotations
        self.ann = pd.read_csv(
            os.path.join(root_dir, "train_annotations.csv" if self.train else "test_annotations.csv"))

        # Weather data
        self.weather_data = pd.read_csv(
            os.path.join(root_dir, "weather/weather.csv"), header=0)
        self.weather_data = self._init_weather(self.weather_data)

        # Image transform
        if train:
            self.img_tf = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                                    0.229, 0.224, 0.225]),
                transforms.Resize((224, 224)),
                transforms.CenterCrop(192),
                transforms.Resize((224, 224)),
                # data augmentation
                transforms.ColorJitter(),
                transforms.RandomRotation(15),
                transforms.RandomInvert(),
                transforms.RandomAutocontrast(),
            ])
        else:
            self.img_tf = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                                    0.229, 0.224, 0.225]),
                transforms.Resize((224, 224)),
                transforms.CenterCrop(192),
                transforms.Resize((224, 224))
            ])

    def _init_weather(self, weather_df):
        """
            Initializes the weather dataframe by normalizing/scaling
            the data in the dataframe. This functions also drops useless
            columns.

            Parameters:
            -----------
            weather_df: pd.DataFrame
                The weather dataframe.

            Returns:
            --------
            weather_df: pd.DataFrame 
                The scaled dataframe
        """
        dates = pd.Series(weather_df['date'])

        # Scaling dataframe
        scaled = MinMaxScaler().fit_transform(weather_df.drop(['date'], axis=1).values)

        # Reconstructing dataframe
        weather_df = pd.concat([
            dates,
            pd.DataFrame(
                scaled,
                columns=weather_df.drop(['date'], axis=1).columns
            ).drop(['day', 'month', 'year'], axis=1)],
            axis=1)

        return weather_df

    def _get_image(self, img_paths):
        """
            Transforms and prepare images for the model.

            Parameters:
            -----------
            img_paths: str
                The paths to the images

            Returns:
            --------
            image: np.array
                The transformed image.
        """
        img_list = []
        # Loading image as RGB
        for path in img_paths:
            img_list.append(skimage.color.rgba2rgb(skimage.io.imread(path)).astype(np.float32))
            
        # Apply transforms
        for i in range(len(img_list)):
            img_list[i] = self.img_tf(img_list[i])

        # Concatenate 
        image = np.concatenate(img_list, axis=0)
        return image
    
    def _get_weather_vector(self, img_date):
        """
            Returns the weather vector for a given date.

            Parameters:
            -----------
            img_date: str
                The date of the image.

            Returns:
            --------
            weather_vector: np.array
                The weather vector for the given date.
        """
        #print(self.weather_data['date'])
        #print(pd.to_datetime(img_date,format ="%Y-%m-%d"))
        return self.weather_data.loc[self.weather_data['date']
                                     == img_date, :].values[0][1:].astype(np.float32).reshape(1, -1)

    def __len__(self):
        """
            Returns the length of the dataset.
        """
        if self.train:
            return len(os.listdir(os.path.join(self.root_dir, "images", "train", self.vi_dirs[0])))
        else:
            return len(os.listdir(os.path.join(self.root_dir, "images", "test", self.vi_dirs[0])))

    def __getitem__(self, idx):
        """
            Returns a dictionary containing the image, weather and target vectors.
            The dict is composed of 3 keys: 'image', 'weather' and 'target'.

            Parameters:
            -----------
            idx: int
                The index of the dataset entry to return 
            
            Returns:
            --------
            sample: dict 
                The sample data
        """
        # Transforming to list in case of single image
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Getting image
        img_paths = [os.path.join(self.root_dir, "images", "train" if self.train else "test", vi_dir, self.ann.iloc[idx, 0]) for vi_dir in self.vi_dirs]
        image = self._get_image(img_paths)
        #print(image.size)

        # Getting weather vector
        img_date = img_paths[0].split("/")[-1].split(".")[0].split("_")[0]
        #print(img_date)
        weather_vector = self._get_weather_vector(img_date)
        #print(weather_vector)

        # Label is one hot with 1 where the class with max value is
        #print(np.array(self.ann.iloc[idx, 1:]))
        target = np.array(self.ann.iloc[idx, 1:], dtype=np.float32)

        sample = {
            'image': torch.from_numpy(image).float(),
            'weather': torch.from_numpy(weather_vector).float(),
            'target': torch.from_numpy(target).float()
        }

        return sample

    def _get_img_name(self, idx):
        """
            Returns the name of the image.
                
            Parameters:
            -----------
            idx: int
                The index of the dataset entry to return 
            
            Returns:
            --------
            img_name: str
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        return self.ann.iloc[idx, 0]