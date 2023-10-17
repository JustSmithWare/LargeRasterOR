import pytorch_lightning as pl
import torch
from torchvision.models.detection.retinanet import retinanet_resnet50_fpn
from torch.utils.data import DataLoader, Dataset
from torch.optim import SGD
import pandas as pd
import numpy as np
from torchvision.ops import box_iou
from typing import Optional, Any
from BboxDataset import BboxDataset
from torchvision.transforms import ToTensor

from ModelConfig import ModelConfig

class LargeRasterORModel(pl.LightningModule):
    """
    Pytorch Lightning Module for performing object recognition on large raster images.

    Parameters
    ----------
    label_set : set
        A set of unique string identifiers for labels to be recognized.
    """
    def __init__(self, label_set: str) -> None:
        super(LargeRasterORModel, self).__init__()
        self.model = retinanet_resnet50_fpn(pretrained=False, num_classes=len(label_set))
        
        self.conf = ModelConfig() 

    def training_step(self, batch: tuple, _: int) -> dict[str, float]:
        """
        Performs a single training step on the model.

        Logs training losses on bounding box regression and categry classification.

        Parameters:
        - batch (tuple): A batch of data consisting of images and targets.
        - _ (int): Batch index, not used.

        Returns:
        - dict[str, float]: A dictionary containing the `loss` key, and the sum of the losses reached during training as a value.
        """
        self.model.train()
        images, targets = batch

        loss_dict = self.model.forward(images, targets)
        
        losses = sum(loss for loss in loss_dict.values())

        for k, v in loss_dict.items():
            self.log(f'train_{k}', v, on_epoch=True, prog_bar=True, batch_size=self.conf.batch_size)

        return {'loss': losses}
    
    def validation_step(self, batch: tuple, _: int) -> dict[str, float]:
        """
        Performs a single validation step on the model.

        Logs validation losses on bounding box regression and categry classification.

        Parameters:
        - batch (tuple): A batch of data consisting of images and targets.
        - _ (int): Batch index, not used.

        Returns:
        - dict[str, float]: A dictionary containing the `val_loss` key, and the sum of the losses reached during validation as a value.
        """
        images, targets = batch

        self.model.train()
        with torch.no_grad():
            loss_dict = self.model.forward(images, targets)
        
        val_loss = sum(loss for loss in loss_dict.values())

        for key, val in loss_dict.items():
            self.log(key, val, on_epoch=True, batch_size=self.conf.batch_size)

        self.log('val_loss', val_loss, on_epoch=True)

        return {'val_loss': val_loss}
    
    def test_step(self, batch: tuple, _: int) -> dict[str, float]:
        """
        Performs a single test step on the model.

        Logs test losses on bounding box regression and categry classification.

        Parameters:
        - batch (tuple): A batch of data consisting of images and targets.
        - _ (int): Batch index, not used.

        Returns:
        - dict[str, float]: A dictionary containing the `test_loss` key, and the sum of the losses reached during testing as a value.
        """
        images, targets = batch
        self.model.train()
        
        with torch.no_grad():
            loss_dict = self.model.forward(images, targets)
        
        test_loss = sum(loss for loss in loss_dict.values())
        
        for key, val in loss_dict.items():
            self.log(f'test_{key}', val, batch_size=self.conf.batch_size)
        
        self.log('test_loss', test_loss, on_epoch=True)
        
        return {'test_loss': test_loss}
    
    def predict_step(self, batch: tuple, _: int) -> pd.DataFrame:
        """
        Performs a single prediction step on the model.

        Does not log losses and returns the predicted bounding boxes and their labels.

        Parameters:
        - batch (tuple): A batch of data consisting of images and targets.
        - _ (int): Batch index, not used.

        Returns:
        - pd.Dataframe: A pandas Dtaframe object containing the boxes and labels predicted by the model.
        """
        images, _ = batch

        predictions = self.model(images)
        
        all_boxes = []
        all_labels = []
        for pred in predictions:
            all_boxes.extend(pred['boxes'].cpu().numpy())
            all_labels.extend(pred['labels'].cpu().numpy())
        
        df = pd.DataFrame({
            'boxes': all_boxes,
            'labels': all_labels
        })
    
        return df

    def configure_optimizers(self) -> SGD:
        '''
        Defines a simple optimizer for a RetinaNet model.
        Uses the model's configuration learning rate value.
        '''
        optimizer = SGD(self.model.parameters(), lr=self.conf.train.lr, momentum=0.9, weight_decay=0.0005)
        return optimizer


    def create_trainer(self, logger: Optional[Any] = None, callbacks: Optional[list[pl.Callback]] = None, **kwargs: Any) -> None:
        '''
        Creates a Pytorch Lightning trainer.
        Takes Pytorch lightning callbacks, a logger, accelerator and devices from the model's configuration.
        '''
        self.trainer = pl.Trainer(
            max_epochs=self.conf.train.epochs,
            enable_checkpointing=True,
            devices=self.conf.devices,
            logger=logger,
            callbacks=callbacks,
            accelerator=self.conf.accelerator,
            num_sanity_val_steps=0,
            **kwargs
        )

    def on_validation_epoch_end(self) -> None:
        '''
        Callback for Pytorch Lightning to call after a validation epoch ends.

        If the epoch number is multiple of the configuration value pr_compute_interval,
        calculates bounding box and label precision and recall based on the `iou_threshold` set up in self.conf.
        '''
        outputs = self.trainer.callback_metrics

        for key, value in outputs.items():
            self.log(key, float(value))

        print(f'Epoch {self.current_epoch}, Avg Validation Loss: {outputs["val_loss"]}')


        if self.trainer.current_epoch % self.conf.validation.pr_compute_interval == 0:
            print('Evaluating Validation Predictions...')
            validation_set = BboxDataset(csv_file=self.conf.validation.csv_file, root_dir=self.conf.validation.root_dir, transform=ToTensor())
            pr_results = self.predict_data(self.predict_dataloader(validation_set))

            for key, value in pr_results.items():
                self.log(key, float(value), on_epoch=True)

    def on_test_epoch_end(self) -> None:
        '''
        Callback for Pytorch Lightning to call after a testing epoch ends.

        Calculates bounding box and label precision and recall based on the `iou_threshold` set up in self.conf and logs the values.
        '''
        print('Evaluating Test Predictions...')
        test_set = BboxDataset(csv_file=self.conf.test.csv_file, root_dir=self.conf.test.root_dir, transform=ToTensor())
        pr_results = self.predict_data(self.predict_dataloader(test_set))

        for key, value in pr_results.items():
            self.log(f'test_{key}', float(value), on_epoch=True)

    def predict_data(self, dataloader: DataLoader) -> dict[str, float]:
        '''
        Predicts data from a dataloader using the model.

        Parameters:
        - dataloader (DataLoader): The DataLoader object containing the data to be predicted.

        Returns:
        - pr_results (Dict[str, float]): A dictionary containing precision and recall scores for bounding box regression and label classification.
        '''
        self.model.eval()
        self.model.score_thresh = self.conf.test.score_threshold

        all_dfs = []
        batched_ground_truths = []

        for batch_idx, batch in enumerate(dataloader):
            batch = self.transfer_batch_to_device(batch, self.device, dataloader_idx=0)
            _, targets = batch  

            df = self.predict_step(batch, batch_idx)
            
            df['batch_idx'] = batch_idx
            all_dfs.append(df)

            gt_boxes = targets[0]['boxes'].cpu().numpy()
            gt_labels = targets[0]['labels'].cpu().numpy()

            batch_ground_truth = {'boxes': gt_boxes.tolist(), 'labels': gt_labels.tolist()}
            batched_ground_truths.append(batch_ground_truth)

        if all(df.empty for df in all_dfs):
            print('No predictions from predict_step')
            return {
                'precision_box': 0,
                'recall_box': 0,
                'precision_label': 0,
                'recall_label': 0
            }

        combined_df = pd.concat(all_dfs, ignore_index=True)
        predictions = combined_df.groupby('batch_idx').agg(list).reset_index()

        ground_truths = pd.DataFrame(batched_ground_truths)
        
        pr_results = self._precision_recall_score(ground_truths, predictions, self.conf.test.iou_threshold, device=self.device)

        return pr_results

    def _precision_recall_score(self, ground_truths, predictions, iou_threshold=0.5, device='cpu') -> dict[str, float]:
        """
        Calculate precision and recall scores for bounding boxes and labels.

        Parameters:
        - ground_truths (pd.DataFrame): The ground truths, must be a Dataframe containing `labels` and `boxes`.
        - predictions (pd.DataFrame): The predictions, must be a Dataframe containing `labels` and `boxes`.
        - iou_threshold (float): If IOU value between a ground truth box and a predicted box is under this value, 
            then it is not considered a considered a correct prediction when calculating precision and recall.
        - device (str): The device to run calculations on, defaults to 'cpu'.

        Returns:
        - pr_scores (Dict[str, float]): A dictionary containing precision and recall scores for bounding boxes and labels.
        """
        TP_box, FP_box, FN_box = 0, 0, 0
        TP_label, FP_label, FN_label = 0, 0, 0

        for idx in range(len(ground_truths)):
            gt_boxes = torch.tensor(ground_truths.iloc[idx]['boxes'], dtype=torch.float32, device=device).reshape(-1, 4)
            gt_labels = np.array(ground_truths.iloc[idx]['labels'])

            pred_boxes = torch.tensor(predictions.iloc[idx]['boxes'], dtype=torch.float32, device=device).reshape(-1, 4)
            pred_labels = np.array(predictions.iloc[idx]['labels'])

            iou_matrix = box_iou(gt_boxes, pred_boxes)
            matched_preds = set()

            for i, (gt_box, gt_label) in enumerate(zip(gt_boxes, gt_labels)):
                max_iou, max_j = iou_matrix[i].max(0)
                if max_iou >= iou_threshold:
                    TP_box += 1
                    matched_preds.add(max_j.item())
                    if np.isin(pred_labels[max_j.item()], gt_label):
                        TP_label += 1

            FN_box += len(gt_boxes) - TP_box
            FN_label += len(gt_labels) - TP_label

            FP_box += len(pred_boxes) - len(matched_preds)
            FP_label += len(pred_labels) - len(matched_preds)

        precision_box = TP_box / (TP_box + FP_box) if TP_box + FP_box > 0 else 0
        recall_box = TP_box / (TP_box + FN_box) if TP_box + FN_box > 0 else 0
        precision_label = TP_label / (TP_label + FP_label) if TP_label + FP_label > 0 else 0
        recall_label = TP_label / (TP_label + FN_label) if TP_label + FN_label > 0 else 0

        return {
            'precision_box': precision_box,
            'recall_box': recall_box,
            'precision_label': precision_label,
            'recall_label': recall_label
        }

        
    def train_dataloader(self) -> DataLoader:
        """
        DataLoader for the training set.

        Creates a Dataset from the data found in `self.conf.training.csv_file`
        Returns:
        - dataloader (DataLoader): DataLoader object for the training set.
        """
        bbox_dataset = BboxDataset(csv_file=self.conf.train.csv_file, root_dir=self.conf.train.root_dir, transform=ToTensor())
        return DataLoader(bbox_dataset, batch_size=self.conf.batch_size, collate_fn=lambda batch: tuple(zip(*batch)), shuffle=True, num_workers=self.conf.workers)

    def val_dataloader(self) -> DataLoader:
        """
        DataLoader for the validation set.

        Creates a Dataset from the data found in `self.conf.validation.csv_file`
        Returns:
        - dataloader (DataLoader): DataLoader object for the validation set.
        """
        bbox_dataset = BboxDataset(csv_file=self.conf.validation.csv_file, root_dir=self.conf.validation.root_dir, transform=ToTensor())
        return DataLoader(bbox_dataset, batch_size=self.conf.batch_size, collate_fn=lambda batch: tuple(zip(*batch)), shuffle=True, num_workers=self.conf.workers)
    
    def test_dataloader(self) -> DataLoader:
        """
        DataLoader for the testing set.

        Creates a Dataset from the data found in `self.conf.test.csv_file`
        Returns:
        - dataloader (DataLoader): DataLoader object for the testing set.
        """
        bbox_dataset = BboxDataset(csv_file=self.conf.test.csv_file, root_dir=self.conf.test.root_dir, transform=ToTensor())
        return DataLoader(bbox_dataset, batch_size=self.conf.batch_size, collate_fn=lambda batch: tuple(zip(*batch)), shuffle=False, num_workers=self.conf.workers)

    def predict_dataloader(self, dataset: Dataset) -> DataLoader:
        """
        DataLoader made from a given Dataset's data.

        Parameters:
        - dataset (Any): The dataset object.

        Returns:
        - dataloader (Dataset): DataLoader object for the training set.
        """
        return DataLoader(dataset, batch_size=self.conf.batch_size, shuffle=False, num_workers=self.conf.workers, collate_fn=lambda batch: tuple(zip(*batch)))
    