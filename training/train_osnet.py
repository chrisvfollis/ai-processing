# standard dependencies
import os
import argparse
import warnings

# 3rd-party dependencies
from openpyxl import load_workbook
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

warnings.filterwarnings(
    "ignore", message="Cython evaluation",
    module="torchreid.reid.metrics.rank"
)
from torchreid import losses

# internal dependencies
from models import OSNet
from utilities import admin_utils, io_utils
from training import datasets, train_utils


def event_img_finetune(
        weights_file: str = 'OSNet.pth.tar-250',
        num_epochs: int = 20,
        split: str = 'validate'
    ) -> pd.DataFrame:
    project_root = io_utils.get_project_root()

    img_data_df = admin_utils.save_approved_img_data()
    img_data_df = train_utils.clean_dataset(
        dataset=img_data_df,
        class_col='person_id',
        split=split,
    )
    num_classes = img_data_df['person_id'].nunique()

    osnet = OSNet(weights_file, num_classes=num_classes, mode='train')

    train_df, val_df = train_test_split(
        img_data_df,
        test_size=0.2,
        stratify=img_data_df['person_id'],
        random_state=42,
    )
    dataset_manifest_path = os.path.join(
        project_root, 'files/output/', 'OSNet_training_manifest.csv'
    )
    train_utils.save_dataset_manifest(
        train_df, val_df, output_path=dataset_manifest_path
    )

    train_dataset = datasets.EventImgs(train_df, transform=osnet.transform)
    val_dataset = datasets.EventImgs(val_df, transform=osnet.transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True,
        num_workers=4,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=4,
        drop_last=False,
    )

    criterion = losses.TripletLoss(margin=0.3)
    optimizer = torch.optim.Adam(
        osnet.model.parameters(),
        lr=3.5e-4,
        weight_decay=5e-4
    )

    training_run_data = []

    for epoch in range(num_epochs):
        print(f'Starting epoch {epoch}...')
        loss = train_one_epoch(
            model=osnet.model,
            loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=osnet.device,
        )
        val_loss = evaluate(osnet.model, val_loader, criterion, osnet.device)
        print(f'Epoch {epoch} - Train Loss: {loss:.4f}, Val Loss: {val_loss:.4f}')

        training_run_data.append({
            'epoch': epoch,
            'train_loss': loss,
            'val_loss': val_loss,
        })

    output_path = os.path.join(
        project_root, 'models/weights/', 'OSNet_finetuned.pth.tar'
    )
    torch.save({'state_dict': osnet.model.state_dict()}, output_path)

    training_run_df = pd.DataFrame(training_run_data)

    spreadsheet_path = os.path.join(
        project_root, 'files/output/', 'OSNet_training_run.xlsx'
    )
    with pd.ExcelWriter(spreadsheet_path, engine='openpyxl', mode='w') as writer:
        training_run_df.to_excel(writer, sheet_name='Epochs', index=False)

    return training_run_df


def train_one_epoch(model, loader, optimizer, criterion, device):
    batch = 0
    progress_interval = len(loader) // 4

    total_loss = 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)

        feats, _ = model(imgs)
        loss = criterion(feats, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        batch += 1

        if (batch % progress_interval) == 0:
            print(f'{batch} of {len(loader)} batches complete')
    
    return total_loss / len(loader)


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()

    total_loss = 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        output = model(imgs)

        if isinstance(output, tuple):
            feats = output[0]
        else:
            feats = output

        loss = criterion(feats, labels)
        total_loss += loss.item()

    model.train()
    return total_loss / len(loader)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights-file', type=str)
    parser.add_argument('--num-epochs', type=int)
    parser.add_argument('--dataset', type=str)

    args = parser.parse_args()

    weights_file = args.weights_file or 'OSNet.pth.tar-250'
    num_epochs = args.num_epochs or 20
    dataset = args.dataset or 'event_imgs'

    if dataset == 'event_imgs':
        training_run_data = event_img_finetune(
            weights_file=weights_file,
            num_epochs=num_epochs
        )
