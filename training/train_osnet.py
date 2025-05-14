# standard dependencies
import os
import argparse
import warnings

# 3rd-party dependencies
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
from training import datasets


def event_img_finetune(num_epochs: int = 20):
    img_data_df = admin_utils.save_approved_img_data()

    img_counts = img_data_df['person_id'].value_counts()
    valid_ids = img_counts[img_counts >= 2].index
    img_data_df = img_data_df[img_data_df['person_id'].isin(valid_ids)]

    num_classes = img_data_df['person_id'].nunique()

    train_df, val_df = train_test_split(
        img_data_df,
        test_size=0.2,
        stratify=img_data_df['person_id'],
        random_state=42,
    )

    project_root = io_utils.get_project_root()
    weights_path = os.path.join(project_root, 'models/weights/', 'OSNet.pth.tar-250')

    device = torch.device('gpu:0' if torch.cuda.is_available() else 'cpu')
    osnet = OSNet(weights_path, device, num_classes=num_classes, mode='train')

    train_dataset = datasets.EventImgs(train_df, transform=osnet.transform)
    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True,
        num_workers=4,
        drop_last=True,
    )
    val_dataset = datasets.EventImgs(val_df, transform=osnet.transform)
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
            device=device,
        )
        val_loss = evaluate(osnet.model, val_loader, criterion, device)
        print(f'Epoch {epoch} - Train Loss: {loss:.4f}, Val Loss: {val_loss:.4f}')

        training_run_data.append({
            'epoch': epoch,
            'train_loss': loss,
            'val_loss': val_loss,
        })

    output_path = os.path.join(project_root, 'models/weights/', 'OSNet_finetuned.pth.tar')
    torch.save({'state_dict': osnet.model.state_dict()}, output_path)

    return pd.DataFrame(training_run_data)


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
    parser.add_argument('--num-epochs', type=int)
    parser.add_argument('--dataset', type=str)

    args = parser.parse_args()

    num_epochs = args.num_epochs or 20
    dataset = args.dataset or 'event_imgs'

    if dataset == 'event_imgs':
        event_img_finetune(num_epochs=num_epochs)
