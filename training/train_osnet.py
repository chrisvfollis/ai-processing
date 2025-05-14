# standard dependencies
import os
import argparse
import warnings

# 3rd-party dependencies
import torch
from torch.utils.data import DataLoader

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
    num_classes = img_data_df['person_id'].nunique()

    project_root = io_utils.get_project_root()
    weights_path = os.path.join(project_root, 'models/weights/', 'OSNet.pth.tar-250')

    device = torch.device('gpu:0' if torch.cuda.is_available() else 'cpu')
    osnet = OSNet(weights_path, device, num_classes=num_classes, mode='train')

    train_dataset = datasets.EventImgs(img_data_df, transform=osnet.transform)
    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True,
        num_workers=4,
        drop_last=True,
    )

    criterion = losses.TripletLoss(margin=0.3)
    optimizer = torch.optim.Adam(
        osnet.model.parameters(),
        lr=3.5e-4,
        weight_decay=5e-4
    )

    for epoch in range(num_epochs):
        print(f'Starting epoch {epoch+1}...')
        loss = train_one_epoch(
            model=osnet.model,
            loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
        )
        print(f'Epoch {epoch+1}/{num_epochs} - Loss: {loss:.4f}')

    output_path = os.path.join(project_root, 'models/weights/', 'OSNet_finetuned.pth.tar')
    torch.save({'state_dict': osnet.model.state_dict()}, output_path)


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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-epochs', type=int)
    parser.add_argument('--dataset', type=str)

    args = parser.parse_args()

    num_epochs = args.num_epochs or 20
    dataset = args.dataset or 'event_imgs'

    if dataset == 'event_imgs':
        event_img_finetune(num_epochs=num_epochs)
