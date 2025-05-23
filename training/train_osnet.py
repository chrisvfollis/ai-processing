# standard dependencies
import os
import argparse
import warnings
import uuid
import itertools

# 3rd-party dependencies
from openpyxl import load_workbook
import pandas as pd
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from sklearn.model_selection import train_test_split

warnings.filterwarnings(
    "ignore", message="Cython evaluation",
    module="torchreid.reid.metrics.rank"
)
from torchreid import losses

# internal dependencies
from models import OSNet
from utilities import admin_utils, io_utils
from training import train_utils
from training.datasets import PKSampler, EventImgs


def run_grid_search(
        source_checkpoint: str = 'OSNet.pth.tar-250',
        num_epochs: int = 25,
        split: str = 'validate',
        hyperparam_grid: dict = {
            'triplet_margin': [0.2, 0.3, 0.5],
            'lr': [3e-4, 1e-3, 3e-3],
            'weight_decay': [0, 1e-4, 1e-3],
        }
    ):
    combinations = list(itertools.product(
        hyperparam_grid['triplet_margin'],
        hyperparam_grid['lr'],
        hyperparam_grid['weight_decay'],
    ))
    for i, combination in enumerate(combinations):
        print(f'Training with hyperparam combination {i}/{len(combinations)}...')
        hyperparams = {}
        hyperparams['triplet_margin'] = combination[0]
        hyperparams['lr'] = combination[1]
        hyperparams['weight_decay'] = combination[2]
    
        output_paths = event_img_finetune(
            source_checkpoint=source_checkpoint,
            num_epochs=num_epochs,
            split=split,
            **hyperparams,
        )
        train_utils.upload_training_files(*output_paths)


def event_img_finetune(
        source_checkpoint: str = 'OSNet.pth.tar-250',
        num_epochs: int = 25,
        split: str = 'validate',
        triplet_margin: float = 0.3,
        P: int = 4,
        K: int = 8,
        lr: float = 3e-4,
        weight_decay: float = 1e-4,
    ) -> tuple[str]:
    checkpoint_id = str(uuid.uuid4())

    project_root = io_utils.get_project_root()
    output_dir = os.path.join(project_root, 'files/output/')

    dataset_name = 'event_imgs'
    img_data_df = train_utils.clean_dataset(
        dataset=admin_utils.save_approved_img_data(),
        class_col='person_id',
        split=split,
    )
    num_classes = img_data_df['person_id'].nunique()

    model_name = 'OSNet'
    osnet = OSNet(source_checkpoint, num_classes=num_classes, mode='train')
    criterion = losses.TripletLoss(margin=triplet_margin)
    optimizer = torch.optim.Adam(
        osnet.model.parameters(), lr=lr, weight_decay=weight_decay
    )

    train_df, val_df = train_test_split(
        img_data_df,
        test_size=0.2,
        stratify=img_data_df['person_id'],
        random_state=42,
    )
    train_labels = train_df['person_id'].tolist()
    
    train_dataset = EventImgs(train_df, transform=osnet.transform)
    val_dataset = EventImgs(val_df, transform=osnet.transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=P*K,
        sampler=PKSampler(train_labels, P=P, K=K),
        shuffle=True,
        drop_last=True,
        num_workers=4,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=64,
        shuffle=False,
        drop_last=False,
        num_workers=4,
    )

    manifest_filename = f'{checkpoint_id}_manifest.csv'
    manifest_path = os.path.join(output_dir, manifest_filename)

    manifest_df = train_utils.create_dataset_manifest(
        train_df, val_df,
        output_path=manifest_path,
    )

    epoch_data = []

    for epoch in range(num_epochs):
        print(f'Starting epoch {epoch}...')
        loss = train_one_epoch(
            model=osnet.model,
            loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=osnet.device,
        )
        print(f'Epoch {epoch} - Train Loss: {loss:.4f}')
        val_dist_ratio = evaluate(osnet.model, val_loader, osnet.device)
        print(f'Epoch {epoch} — Val Distance Ratio: {val_dist_ratio:.4f}')
        
        epoch_data.append({
            'epoch': epoch,
            'train_loss': loss,
            'val_dist_ratio': val_dist_ratio,
        })

    final_epoch = epoch_data[-1]
    final_train_loss = final_epoch['train_loss']
    final_val_dist_ratio = final_epoch['val_dist_ratio']
    num_train = manifest_df.loc[manifest_df['split'] == 'train'].shape[0]
    num_val = manifest_df.loc[manifest_df['split'] == 'val'].shape[0]

    output_checkpoint = f'OSNet_{checkpoint_id}.pth'
    output_checkpoint_path = os.path.join(
        project_root, 'models/weights/', output_checkpoint
    )

    model_info = {
        'state_dict': osnet.model.state_dict(),
        'checkpoint_id': checkpoint_id,
        'model_name': model_name,
        'source_checkpoint': source_checkpoint,
    }
    dataset_info = {
        'dataset_name': dataset_name,
        'manifest_file': manifest_filename,
        'num_classes': num_classes,
        'num_train_samples': num_train,
        'num_val_samples': num_val,
    }
    hyperparameters = {
        'triplet_loss_margin': triplet_margin,
        'learning_rate': lr,
        'weight_decay': weight_decay,
        'num_epochs': num_epochs,
    }
    results = {
        'final_train_loss': final_train_loss,
        'final_val_score': final_val_dist_ratio,
    }
    checkpoint = model_info | dataset_info | hyperparameters | results
    torch.save(checkpoint, output_checkpoint_path)

    for key, value in checkpoint.items():
        if isinstance(value, (str, int, float)):
            print(f'{key}: {value}')

    epoch_data_df = pd.DataFrame(epoch_data)

    epoch_csv_path = os.path.join(output_dir, f'{checkpoint_id}_epochs.csv')
    epoch_data_df.to_csv(epoch_csv_path, index=False)

    train_utils.log_training_run(
        checkpoint_id,
        model_name,
        source_checkpoint,
        output_checkpoint,
        **dataset_info,
        **hyperparameters,
        **results,
    )
    train_utils.log_epoch_data(checkpoint_id, epoch_data)

    return output_checkpoint_path, manifest_path, epoch_csv_path


def train_one_epoch(model, loader, optimizer, criterion, device):
    batch = 0
    progress_interval = len(loader) // 4

    total_loss = 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)

        embeddings, _ = model(imgs)
        loss = criterion(embeddings, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        batch += 1

        if (batch % progress_interval) == 0:
            print(f'{batch} of {len(loader)} batches complete')
    
    return total_loss / len(loader)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()

    all_embeddings = []
    all_labels = []

    for imgs, labels in loader:
        imgs = imgs.to(device)
        output = model(imgs)

        embeddings = output[0] if isinstance(output, tuple) else output
        embeddings = F.normalize(embeddings, dim=1) # L2 normalization

        all_embeddings.append(embeddings.cpu())
        all_labels.append(labels)

    model.train()

    embeddings = torch.cat(all_embeddings)
    labels = torch.cat(all_labels)

    # Since we've L2-normalized our embedding vectors, their cosine similarities
    # are equal to their dot products. Therefore we can compute all the
    # similarity scores via matrix multiplication:
    sim_matrix = torch.mm(embeddings, embeddings.t())
    dist_matrix = 1 - sim_matrix

    same_mask = labels.unsqueeze(0) == labels.unsqueeze(1)
    diff_mask = ~same_mask

    same_dists = dist_matrix[same_mask].cpu().numpy()
    diff_dists = dist_matrix[diff_mask].cpu().numpy()

    avg_same = same_dists.mean() if len(same_dists) > 0 else float('inf')
    avg_diff = diff_dists.mean() if len(diff_dists) > 0 else float('inf')

    avg_distance_ratio = avg_same / avg_diff    # smaller is better

    print(
        f'Avg dist (same ID): {avg_same:.4f}, ' +
        f'Avg dist (different ID): {avg_diff:.4f}'
    )

    return avg_distance_ratio

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Model info:
    parser.add_argument('--checkpoint', type=str)
    # Dataset info:
    parser.add_argument('--dataset', type=str)
    # Training run:
    parser.add_argument('--num-epochs', type=int)
    parser.add_argument('--grid-search', action='store_true')
    # Hyperparameters:
    parser.add_argument('--triplet-margin', type=float)
    parser.add_argument('--lr', type=float)
    parser.add_argument('--weight-decay', type=float)

    args = parser.parse_args()

    source_checkpoint = args.checkpoint or 'OSNet.pth.tar-250'
    dataset = args.dataset or 'event_imgs'

    num_epochs = args.num_epochs or 25
    grid_search = args.num_epochs or False

    triplet_margin = args.triplet_margin or 0.3
    lr = args.lr or 3e-4
    weight_decay = args.weight_decay or 1e-4
    
    if dataset == 'event_imgs':
        if not grid_search:
            output_paths = event_img_finetune(
                source_checkpoint=source_checkpoint,
                num_epochs=num_epochs,
                triplet_margin=triplet_margin,
                lr=lr,
                weight_decay=weight_decay
            )
            train_utils.upload_training_files(*output_paths)
        elif grid_search:
            run_grid_search(
                source_checkpoint=source_checkpoint,
                num_epochs=num_epochs,
            )
