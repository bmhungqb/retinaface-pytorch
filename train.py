import os
import time
import random
import logging
import numpy as np
import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader

from config import get_config
from models import RetinaFace
from layers import PriorBox, MultiBoxLoss
from utils.dataset import WiderFaceDetection
from utils.transform import Augmentation


def parse_args():
    import argparse

    parser = argparse.ArgumentParser(description='Training Arguments for RetinaFace')
    parser.add_argument(
        '--train-data',
        type=str,
        default='./data/widerface/train',
        help='Path to the training dataset directory.'
    )
    parser.add_argument(
        '--network',
        type=str,
        default='resnet34',
        choices=[
            'mobilenetv1', 'mobilenetv1_0.25', 'mobilenetv1_0.50',
            'mobilenetv2', 'mobilenetv2_0.25', 'resnet50', 'resnet34', 'resnet18'
        ],
        help='Backbone network architecture to use'
    )
    parser.add_argument('--num-workers', default=8, type=int, help='Number of workers to use for data loading.')
    parser.add_argument('--batch-size', default=32, type=int, help='Batch size.')
    parser.add_argument('--print-freq', type=int, default=50, help='Print frequency during training.')
    parser.add_argument('--learning-rate', default=1e-3, type=float, help='Initial learning rate.')
    parser.add_argument('--lr-warmup-epochs', type=int, default=1, help='Number of warmup epochs.')
    parser.add_argument('--power', type=float, default=0.9, help='Power for learning rate policy.')
    parser.add_argument('--momentum', default=0.9, type=float, help='Momentum for SGD optimizer.')
    parser.add_argument('--weight-decay', default=5e-4, type=float, help='Weight decay (L2 penalty).')
    parser.add_argument('--gamma', default=0.1, type=float, help='Gamma update for SGD.')
    parser.add_argument(
        '--save-dir',
        default='./weights',
        type=str,
        help='Directory where trained model checkpoints will be saved.'
    )
    parser.add_argument(
        '--resume',
        action='store_true',
        help='Resume training from checkpoint')
    return parser.parse_args()


def setup_logging(save_dir, network):
    os.makedirs(save_dir, exist_ok=True)
    log_file = os.path.join(save_dir, f'training_{network}.log')
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s: %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    logging.info('Logging initialized')


def random_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def train_one_epoch(
    model,
    criterion,
    optimizer,
    data_loader,
    epoch,
    device,
    print_freq=50
) -> float:
    model.train()
    running_loss = 0.0
    for batch_idx, (images, targets) in enumerate(data_loader):
        start_time = time.time()
        images = images.to(device)
        targets = [t.to(device) for t in targets]

        outputs = model(images)
        loss_loc, loss_conf, loss_land = criterion(outputs, targets)
        loss = cfg['loc_weight'] * loss_loc + loss_conf + loss_land

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        if (batch_idx + 1) % print_freq == 0:
            lr = optimizer.param_groups[0]['lr']
            msg = (f"Epoch [{epoch+1}/{cfg['epochs']}], Step [{batch_idx+1}/{len(data_loader)}], "
                   f"Loss loc: {loss_loc.item():.4f}, conf: {loss_conf.item():.4f}, "
                   f"land: {loss_land.item():.4f}, lr: {lr:.6f}, "
                   f"time: {(time.time()-start_time):.3f}s")
            logging.info(msg)

    avg_loss = running_loss / len(data_loader)
    logging.info(f"Epoch {epoch+1} Average Loss: {avg_loss:.4f}")
    return avg_loss


def plot_losses(losses, save_dir):
    plt.figure()
    plt.plot(range(1, len(losses)+1), losses)
    plt.xlabel('Epoch')
    plt.ylabel('Average Loss')
    plt.title('Training Loss per Epoch')
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, 'loss_curve.png'))
    plt.close()


def main(params):
    global cfg
    cfg = get_config(params.network)
    random_seed()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    setup_logging(params.save_dir, params.network)
    logging.info(f"Using device: {device}")

    # Data
    dataset = WiderFaceDetection(params.train_data, Augmentation(cfg['image_size'], rgb_mean))
    data_loader = DataLoader(
        dataset,
        batch_size=params.batch_size,
        shuffle=True,
        num_workers=params.num_workers,
        collate_fn=dataset.collate_fn,
        pin_memory=True,
        drop_last=True
    )
    logging.info('Data loaded')

    # Priors and loss
    priorbox = PriorBox(cfg, image_size=(cfg['image_size'], cfg['image_size']))
    priors = priorbox.generate_anchors().to(device)
    criterion = MultiBoxLoss(priors, threshold=0.35, neg_pos_ratio=7, variance=cfg['variance'], device=device)

    # Model and optimizer
    model = RetinaFace(cfg).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=params.learning_rate,
                                momentum=params.momentum, weight_decay=params.weight_decay)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=cfg['milestones'], gamma=params.gamma)

    start_epoch = 0
    if params.resume:
        ckpt_path = os.path.join(params.save_dir, f"{params.network}_checkpoint.ckpt")
        if os.path.exists(ckpt_path):
            checkpoint = torch.load(ckpt_path, map_location='cpu')
            model.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler.load_state_dict(checkpoint['lr_scheduler'])
            start_epoch = checkpoint['epoch'] + 1
            logging.info(f"Resumed from epoch {start_epoch}")

    epoch_losses = []
    logging.info('Starting training')
    for epoch in range(start_epoch, cfg['epochs']):
        avg_loss = train_one_epoch(
            model, criterion, optimizer, data_loader,
            epoch, device, params.print_freq
        )
        epoch_losses.append(avg_loss)

        scheduler.step()

        # Save checkpoints
        ckpt = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                'lr_scheduler': scheduler.state_dict(), 'epoch': epoch}
        torch.save(ckpt, os.path.join(params.save_dir, f"{params.network}_checkpoint.ckpt"))
        torch.save(model.state_dict(), os.path.join(params.save_dir, f"{params.network}_last.pth"))

        # Plot every 20 epochs
        if (epoch + 1) % 20 == 0:
            plot_losses(epoch_losses, params.save_dir)

    # Final plot
    plot_losses(epoch_losses, params.save_dir)
    torch.save(model.state_dict(), os.path.join(params.save_dir, f"{params.network}_final.pth"))
    logging.info('Training complete')


if __name__ == '__main__':
    args = parse_args()
    rgb_mean = (104, 117, 123)
    main(args)
