import argparse
import logging
import os
import warnings
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.init as init
import wandb
from torch import optim
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

from model.Classifier import train_class_drbms
from model.SAFusionNet import SAFusionNet as net
from model.utils import DoubleDataset
from model.utils import FocalLoss
from model.utils import MultiLabelContrastiveLoss as MulConLoss

model = net(num_classes=8)


def get_args():
    parser = argparse.ArgumentParser(description='Train')
    parser.add_argument('--epochs', '-e', metavar='E', type=int, default=100, help='Number of epochs')
    parser.add_argument('--batch-size', '-b', dest='batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--virtual-batch', '-vb', type=int, default=16, help='Virtual batch size')
    parser.add_argument('--learning-rate', '-l', metavar='LR', type=float, default=1e-4,dest='lr')
    parser.add_argument('--load', '-f', type=str,default=r"", help='Load model from a .pth file')
    parser.add_argument('--con-loss', '-con', type=bool, default=True, help='Using Contrastive Loss')
    parser.add_argument('--amp', action='store_true', default=False, help='Use mixed precision')
    parser.add_argument('--classes', '-c', type=int, default=8, help='Number of classes')
    parser.add_argument('--save_checkpoint', type=bool, default=True, help='Save model')
    parser.add_argument('--eval_times', type=int, default=0, help='Evaluate times every epoch')

    return parser.parse_args()


def load_checkpoint(model, checkpoint_path):
    try:
        state_dict = torch.load(checkpoint_path, map_location=device)
    except RuntimeError:
        state_dict = torch.load(checkpoint_path, map_location="cpu")

    missing_keys = []
    unmatched_keys = []
    if 'mask_values' in state_dict:
        del state_dict['mask_values']
    for k in list(state_dict.keys()):
        if k not in model.state_dict():
            missing_keys.append(k)
        elif state_dict[k].shape != model.state_dict()[k].shape:
            unmatched_keys.append(k)
    for k in missing_keys:
        del state_dict[k]
        logging.info(f"Delete missing keys:  {k}.")
    for k in unmatched_keys:
        del state_dict[k]
        logging.info(f"Delete unmatched keys:  {k}.")
    logging.info(f'Model loaded from {args.load}')
    logging.info(model.load_state_dict(state_dict, strict=False))
    # freeze_parameters = list(state_dict.keys())


def train_model(
        model,
        device,
        epochs: int = 5,
        batch_size: int = 1,
        learning_rate: float = 1e-5,
        save_checkpoint: bool = True,
        contrastive: bool = False,
        amp: bool = True,
        weight_decay: float = 1e-8,
        momentum: float = 0.999,
        gradient_clipping: float = 1.0
):

    dir_checkpoint = Path('./checkpoints/')
    root_path = os.path.abspath("")
    train_dataset = DoubleDataset(images_dir=os.path.join(root_path, "dataset/train"),
                                  label_path=os.path.join(root_path, "dataset/train.xlsx"), resize=224)
    val_dataset = DoubleDataset(images_dir=os.path.join(root_path, "dataset/test"),
                                label_path=os.path.join(root_path, "dataset/test.xlsx"), resize=224)

    # 3. Create data loaders
    loader_args_train = dict(
        batch_size=batch_size,
        num_workers=os.cpu_count(),
        pin_memory=True,
        persistent_workers=True  # 避免 DataLoader 进程重启
    )

    loader_args_val = dict(
        batch_size=batch_size,
        num_workers=0,
        pin_memory=False,
        persistent_workers=False
    )
    train_loader = DataLoader(train_dataset, shuffle=True, **loader_args_train)
    val_loader = DataLoader(val_dataset, shuffle=False, drop_last=True, **loader_args_val)

    # (Initialize logging)
    experiment = wandb.init(project='SACon-Net', resume='allow', anonymous='must')
    experiment.config.update(
        dict(epochs=epochs, batch_size=batch_size, learning_rate=learning_rate,
             save_checkpoint=save_checkpoint, amp=amp)
    )

    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {learning_rate}
        Training size:   {len(train_dataset)}
        Validation size: {len(val_dataset)}
        Save checkpoints:{save_checkpoint}
        Device:          {device.type}
        Mixed Precision: {amp}
    ''')
    if save_checkpoint is False:
        logging.warning('Checkpoint will not be saved!')

    # 4. Set up the optimizer, the loss, the learning rate scheduler and the loss scaling for AMP
    for name, para in model.named_parameters():
        if name in freeze_parameters:
            para.requires_grad_(False)
            logging.info(f'Freeze {name}')

    model_optimizer = torch.optim.AdamW(
        [{'params': model.main.parameters(), 'lr': args.lr, "weight_decay": weight_decay},
         {'params': model.drbm.binary_mlp.parameters(), 'lr': args.lr, "weight_decay": weight_decay},
         {'params': model.drbm.all_mlp.parameters(), 'lr': args.lr, "weight_decay": weight_decay},
         {'params': model.drbm.mlp_abnormal .parameters(), 'lr': args.lr, "weight_decay": weight_decay}])
    drbm_optimizer = torch.optim.Adam([
        {'params': model.drbm.class_drbms.parameters(), 'lr': args.lr * 5, 'weight_decay': weight_decay},
        {'params': model.drbm.abnormal_drbm1.parameters(), 'lr': args.lr * 5, 'weight_decay': weight_decay},
        {'params': model.drbm.abnormal_drbm2.parameters(), 'lr': args.lr * 5, 'weight_decay': weight_decay},
    ])

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(model_optimizer, 'max', factor=0.6, patience=3,
                                                     min_lr=1e-8)  # goal: maximize Dice score # goal: maximize Dice score
    grad_scaler = GradScaler('cuda')
    con_lossfunc = MulConLoss()
    focal_lossfunc = FocalLoss(alpha=0.5, gamma=2)
    bce_lossfunc = nn.BCELoss()
    global_step = 0
    con_weight = 0.001
    # 5. Begin training
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0
        update_step = args.virtual_batch // batch_size
        update_epoch = [i for i in range(update_step, len(train_loader) + 1, update_step)]
        update_epoch.append(len(train_loader))

        with tqdm(total=len(train_dataset), desc=f'Epoch {epoch}/{epochs}', unit='img') as pbar:
            for epoch_step, batch in enumerate(train_loader):

                torch.cuda.empty_cache()
                epoch_step += 1

                l_img, r_img, labels = batch['left_image'], batch['right_image'], batch['labels']

                if l_img.shape[0] <= 1:
                    continue

                l_img = l_img.to(device=device, memory_format=torch.channels_last)
                r_img = r_img.to(device=device, memory_format=torch.channels_last)
                labels = labels.to(device=device)
                binary_labels = torch.stack([labels[:, 0], 1 - labels[:, 0]], dim=1)  # [batch, 2]

                with autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
                    label_emd = model.main(l_img, r_img)
                    train_class_drbms(model.drbm, drbm_optimizer, label_emd)
                    drbm_out, binary_probs = model.drbm(label_emd)

                    if contrastive:
                        con_loss = con_lossfunc(label_emd, labels)
                        while con_weight * con_loss > 1:
                            con_weight = con_weight * 0.1
                        while con_weight * con_loss < 0.1:
                            con_weight = con_weight * 10
                        con_loss = con_weight * con_loss

                    bce_loss = focal_lossfunc(drbm_out, labels.to(torch.float32))
                    # bce_loss = bce_loss + 0.1 * bce_lossfunc(binary_probs, binary_labels.to(torch.float32))

                    if contrastive:
                        loss = con_loss + bce_loss
                    else:
                        loss = bce_loss

                if epoch_step in update_epoch:
                    # 清空梯度
                    model_optimizer.zero_grad()

                # 反向传播
                loss.backward()

                if epoch_step in update_epoch:
                    # 更新参数
                    model_optimizer.step()

                pbar.update(l_img.shape[0])
                global_step += 1
                epoch_loss += loss.item()

                experiment.log({
                    'train loss': loss.item(),
                    'epoch': epoch
                })
                if contrastive:
                    pbar.set_postfix(
                        **{
                            'loss (batch)': f"{loss.item():.4f}",
                            'con_loss': f"{con_loss.item():.4f}",
                            'bce_loss': f"{bce_loss.item():.4f}"
                        }
                    )
                else:
                    pbar.set_postfix(
                        **{
                            'loss (batch)': f"{loss.item():.4f}"
                        }
                    )

        if save_checkpoint:
            Path(dir_checkpoint).mkdir(parents=True, exist_ok=True)
            state_dict = model.state_dict()
            torch.save(state_dict, str(dir_checkpoint / 'checkpoint_epoch{}.pth'.format(epoch)))
            logging.info(f'Epoch {epoch} completed! Checkpoint {epoch} saved!\n')
        else:
            logging.info(f'Epoch {epoch} completed! Checkpoint {epoch} not saved, because save_checkpoint is False.\n')


def init_weights(m):
    """对 Conv2d 和 Linear 层进行 Kaiming 初始化"""
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        init.kaiming_uniform_(m.weight, nonlinearity='relu')  # 适用于 ReLU 激活
        if m.bias is not None:
            init.zeros_(m.bias)  # 偏置初始化为 0


if __name__ == '__main__':
    args = get_args()
    freeze_parameters = []
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')
    if args.save_checkpoint is False:
        warnings.warn('save_checkpoint is False, checkpoints will not be saved!')
    # Change here to adapt to your data
    # n_channels=3 for RGB images
    # n_classes is the number of probabilities you want to get per pixel

    # model = UNet(input_channels=3, output_classes=2, bilinear=args.bilinear)

    model = model.to(memory_format=torch.channels_last)

    logging.info(f'Network:\n'
                 f'\t{args.classes} output channels (classes)\n')

    if args.load != "":
        load_checkpoint(model, args.load)
        model.to(device=device)
    else:
        model.to(device=device)
        model.apply(init_weights)

    try:
        train_model(
            model=model,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            device=device,
            contrastive=args.con_loss,
            amp=args.amp,
            save_checkpoint=args.save_checkpoint
        )
    except torch.cuda.OutOfMemoryError:
        logging.error('Detected OutOfMemoryError! '
                      'Enabling checkpointing to reduce memory usage, but this slows down training. '
                      'Consider enabling AMP (--amp) for fast and memory efficient training')
        torch.cuda.empty_cache()
        raise torch.cuda.OutOfMemoryError
