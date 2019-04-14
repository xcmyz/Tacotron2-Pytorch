import torch
import torch.nn as nn
from torch import optim

from network import Tacotron2
from data_utils import DataLoader, collate_fn
from data_utils import Tacotron2DataLoader
from loss_function import Tacotron2Loss
import hparams as hp

from multiprocessing import cpu_count
import numpy as np
import argparse
import os
import time


cuda_available = torch.cuda.is_available()


def main(args):
    # Get device
    device = torch.device('cuda'if torch.cuda.is_available()else 'cpu')

    # Define model
    model = nn.DataParallel(Tacotron2(hp)).to(device)
    # model = Tacotron2(hp).to(device)
    print("Model Have Been Defined")

    # Get dataset
    dataset = Tacotron2DataLoader(hp.dataset_path)

    # Optimizer
    optimizer = torch.optim.Adam(
        model.parameters(), lr=hp.learning_rate, weight_decay=hp.weight_decay)

    # Criterion
    criterion = Tacotron2Loss()

    # Get training loader
    print("Get Training Loader")
    training_loader = DataLoader(dataset, batch_size=hp.batch_size, shuffle=True,
                                 collate_fn=collate_fn, drop_last=True, num_workers=cpu_count())

    # Load checkpoint if exists
    try:
        checkpoint = torch.load(os.path.join(
            hp.checkpoint_path, 'checkpoint_%d.pth.tar' % args.restore_step))
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        print("\n---Model Restored at Step %d---\n" % args.restore_step)

    except:
        print("\n---Start New Training---\n")
        if not os.path.exists(hp.checkpoint_path):
            os.mkdir(hp.checkpoint_path)

    # Define Some Information
    total_step = hp.epochs * len(training_loader)
    Time = np.array([])
    Start = time.perf_counter()

    # Training
    model = model.train()

    for epoch in range(hp.epochs):
        for i, batch in enumerate(training_loader):
            start_time = time.perf_counter()

            # Count step
            current_step = i + args.restore_step + \
                epoch * len(training_loader) + 1

            # Init
            optimizer.zero_grad()

            # Load Data
            text_padded, input_lengths, mel_padded, gate_padded, output_lengths = batch

            if cuda_available:
                text_padded = torch.from_numpy(text_padded).type(
                    torch.cuda.LongTensor).to(device)
            else:
                text_padded = torch.from_numpy(text_padded).type(
                    torch.LongTensor).to(device)
            mel_padded = torch.from_numpy(mel_padded).to(device)

            gate_padded = torch.from_numpy(gate_padded).to(device)

            input_lengths = torch.from_numpy(input_lengths).to(device)
            output_lengths = torch.from_numpy(output_lengths).to(device)

            # print("mel", mel_padded.size())
            # print("text", text_padded.size())
            # print("gate", gate_padded.size())

            batch = text_padded, input_lengths, mel_padded, gate_padded, output_lengths

            x, y = model.module.parse_batch(batch)
            y_pred = model(x)

            # Loss
            loss, mel_loss, gate_loss = criterion(y_pred, y)

            # Backward
            loss.backward()

            # Clipping gradients to avoid gradient explosion
            nn.utils.clip_grad_norm_(model.parameters(), hp.grad_clip_thresh)

            # Update weights
            optimizer.step()

            if current_step % hp.log_step == 0:
                Now = time.perf_counter()
                str_loss = "Epoch [{}/{}], Step [{}/{}], Mel Loss: {:.4f}, Gate Loss: {:.4f}, Total Loss: {:.4f}.".format(
                    epoch + 1, hp.epochs, current_step, total_step, mel_loss.item(), gate_loss.item(), loss.item())
                str_time = "Time Used: {:.3f}s, Estimated Time Remaining: {:.3f}s.".format(
                    (Now - Start), (total_step - current_step) * np.mean(Time))

                print(str_loss)
                print(str_time)
                with open("logger.txt", "a")as f_logger:
                    f_logger.write(str_loss + "\n")
                    f_logger.write(str_time + "\n")
                    f_logger.write("\n")

            if current_step % hp.save_step == 0:
                torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(
                )}, os.path.join(hp.checkpoint_path, 'checkpoint_%d.pth.tar' % current_step))
                print("\nsave model at step %d ...\n" % current_step)

            if current_step in hp.decay_step:
                optimizer = adjust_learning_rate(optimizer, current_step)

            end_time = time.perf_counter()
            Time = np.append(Time, end_time - start_time)
            if len(Time) == hp.clear_Time:
                temp_value = np.mean(Time)
                Time = np.delete(
                    Time, [i for i in range(len(Time))], axis=None)
                Time = np.append(Time, temp_value)


def adjust_learning_rate(optimizer, step):
    if step == 100000:
        for param_group in optimizer.param_groups:
            param_group['lr'] = 0.0005

    elif step == 200000:
        for param_group in optimizer.param_groups:
            param_group['lr'] = 0.0003

    elif step == 300000:
        for param_group in optimizer.param_groups:
            param_group['lr'] = 0.0001

    return optimizer


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--restore_step', type=int,
                        help='Global step to restore checkpoint', default=170100)

    args = parser.parse_args()
    main(args)
