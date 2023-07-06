import random
import warnings
import argparse
import shutil
import scipy as sp

import torch
import torch.backends.cudnn as cudnn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

import clip

from utils import CompleteLogger, TensorboardWriter
from engine import GeneralMovingAverage, get_dataset, get_text_features, train, evaluate_all


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main(args: argparse.Namespace):
    logger = CompleteLogger(args.log, args.phase)
    print(args)

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    cudnn.benchmark = True
    
    clip_model, _ = clip.load(args.arch, device)
    
    train_iter, val_loader, test_loaders, train_class_names, template = get_dataset(args)

    # create model
    print("=> using pre-trained model '{}'".format(args.arch))
    classifier = clip_model.visual
    classifier = classifier.to(device)
    clip.model.convert_weights(classifier)
    classifier.eval()
    
    # obtain text features
    train_text_features = get_text_features(clip_model, template, train_class_names, device)
    for test_loader in test_loaders:
        test_loader["text_features"] = get_text_features(clip_model, template, test_loader["class_names"], device)

    # define beta moving average
    beta_dist = sp.stats.beta(args.beta, args.beta)
    total_iter = args.epochs * args.iters_per_epoch
    weight_func = lambda it: beta_dist.pdf((it + 0.5) / (total_iter + 1))

    bma_classifier = GeneralMovingAverage(classifier, weight_func)

    # define optimizer and lr scheduler
    optimizer = AdamW(classifier.parameters(), lr=args.lr, weight_decay=args.wd)
    lr_scheduler = CosineAnnealingLR(optimizer, args.epochs * args.iters_per_epoch)
    
    # define temperature for training
    if args.temperature is None:
        args.temperature = clip_model.logit_scale.exp().item()

    # define tensorboard writer
    writer = TensorboardWriter(args.log, flush_freq=20)

    # evaluate zero-shot performance
    best_val_acc1 = evaluate_all(classifier, val_loader, train_text_features, test_loaders, args, writer, device)

    # start training
    for epoch in range(args.epochs):
        print(f"Learning rate: {lr_scheduler.get_lr()}")
        
        # train for one epoch
        train(train_iter, classifier, bma_classifier, train_text_features, optimizer, lr_scheduler, epoch, args, writer, device)

        # evaluate all
        val_acc1 = evaluate_all(classifier, val_loader, train_text_features, test_loaders, args, writer, device)

        # remember best acc@1 and save checkpoint
        torch.save(classifier.state_dict(), logger.get_checkpoint_path('latest'))
        if val_acc1 > best_val_acc1:
            shutil.copy(logger.get_checkpoint_path('latest'), logger.get_checkpoint_path('best'))
            best_val_acc1 = val_acc1

    print("Training completed.")

    classifier.load_state_dict(torch.load(logger.get_checkpoint_path('best')))
    print("Evaluate best model:")
    evaluate_all(classifier, val_loader, train_text_features, test_loaders, args, writer, device)

    torch.save(bma_classifier.state_dict(), logger.get_checkpoint_path('bma'))
    print("Evaluate BMA model:")
    evaluate_all(bma_classifier, val_loader, train_text_features, test_loaders, args, writer, device)
    
    logger.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Baseline for Domain Generalization')
    # dataset parameters
    parser.add_argument('root', metavar='DIR',
                        help='root path of dataset')
    parser.add_argument('-d', '--data', metavar='DATA', default='DomainNet')
    parser.add_argument('--task', default='domain_shift', choices=
                        ['domain_shift', 'open_class', 'in_the_wild'])
    parser.add_argument('--n-shot', type=int, default=0)
    # model parameters
    parser.add_argument('-a', '--arch', metavar='ARCH', default='ViT-B/16')
    # training parameters
    parser.add_argument('-b', '--batch-size', default=36, type=int,
                        metavar='N',
                        help='mini-batch size (default: 36)')
    parser.add_argument('--lr', '--learning-rate', default=5e-6, type=float,
                        metavar='LR', help='initial learning rate', dest='lr')
    parser.add_argument('--wd', '--weight-decay', default=0.1, type=float,
                        metavar='W', help='weight decay (default: 0.1)')
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--epochs', default=20, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('-i', '--iters-per-epoch', default=500, type=int,
                        help='Number of iterations per epoch')
    parser.add_argument('-p', '--print-freq', default=100, type=int,
                        metavar='N', help='print frequency (default: 100)')
    parser.add_argument('--seed', default=0, type=int,
                        help='seed for initializing training. ')
    parser.add_argument("--log", type=str, default='exp0',
                        help="Where to save logs, checkpoints and debugging images.")
    # parameters for CLIPood
    parser.add_argument('--temperature', type=float, default=None, help=
                        "Use CLIP's original temperature in default.")
    parser.add_argument('--lambda', type=float, default=0.3)
    parser.add_argument('--beta', type=float, default=0.5)

    args = parser.parse_args()
    main(args)
