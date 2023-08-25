import time
import copy

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, ConcatDataset

import clip

import converter_dassl, converter_domainbed
from utils import accuracy, AverageMeter, ProgressMeter, TensorboardWriter, ForeverDataIterator


class GeneralMovingAverage(object):
    def __init__(self, model, weight_func):
        self.model = model
        self.weight_func = weight_func
        self.iter = 0
        self.weight = weight_func(self.iter)
        self.weight_sum = self.weight
        self.moving_avg = copy.deepcopy(model)
        for param in self.moving_avg.parameters():
            param.requires_grad = False

    def update(self):
        self.iter += 1
        self.weight = self.weight_func(self.iter)
        relative_weight = self.weight / self.weight_sum
        for moving_avg_param, param in zip(self.moving_avg.parameters(), self.model.parameters()):
            moving_avg_param.data = (moving_avg_param + relative_weight * param) / (1 + relative_weight)
        self.weight_sum += self.weight

    def __call__(self, x: torch.Tensor):
        return self.moving_avg(x)

    def train(self, mode=True):
        self.moving_avg.train(mode)

    def eval(self):
        self.train(False)

    def state_dict(self):
        return self.moving_avg.state_dict()

    def load_state_dict(self, state_dict):
        self.moving_avg.load_state_dict(state_dict)

    @property
    def module(self):
        return self.moving_avg.module


def get_dataset(args):
    if args.task == "domain_shift":
        # load domainbed data
        train_datasets, val_datasets, test_datasets, class_names = \
            converter_domainbed.get_domainbed_datasets(dataset_name=args.data, root=args.root, targets=args.targets, holdout=0.2)
        train_class_names = class_names
        train_iter = converter_domainbed.get_forever_iter(train_datasets, args.batch_size, num_workers=args.workers)
        val_loader = DataLoader(ConcatDataset(val_datasets), batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
        test_loaders = [
            {
                "name": "test",
                "loader": DataLoader(ConcatDataset(test_datasets), batch_size=args.batch_size, shuffle=False, num_workers=args.workers),
                "class_names": class_names
            }
        ]
        template = "a photo of a {}."
    
    elif args.task == "open_class":
        # load dassl data
        train_dataset, val_dataset, test_dataset, open_dataset, base_class_names, open_class_names, template = \
            converter_dassl.get_dassl_datasets(dataset_name=args.data, root=args.root, n_shot=args.n_shot)
        train_class_names = base_class_names
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, drop_last=True)
        train_iter = ForeverDataIterator(train_loader)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
        test_loaders = [
            {
                "name": "test",
                "loader": DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers),
                "class_names": base_class_names
            },
            {
                "name": "open",
                "loader": DataLoader(open_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers),
                "class_names": open_class_names
            }
        ]

    elif args.task == "in_the_wild":
        # load domainbed data
        train_datasets, val_datasets, test_datasets, open_datasets, base_class_names, open_class_names = \
            converter_domainbed.get_domainbed_datasets(dataset_name=args.data, root=args.root, targets=args.targets, holdout=0.2, seed=args.seed, open_ratio=0.5)
        train_class_names = base_class_names
        train_iter = converter_domainbed.get_forever_iter(train_datasets, args.batch_size, num_workers=args.workers)
        val_loader = DataLoader(ConcatDataset(val_datasets), batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
        test_loaders = [
            {
                "name": "test",
                "loader": DataLoader(ConcatDataset(test_datasets), batch_size=args.batch_size, shuffle=False, num_workers=args.workers),
                "class_names": base_class_names + open_class_names
            },
            {
                "name": "open",
                "loader": DataLoader(ConcatDataset(open_datasets), batch_size=args.batch_size, shuffle=False, num_workers=args.workers),
                "class_names": base_class_names + open_class_names
            }
        ]
        template = "a photo of a {}."
    
    return train_iter, val_loader, test_loaders, train_class_names, template


def get_text_features(clip_model, template, class_names, device):
    with torch.no_grad():
        texts = torch.cat(
            [clip.tokenize(template.format(c.replace("_", " ")))
            for c in class_names]).to(device)
        text_features = clip_model.encode_text(texts)
        text_features /= text_features.norm(dim=-1, keepdim=True)


def convert_models_to_fp32(model):
    for p in model.parameters():
        p.data = p.data.float()
        p.grad.data = p.grad.data.float()


def train(train_iter: ForeverDataIterator, model, moving_avg_model: GeneralMovingAverage, text_features: torch.Tensor,
          optimizer, lr_scheduler, epoch: int, args, writer: TensorboardWriter, device):
    batch_time = AverageMeter('Time', ':4.2f')
    data_time = AverageMeter('Data', ':3.1f')
    losses = AverageMeter('Loss', ':3.2f')
    cls_accs = AverageMeter('Cls Acc', ':3.1f')

    progress = ProgressMeter(
        args.iters_per_epoch,
        [batch_time, data_time, losses, cls_accs],
        prefix="Epoch: [{}]".format(epoch))

    # Freeze all Norm Layers
    model.eval()

    end = time.time()
    for i in range(args.iters_per_epoch):
        # obtain data
        if args.tasks in ["domain_shift", "in_the_wild"]:
            for x_d, labels_d in next(train_iter):
                x.append(x_d)
                labels.append(labels_d)
            x, labels = torch.cat(x), torch.cat(labels)
        else:
            x, labels = next(train_iter)
        x, labels = x.to(device), labels.to(device)

        # measure data loading time
        data_time_step = time.time() - end
        data_time.update(data_time_step)

        # compute output
        f = model(x)
        f = f / f.norm(dim=-1, keepdim=True)
        f -= args.lambda * text_features[labels]
        y = f @ text_features.T
        y = args.temperature * y

        loss = F.cross_entropy(y, labels)

        cls_acc = accuracy(y, labels)[0]
        losses.update(loss.item(), x.size(0))
        cls_accs.update(cls_acc.item(), x.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        convert_models_to_fp32(model)
        optimizer.step()
        clip.model.convert_weights(model)
        lr_scheduler.step()

        moving_avg_model.update()

        bma_weight = moving_avg_model.weight

        # measure elapsed time
        batch_time_step = time.time() - end
        batch_time.update(batch_time_step)

        writer.record_training_values(
            {
                "Loss": (loss.item(), x.shape[0]),
                "Acc@1": (cls_acc.item(), x.shape[0]),
                "Time": (batch_time_step,),
                "Data": (data_time_step,),
                "Weight": (bma_weight,),
            }
        )

        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)


def validate(val_loader, model, text_features, args, device, shift=0) -> float:
    batch_time = AverageMeter('Time', ':6.3f')
    top1 = AverageMeter('Acc@1', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, top1],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            images = images.to(device)
            target = target.to(device) - shift

            # compute output
            image_features = model(images)
            image_features /= image_features.norm(dim=-1, keepdim=True)

            # measure accuracy and record loss
            output_similarity = image_features @ text_features.T
            acc1, = accuracy(output_similarity, target, topk=(1,))

            top1.update(acc1.item(), images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)

        print(' * Acc@1 {top1.avg:.3f}'.format(top1=top1))
    
    return top1.avg


def evaluate_all(model, val_loader, train_text_features, test_loaders, args, writer, device):
    print("Evaluate on validation set...")
    val_acc1 = validate(val_loader, model, train_text_features, args, device)
    writer.write_eval_values({"Acc@1": val_acc1}, prefix="val")

    for test_loader in test_loaders:
        split_name = test_loader["name"]
        print(f"Evaluate on {split_name} set...")
        validate(test_loader["loader"], model, test_loader["text_features"], args, device)
        writer.write_eval_values({"Acc@1": val_acc1}, prefix=split_name)

    return val_acc1
