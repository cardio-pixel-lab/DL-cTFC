#!/usr/bin/env python3
"""
测试脚本 - 自动检测标签 + 完整分类指标
支持有标签评估和无标签推理
计算 SOTA 常用指标：Accuracy@1/5, Precision, Recall, F1-Score, Confusion Matrix, AUC-ROC
"""

import argparse
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import json
import os
from pathlib import Path
from collections import Counter
from timm.models import create_model
import MedViT
import csv

def parse_class_names(raw: str):
    if not raw:
        return None
    names = [x.strip() for x in raw.split(",") if x.strip()]
    return names if names else None

def get_args_parser():
    parser = argparse.ArgumentParser('MedViT testing script', add_help=False)
    
    # Model parameters
    parser.add_argument('--model', default='MedViT_small', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--input-size', default=224, type=int, help='images input size')

    # Dataset parameters
    parser.add_argument('--data-path', default='/data/gzist/clr/share/data/cag_cls', type=str,
                        help='dataset path')
    parser.add_argument('--test-dir', default='test', type=str,
                        help='test directory name (default: test)')
    parser.add_argument('--nb_classes', default=2, type=int, help='number of classes')
    parser.add_argument('--has-labels', action='store_true', default=None,
                        help='whether test set has labels (auto-detect if not specified)')
    parser.add_argument('--no-labels', action='store_false', dest='has_labels',
                        help='force test set as unlabeled')
    
    # Test parameters
    parser.add_argument('--checkpoint', default='', type=str, required=True,
                        help='path to checkpoint file')
    parser.add_argument('--output-dir', default='./test_results', type=str,
                        help='path where to save test results')
    parser.add_argument('--batch-size', default=64, type=int, help='batch size for testing')
    parser.add_argument('--num_workers', default=4, type=int, help='number of data loading workers')
    
    # Device
    parser.add_argument('--device', default='cuda', help='device to use for testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--pin-mem', action='store_true', default=True)
    
    # Output options
    parser.add_argument('--save-pred', action='store_true', help='save predictions to file')
    parser.add_argument('--save-confusion', action='store_true', help='save confusion matrix')
    # parser.add_argument('--verbose', action='store_true', help='print detailed results')
    
    parser.add_argument(
        '--class-names',
        default='lca,rca',
        type=str,
        help='comma-separated class names, e.g. "lca,rca"'
    )
    
    return parser


class TestDataset(torch.utils.data.Dataset):
    """
    测试数据集 - 支持有标签和无标签模式
    """
    def __init__(self, root, transform=None, has_labels=None):
        self.root = root
        self.transform = transform
        self.samples = []
        self.classes = []
        self.class_to_idx = {}
        self.has_labels = False
        
        # 检查是否有子目录（有标签）
        subdirs = [d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))]
        
        if has_labels is True or (has_labels is None and subdirs):
            # 有标签的测试集（按类别子目录组织）
            self.classes = sorted(subdirs)
            self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
            self.has_labels = True
            
            for class_name in self.classes:
                class_dir = os.path.join(root, class_name)
                for fname in sorted(os.listdir(class_dir)):
                    if fname.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.JPEG', '.PNG')):
                        self.samples.append((os.path.join(class_dir, fname), self.class_to_idx[class_name]))
        else:
            # 无标签的测试集（所有图片在同一目录）
            for fname in sorted(os.listdir(root)):
                if fname.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.JPEG', '.PNG')):
                    self.samples.append((os.path.join(root, fname), -1))
        
        print(f"Loaded {len(self.samples)} samples from {root}")
        print(f"Has labels: {self.has_labels}")
        if self.classes:
            print(f"Classes: {self.classes} (idx: {self.class_to_idx})")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, index):
        from PIL import Image
        path, target = self.samples[index]
        image = Image.open(path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, target, path


def build_test_dataset(args):
    """构建测试数据集"""
    from torchvision import transforms
    from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
    
    # 构建测试变换
    t = []
    t.append(transforms.Resize(int((256 / 224) * args.input_size), interpolation=3))
    t.append(transforms.CenterCrop(args.input_size))
    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD))
    transform = transforms.Compose(t)
    
    # 测试集路径
    test_root = os.path.join(args.data_path, args.test_dir)
    
    if not os.path.exists(test_root):
        raise FileNotFoundError(f"Test directory not found: {test_root}")
    
    dataset = TestDataset(test_root, transform=transform, has_labels=args.has_labels)
    args.nb_classes = len(dataset.classes) if dataset.has_labels else args.nb_classes
    return dataset


def calculate_metrics(predictions, targets, num_classes, probabilities=None):
    """
    计算完整的分类指标
    """
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score,
        confusion_matrix, roc_auc_score
    )
    predictions = np.array(predictions)
    targets = np.array(targets)

    metrics = {}
    metrics['accuracy'] = float(accuracy_score(targets, predictions))
    metrics['acc1'] = metrics['accuracy'] * 100

    precision_per_class = precision_score(targets, predictions, average=None, labels=range(num_classes), zero_division=0)
    recall_per_class = recall_score(targets, predictions, average=None, labels=range(num_classes), zero_division=0)
    f1_per_class = f1_score(targets, predictions, average=None, labels=range(num_classes), zero_division=0)

    metrics['precision_macro'] = float(precision_score(targets, predictions, average='macro', labels=range(num_classes), zero_division=0)) * 100
    metrics['precision_weighted'] = float(precision_score(targets, predictions, average='weighted', labels=range(num_classes), zero_division=0)) * 100
    metrics['recall_macro'] = float(recall_score(targets, predictions, average='macro', labels=range(num_classes), zero_division=0)) * 100
    metrics['recall_weighted'] = float(recall_score(targets, predictions, average='weighted', labels=range(num_classes), zero_division=0)) * 100
    metrics['f1_macro'] = float(f1_score(targets, predictions, average='macro', labels=range(num_classes), zero_division=0)) * 100
    metrics['f1_weighted'] = float(f1_score(targets, predictions, average='weighted', labels=range(num_classes), zero_division=0)) * 100

    metrics['per_class'] = {
        i: {
            'precision': float(precision_per_class[i]) * 100,
            'recall': float(recall_per_class[i]) * 100,
            'f1': float(f1_per_class[i]) * 100
        } for i in range(num_classes)
    }

    cm = confusion_matrix(targets, predictions, labels=range(num_classes))
    metrics['confusion_matrix'] = cm.tolist()

    try:
        if probabilities is not None:
            probabilities = np.array(probabilities)
            if num_classes == 2:
                metrics['auc_roc'] = float(roc_auc_score(targets, probabilities[:, 1])) * 100
            else:
                metrics['auc_roc'] = float(roc_auc_score(targets, probabilities, average='macro', multi_class='ovr')) * 100
        else:
            metrics['auc_roc'] = None
    except Exception:
        metrics['auc_roc'] = None

    return metrics


def main(args):
    print("="*60)
    print("MedViT Testing Script - Full Classification Metrics")
    print("="*60)
    print(args)

    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    cudnn.benchmark = True

    # 构建测试数据集
    print("\n" + "-"*60)
    print("Building test dataset...")
    dataset_test = build_test_dataset(args)
    print(f"Test samples: {len(dataset_test)}")
    print(f"Number of classes: {args.nb_classes}")
    print(f"Has labels: {dataset_test.has_labels}")

    # 1) 统一类别名（优先 --class-names）
    class_names = dataset_test.classes if getattr(dataset_test, "classes", None) else [str(i) for i in range(args.nb_classes)]
    user_names = parse_class_names(args.class_names)
    if user_names is not None:
        if len(user_names) != args.nb_classes:
            raise ValueError(f'--class-names数量({len(user_names)}) != nb_classes({args.nb_classes})')
        class_names = user_names

    print(f"Class names: {class_names}")

    # 数据加载器
    data_loader_test = torch.utils.data.DataLoader(
        dataset_test,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False,
        shuffle=False
    )

    # 创建模型（参考 main.py，只传递 num_classes）
    print("\n" + "-"*60)
    print(f"Creating model: {args.model}")
    model = create_model(
        args.model,
        num_classes=args.nb_classes,
    )

    # 加载检查点
    print(f"Loading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    
    if 'model' in checkpoint:
        checkpoint_model = checkpoint['model']
    elif 'state_dict' in checkpoint:
        checkpoint_model = checkpoint['state_dict']
    else:
        checkpoint_model = checkpoint
    
    # 移除可能不匹配的分类层
    state_dict = model.state_dict()
    for k in ['head.weight', 'head.bias', 'proj_head.0.weight', 'proj_head.0.bias']:
        if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
            print(f"Removing key {k} from pretrained checkpoint")
            del checkpoint_model[k]
    
    model.load_state_dict(checkpoint_model, strict=False)
    print("Checkpoint loaded successfully")

    model.to(device)
    model.eval()

    # 合并 BN 加速推理（参考 main.py，安全调用）
    try:
        if hasattr(model, "merge_bn"):
            print("Merging batch norm layers to speedup inference...")
            model.merge_bn()
    except Exception as e:
        print(f"Warning: merge_bn failed ({e}), skipping...")

    # 推理
    print("\n" + "-"*60)
    print("Running inference...")
    
    all_preds, all_paths = [], []
    all_targets = [] if dataset_test.has_labels else None
    need_probs = dataset_test.has_labels or args.save_pred
    all_probs = [] if need_probs else None

    with torch.no_grad():
        for images, targets, paths in data_loader_test:
            images = images.to(device, non_blocking=True)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            preds = outputs.argmax(dim=1)

            all_preds.append(preds.cpu())
            if dataset_test.has_labels:
                all_targets.append(targets.cpu())
            if need_probs:
                all_probs.append(probs.cpu())
            all_paths.extend(paths)

    all_preds = torch.cat(all_preds, dim=0)
    pred_ids = all_preds.tolist()
    pred_names = [class_names[i] for i in pred_ids]

    if dataset_test.has_labels:
        all_targets = torch.cat(all_targets, dim=0)
        target_ids = all_targets.tolist()
        target_names = [class_names[i] for i in target_ids]
    else:
        all_targets = None

    if need_probs:
        all_probs = torch.cat(all_probs, dim=0)
    else:
        all_probs = None

    # 计算指标
    results = {
        'checkpoint': args.checkpoint,
        'num_samples': len(dataset_test),
        'num_classes': args.nb_classes,
        'classes': dataset_test.classes,
        'has_labels': dataset_test.has_labels,
    }
    
    metrics = None
    if dataset_test.has_labels:
        print("\n" + "="*60)
        print("Classification Metrics")
        print("="*60)
        
        metrics = calculate_metrics(
            pred_ids,
            target_ids,
            args.nb_classes,
            probabilities=all_probs.numpy() if all_probs is not None else None
        )
        metrics["per_class_named"] = {
            class_names[i]: metrics["per_class"][i] for i in range(args.nb_classes)
        }
        results["metrics"] = metrics

        # 打印指标
        print(f"\n📊 Accuracy@1:        {metrics['acc1']:.2f}%")
        if metrics['auc_roc']:
            print(f"📊 AUC-ROC:           {metrics['auc_roc']:.2f}%")
        print(f"📊 Precision (macro):  {metrics['precision_macro']:.2f}%")
        print(f"📊 Recall (macro):     {metrics['recall_macro']:.2f}%")
        print(f"📊 F1-Score (macro):   {metrics['f1_macro']:.2f}%")
        print(f"📊 Precision (weighted): {metrics['precision_weighted']:.2f}%")
        print(f"📊 Recall (weighted):    {metrics['recall_weighted']:.2f}%")
        print(f"📊 F1-Score (weighted):  {metrics['f1_weighted']:.2f}%")
        
        # Per-class metrics
        print(f"\n📋 Per-Class Metrics:")
        print("-"*60)
        print(f"{'Class':<15} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}")
        print("-"*60)
        for cls_idx, cls_metrics in metrics['per_class'].items():
            cls_name = dataset_test.classes[cls_idx] if cls_idx < len(dataset_test.classes) else f"Class-{cls_idx}"
            print(f"{cls_name:<15} {cls_metrics['precision']:>8.2f}%   {cls_metrics['recall']:>8.2f}%   {cls_metrics['f1']:>8.2f}%")
        print("-"*60)
        
        # Confusion Matrix
        print(f"\n📋 Confusion Matrix:")
        cm = np.array(metrics['confusion_matrix'])
        print(cm)
        
        results.update(metrics)
        print("="*60)
    else:
        print("\n" + "="*60)
        print("Inference Only (No Labels)")
        print("="*60)
        print(f"Total samples: {len(all_preds)}")
        print("="*60)

    # 保存结果
    if args.output_dir:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存 JSON 结果
        result_file = output_dir / 'test_results.json'
        results_save = {k: v for k, v in results.items() if k != 'confusion_matrix'}
        with open(result_file, 'w') as f:
            json.dump(results_save, f, indent=2)
        print(f"\n💾 Results saved to: {result_file}")
        
        # 保存混淆矩阵
        if args.save_confusion and dataset_test.has_labels:
            cm_file = output_dir / 'confusion_matrix.json'
            with open(cm_file, 'w') as f:
                json.dump({
                    'classes': dataset_test.classes,
                    'matrix': results['confusion_matrix']
                }, f, indent=2)
            print(f"💾 Confusion matrix saved to: {cm_file}")
        
        # 保存预测结果
        if args.save_pred:
            pred_file = output_dir / 'predictions.csv'
            with open(pred_file, 'w', newline='', encoding='utf-8-sig') as f:
                writer = csv.writer(f)
                prob_headers = [f"prob_{name}" for name in class_names]
                header = ["path", "pred", "pred_name", "prob"] + prob_headers
                if dataset_test.has_labels:
                    header.extend(["target", "target_name", "correct"])
                writer.writerow(header)

                for i, path in enumerate(all_paths):
                    pred = int(all_preds[i].item())
                    pred_name = class_names[pred]
                    prob = float(all_probs[i, pred].item()) if all_probs is not None else ""
                    class_probs = []
                    if all_probs is not None:
                        class_probs = [float(all_probs[i, j].item()) for j in range(args.nb_classes)]
                    else:
                        class_probs = [""] * args.nb_classes

                    row = [path, pred, pred_name, prob] + class_probs

                    if dataset_test.has_labels:
                        target = int(all_targets[i].item())
                        target_name = class_names[target]
                        correct = int(pred == target)
                        row.extend([target, target_name, correct])

                    writer.writerow(row)
            print(f"💾 Predictions saved to: {pred_file}")
        
        # 保存详细报告
        report_file = output_dir / 'test_report.txt'
        with open(report_file, 'w') as f:
            f.write("="*60 + "\n")
            f.write("MedViT Test Report\n")
            f.write("="*60 + "\n\n")
            f.write(f"Checkpoint: {args.checkpoint}\n")
            f.write(f"Test samples: {len(dataset_test)}\n")
            f.write(f"Classes: {dataset_test.classes}\n\n")
            
            if dataset_test.has_labels and metrics:
                f.write("Classification Metrics:\n")
                f.write("-"*60 + "\n")
                f.write(f"Accuracy@1:        {metrics['acc1']:.2f}%\n")
                if metrics['auc_roc']:
                    f.write(f"AUC-ROC:           {metrics['auc_roc']:.2f}%\n")
                f.write(f"Precision (macro):  {metrics['precision_macro']:.2f}%\n")
                f.write(f"Recall (macro):     {metrics['recall_macro']:.2f}%\n")
                f.write(f"F1-Score (macro):   {metrics['f1_macro']:.2f}%\n")
                f.write(f"Precision (weighted): {metrics['precision_weighted']:.2f}%\n")
                f.write(f"Recall (weighted):    {metrics['recall_weighted']:.2f}%\n")
                f.write(f"F1-Score (weighted):  {metrics['f1_weighted']:.2f}%\n")
        print(f"💾 Report saved to: {report_file}")

    return all_preds if dataset_test.has_labels else None


if __name__ == '__main__':
    parser = argparse.ArgumentParser('MedViT testing script', parents=[get_args_parser()])
    args = parser.parse_args()
    
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    main(args)


    """
    python test.py --checkpoint ../medvit_cls_exp/0311_162826/checkpoint_best.pth --data-path /home/share/clr/share/data/cag_cls --test-dir test --output-dir ./test_results --save-pred --save-confusion --has-labels
    # 1. 有标签测试集（自动检测）
    python test.py \
        --checkpoint ../medvit_cls_exp/0311_162826/checkpoint_best.pth \
        --data-path /home/share/clr/share/data/cag_cls \
        --test-dir test \
        --output-dir ./test_results \
        --save_pred --save_confusion

    # 2. 有标签测试集（显式指定）
    python test.py \
        --checkpoint ../medvit_cls_exp/0311_162826/checkpoint_best.pth \
        --data-path /home/share/clr/share/data/cag_cls \
        --test-dir test \
        --has-labels \
        --output-dir ./test_results

    # 3. 无标签测试集（仅推理）
    python test.py \
        --checkpoint ../medvit_cls_exp/0311_162826/checkpoint_best.pth \
        --data-path /home/share/clr/share/data/cag_cls \
        --test-dir test_unlabeled \
        --output-dir ./test_results \
        --save_pred
    """