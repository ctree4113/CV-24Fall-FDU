import os
import argparse
import torch
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable
from PIL import Image
from torchvision import transforms
from data_loader.GetDataset_ISIC2018 import ISIC2018_dataset
from data_loader.GetDataset_Retouch import MyDataset
from data_loader.GetDataset_CHASE import MyDataset_CHASE
from model.DconnNet import DconnNet
from connect_loss import Bilateral_voting
from metrics.cldice import clDice
from metrics.cal_betti import getBetti
from skimage import measure
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import pandas as pd


device = 'cuda' if torch.cuda.is_available() else 'cpu'
to_pil = transforms.ToPILImage()


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate trained DconnNet and visualize some results.')

    # dataset info
    parser.add_argument('--dataset', type=str, default='chase',  
                        help='isic, chase')
    parser.add_argument('--data_root', type=str, default='/retouch',  
                        help='dataset directory')
    parser.add_argument('--resize', type=int, default=[256, 256], nargs='+',
                        help='image size: [height, width]')

    # metrics need to use
    parser.add_argument('--metrics', type=str, default='DSC:IOU',  
                        help='optional metrics: DSC, IOU, ACC, PREC, clDice, 0-Betti, 1-Betti')

    # model configuration
    parser.add_argument('--num-class', type=int, default=4, metavar='N',
                        help='number of classes for your data')
    parser.add_argument('--decoder_attention', action='store_true', default=False,
                        help='use attention mechnism in LWDecoder')

    # model weights path
    # the path should contain #folds of directories
    # example:
    # -- /home/xxx/ckpt/
    # ---- /baseline/
    # ------ /CHASEDB1/
    # -------- /1/
    # ---------- best_model.pth
    # ...
    # -------- /5/
    # ---------- best_model.pth
    parser.add_argument('--ckpt_path', type=str, default=None,
                        help='put the path to checkpoints')

    # output setting
    parser.add_argument('--output_path', type=str, default=None,
                        help='path to save output')
    parser.add_argument('--num_pred', type=int, default=-1, metavar='N', 
                        help='number of visualized predictions for each fold, if <=0, visualize all')
    
    # Add new metrics
    parser.add_argument('--eval_freq', action='store_true',
                      help='Evaluate frequency domain metrics')
    parser.add_argument('--eval_topo', action='store_true',
                      help='Evaluate topology preservation metrics')
    
    args = parser.parse_args()

    if not os.path.isdir(args.output_path):
        os.makedirs(args.output_path)

    args.num_pred = max(0, args.num_pred)

    return args


def get_metric(pred, gt, metric_type='DSC'):
    eps = 0.0001
    ret = 0
    FN = torch.sum((1 - pred) * gt, dim=(2,3)) 
    FP = torch.sum((1 - gt) * pred, dim=(2,3))
    inter = torch.sum(gt * pred, dim=(2,3)) # TP
    TN = torch.sum((1 - gt) * (1 - pred), dim=(2,3))
    union = torch.sum(gt, dim=(2,3)) + torch.sum(pred, dim=(2,3))
    if metric_type == 'DSC':
        ret = (2 * inter + eps) / (union + eps)
    elif metric_type == 'IOU':
        ret = (inter + eps) / (inter + FP + FN + eps)
    elif metric_type == 'clDice':
        cldc = []
        for i in range(pred.shape[0]):
            cldc.append(clDice(pred[i].cpu().numpy(), gt[i].cpu().numpy()))
        ret = np.mean(cldc)
    elif metric_type == '0-Betti':
        lst = []
        for i in range(pred.shape[0]):
            assert pred[i].shape[0] == 1
            lst.append(getBetti(pred[i][0], gt[i][0], i=0))
        ret = np.mean(lst)
    elif metric_type == '1-Betti':
        lst = []
        for i in range(pred.shape[0]):
            assert pred[i].shape[0] == 1
            lst.append(getBetti(pred[i][0], gt[i][0], i=1))
        ret = np.mean(lst)
    elif metric_type == 'ACC':
        ret = (inter + TN) / (inter + TN + FP + FN + eps)
    elif metric_type == 'PREC':
        ret = inter / (inter + FP + eps)
    else:
        raise ValueError(f'metric {metric_type} not supported')
    return ret.item()


def get_mask(pred):
    """Convert multi-class prediction to segmentation mask"""
    b, c, h, w = pred.shape
    mask = torch.zeros((b, h, w)).cuda()
    
    # Get class with maximum probability for each pixel
    for i in range(c):
        mask = torch.where(pred[:, i] > mask, 
                          torch.full_like(mask, i),
                          mask)
    return mask

def one_hot(mask, shape):
    """Convert segmentation mask to one-hot encoding"""
    b, c, h, w = shape
    one_hot = torch.zeros((b, c, h, w)).cuda()
    
    # Set corresponding channel to 1 for each class
    for i in range(c):
        one_hot[:, i] = (mask == i).float()
    return one_hot

def predict(model, img, hori, verti, num_class=1):
    """Make prediction with bilateral voting"""
    N, C, H, W = img.shape
    out, _ = model(img)
    if num_class == 1:  
        out = F.sigmoid(out)
        class_pred = out.view([N, -1, 8, H, W]) #(N, C, 8, H, W)
        pred = torch.where(class_pred > 0.5, 1, 0)
        pred, _ = Bilateral_voting(pred.float(), hori, verti) # (N, 1, H, W)
    else:
        class_pred = out.view([N, -1, 8, H, W]) #(N, C, 8, H, W)
        final_pred, _ = Bilateral_voting(class_pred, hori, verti)
        pred = get_mask(final_pred)
        pred = one_hot(pred, img.shape)
    return pred


def evaluate_frequency(pred, target):
    """Evaluate frequency domain metrics"""
    pred_freq = torch.fft.rfft2(pred)
    target_freq = torch.fft.rfft2(target)
    
    # Compute frequency spectrum similarity
    freq_sim = F.cosine_similarity(
        pred_freq.abs().view(pred.size(0), -1),
        target_freq.abs().view(target.size(0), -1),
        dim=1
    ).mean()
    
    return {'freq_sim': freq_sim.item()}

def evaluate_topology(pred, target):
    """Evaluate topology preservation metrics"""
    # Compute connected components
    pred_cc = measure.label(pred.cpu().numpy() > 0.5)
    target_cc = measure.label(target.cpu().numpy() > 0.5)
    
    # Compare component numbers
    topo_diff = abs(pred_cc.max() - target_cc.max()) / max(pred_cc.max(), target_cc.max())
    
    return {'topo_diff': topo_diff}

def generate_summary_plot(args, metrics_df):
    """Generate a summary plot with all metrics and visualizations"""
    plt.figure(figsize=(24, 20))
    gs = GridSpec(3, 2, height_ratios=[1, 1, 3], hspace=0.4, wspace=0.3)
    
    # Create metrics table
    ax0 = plt.subplot(gs[0, 0])
    ax0.axis('tight')
    ax0.axis('off')
    
    # Remove index column and prepare data for table
    table_data = metrics_df.copy()
    table_data.set_index('fold', inplace=True)  # Use 'fold' column as index
    
    table = ax0.table(cellText=table_data.values,
                     colLabels=table_data.columns,
                     rowLabels=table_data.index,  # Row labels will be fold numbers
                     cellLoc='center',
                     loc='center')
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.8)
    ax0.set_title('Evaluation Metrics', pad=20, fontsize=14)
    
    # Create grouped bar plot
    ax1 = plt.subplot(gs[0, 1])
    metrics = metrics_df.columns[1:]  # Skip 'fold' column
    x = np.arange(len(metrics_df))
    width = 0.8 / len(metrics)  # Adjust bar width
    
    for i, metric in enumerate(metrics):
        ax1.bar(x + i * width, metrics_df[metric], 
                width, label=metric, alpha=0.8)
    
    ax1.set_xticks(x + width * (len(metrics) - 1) / 2)
    ax1.set_xticklabels(metrics_df.index, fontsize=10)
    ax1.legend(fontsize=10, loc='upper right')
    ax1.set_title('Metrics by Fold', pad=20, fontsize=14)
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Create visualization grid for results
    num_folds = len(metrics_df) - 1  # Exclude average row
    gs_vis = GridSpec(num_folds, 3, top=0.65, bottom=0.05, 
                     left=0.1, right=0.9, 
                     hspace=0.3, wspace=0.1)
    
    for fold in range(num_folds):
        # Load original image, prediction and ground truth
        orig = plt.imread(os.path.join(args.output_path, f'{fold+1}', 'original_1.png'))
        pred = plt.imread(os.path.join(args.output_path, f'{fold+1}', 'example_1.png'))
        gt = plt.imread(os.path.join(args.output_path, f'{fold+1}', 'ground_truth_1.png'))
        
        # Create subplots using GridSpec
        ax_orig = plt.subplot(gs_vis[fold, 0])
        ax_orig.imshow(orig)
        ax_orig.set_title(f'Fold {fold+1} Original', fontsize=12)
        ax_orig.axis('off')
        
        ax_pred = plt.subplot(gs_vis[fold, 1])
        ax_pred.imshow(pred)
        ax_pred.set_title(f'Fold {fold+1} Prediction', fontsize=12)
        ax_pred.axis('off')
        
        ax_gt = plt.subplot(gs_vis[fold, 2])
        ax_gt.imshow(gt)
        ax_gt.set_title(f'Fold {fold+1} Ground Truth', fontsize=12)
        ax_gt.axis('off')
    
    plt.suptitle(f'Evaluation Summary', fontsize=16, y=0.95)
    
    plt.savefig(os.path.join(args.output_path, 'summary.png'), 
                dpi=300, bbox_inches='tight', pad_inches=0.2)
    plt.close()

def main(args):
    metrics = {
        'DSC': [],
        'IOU': [],
    }
    avg = {
        'DSC': 0.,
        'IOU': 0.,
    }
    for metric in args.metrics.split(':'):
        metrics.setdefault(metric, [])
        avg.setdefault(metric, 0.)
    with open(os.path.join(args.output_path, 'result.csv'), 'w') as f:
        f.write(','.join(['fold'] + list(metrics.keys())) + '\n')

    num_folds = args.num_pred if args.num_pred > 0 else len(os.listdir(args.ckpt_path))
    for fold in range(num_folds):
        if args.dataset == 'isic':
            validset = ISIC2018_dataset(dataset_folder=args.data_root, folder=fold+1, train_type='test',
                                               with_name=False)
        elif args.dataset == 'chase':
            overall_id = ['01','02','03','04','05','06','07','08','09','10','11','12','13','14']
            test_id = overall_id[3*fold:3*(fold+1)]
            train_id = list(set(overall_id)-set(test_id))
            validset = MyDataset_CHASE(args, train_root=args.data_root, pat_ls=test_id, mode='test')
        else:
            raise ValueError('unsupported dataset {}'.format(args.dataset))

        val_loader = torch.utils.data.DataLoader(dataset=validset, batch_size=1, shuffle=False, pin_memory=True, num_workers=6)
        print("Number of test data: %i" % len(val_loader))

        model = DconnNet(num_class=args.num_class, decoder_attention=args.decoder_attention)
        model.load_state_dict(torch.load(f'{args.ckpt_path}/{fold+1}/best_model.pth', map_location='cpu'))
        model = model.to(device)
        model.eval()

        for metric_type in metrics.keys():
            metrics[metric_type] = []

        H, W = args.resize
        hori_translation = torch.zeros([1, args.num_class, W, W])
        for i in range(W-1):
            hori_translation[:, :, i, i + 1] = torch.tensor(1.0)
        verti_translation = torch.zeros([1, args.num_class, H, H])
        for j in range(H-1):
            verti_translation[:, :, j, j + 1] = torch.tensor(1.0)
        hori_translation = hori_translation.float()
        verti_translation = verti_translation.float()
        
        with torch.no_grad():
            for data in val_loader:
                img = Variable(data[0]).to(device)
                gt = Variable(data[1]).long().to(device)

                N, C, H, W = img.shape

                hori = hori_translation.repeat(N, 1, 1, 1).to(device)
                verti = verti_translation.repeat(N, 1, 1, 1).to(device)

                pred = predict(model, img, hori, verti, num_class=1)

                for metric in metrics.keys():
                    metrics[metric].append(get_metric(pred, gt, metric_type=metric))

        with open(os.path.join(args.output_path, 'result.csv'), 'a') as f:
            cur_metrics = []
            for metric_type, metric_list in metrics.items():
                cur_metric = np.mean(metric_list)
                cur_metrics.append('%.6f' % cur_metric)
                avg[metric_type] += cur_metric
            f.write(','.join([f'{fold + 1}'] + cur_metrics) + '\n')
        
        topk = np.argsort(metrics['DSC'])[-min(args.num_pred, len(val_loader)):][::-1]
        if not os.path.isdir(os.path.join(args.output_path, f'{fold + 1}')):
            os.makedirs(os.path.join(args.output_path, f'{fold + 1}'))
        with torch.no_grad():
            for idx, data in enumerate(val_loader):
                if idx not in topk:
                    continue
                img = Variable(data[0]).to(device)
                gt = Variable(data[1]).long().to(device)

                N, C, H, W = img.shape

                hori = hori_translation.repeat(N, 1, 1, 1).to(device)
                verti = verti_translation.repeat(N, 1, 1, 1).to(device)

                pred = predict(model, img, hori, verti, num_class=1)
                
                rank_idx = np.where(topk == idx)[0].item()
                
                # Save original image
                orig_pic = to_pil(img.squeeze(0))
                orig_pic.save(os.path.join(args.output_path, f'{fold + 1}', f'original_{rank_idx + 1}.png'))
                
                # Save prediction result
                pred_pic = to_pil(pred.squeeze(0))
                pred_pic.save(os.path.join(args.output_path, f'{fold + 1}', f'example_{rank_idx + 1}.png'))

                # Save ground truth
                gt_pic = to_pil(gt.to(pred.dtype).squeeze(0))
                gt_pic.save(os.path.join(args.output_path, f'{fold + 1}', f'ground_truth_{rank_idx + 1}.png'))
        
    
    with open(os.path.join(args.output_path, 'result.csv'), 'a') as f:
        avg_datas = ['%.6f' % (sum_metric / num_folds) for _, sum_metric in avg.items()]
        f.write(','.join([f'average'] + avg_datas) + '\n')
    
    metrics_df = pd.read_csv(os.path.join(args.output_path, 'result.csv'))
    generate_summary_plot(args, metrics_df)


if __name__ == '__main__':
    args = parse_args()
    main(args)

