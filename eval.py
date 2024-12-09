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

    num_folds = len(os.listdir(args.ckpt_path))
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
                pred_pic = to_pil(pred.squeeze(0))
                pred_pic.save(os.path.join(args.output_path, f'{fold + 1}', f'example_{rank_idx + 1}.png'))

                gt_pic = to_pil(gt.to(pred.dtype).squeeze(0))
                gt_pic.save(os.path.join(args.output_path, f'{fold + 1}', f'ground_truth_{rank_idx + 1}.png'))
        
    
    with open(os.path.join(args.output_path, 'result.csv'), 'a') as f:
        avg_datas = ['%.6f' % (sum_metric / num_folds) for _, sum_metric in avg.items()]
        f.write(','.join([f'average'] + avg_datas) + '\n')


if __name__ == '__main__':
    args = parse_args()
    main(args)

