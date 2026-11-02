"""
Author: Benny
Date: Nov 2019
"""
import argparse
import os
from data_utils.CircleDataLoader import PartNormalDataset
import torch
import datetime
import logging
from pathlib import Path
import sys
import importlib
import shutil
from tqdm import tqdm
import provider
import numpy as np
from scipy import spatial

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))

seg_classes = {'Circle': [0,1]}
seg_label_to_cat = {} 
for cat in seg_classes.keys():
    for label in seg_classes[cat]:
        seg_label_to_cat[label] = cat


def parse_args():
    parser = argparse.ArgumentParser('Model')
    parser.add_argument('--model', type=str, default='seg_model2', help='model name [default: pointnet2_part_seg_msg]')
    parser.add_argument('--batch_size', type=int, default=2, help='Batch Size during training [default: 16]')
    parser.add_argument('--epoch',  default=500, type=int, help='Epoch to run [default: 251]')
    parser.add_argument('--learning_rate', default=0.000001, type=float, help='Initial learning rate [default: 0.001]')
    parser.add_argument('--gpu', type=str, default='0', help='GPU to use [default: GPU 0]')
    parser.add_argument('--optimizer', type=str, default='Adam', help='Adam or SGD [default: Adam]')
    parser.add_argument('--log_dir', type=str, default='seg_model2', help='Log path [default: None]')
    parser.add_argument('--decay_rate', type=float, default=1e-4, help='weight decay [default: 1e-4]')
    parser.add_argument('--npoint', type=int,  default=500, help='Point Number [default: 2048]')
    parser.add_argument('--normal', action='store_true', default=False, help='Whether to use normal information [default: False]')
    parser.add_argument('--step_size', type=int,  default=20, help='Decay step for lr decay [default: every 20 epochs]')
    parser.add_argument('--lr_decay', type=float,  default=0.5, help='Decay rate for lr decay [default: 0.5]')

    return parser.parse_args()


def pca_normalize(x,x2):
    rotated_data = np.zeros(x.shape, dtype=np.float32)
    rotated_data2 = np.zeros(x2.shape, dtype=np.float32)
    x = x - np.mean(x, axis = -2, keepdims = True)
    H = np.matmul(x.transpose(0,2,1), x)
    for k in range(x.shape[0]):
        eigenvectors, eigenvalues, _ = np.linalg.svd(H[k])
        sort = eigenvalues.argsort()
        eigenvalues = eigenvalues[sort]
        eigenvectors = eigenvectors[:, sort]
        normal = eigenvectors[:,0]
        
        shape_pc = x[k, ...]
        shape_pc2 = x2[k, ...]

        #np.savetxt("./pca/res"+str(k)+".txt", shape_pc)
        q_angle = np.arctan(normal[0]/normal[2])
        q_cos = np.cos(q_angle)
        q_sin = np.sin(q_angle)    
        rotation_matrix1 = np.array([[q_cos, 0, q_sin],
                                    [0, 1, 0],
                                    [-q_sin, 0, q_cos]])
        shape_pc = np.dot(shape_pc.reshape((-1, 3)), rotation_matrix1)
        shape_pc2 = np.dot(shape_pc2.reshape((-1, 3)), rotation_matrix1)

        w_angle = np.arctan(normal[1]/np.sqrt(normal[0]*normal[0]+normal[2]*normal[2]))
        w_cos = np.cos(w_angle)
        w_sin = np.sin(w_angle)   
        rotation_matrix2 = np.array([[1, 0, 0],
                                    [0, w_cos, w_sin],
                                    [0, -w_sin, w_cos]])
        shape_pc = np.dot(shape_pc.reshape((-1, 3)), rotation_matrix2)
        shape_pc2 = np.dot(shape_pc2.reshape((-1, 3)), rotation_matrix2)

        #np.savetxt("./pca/res"+str(k)+"_oz.txt", shape_pc)
        rotated_data[k, ...] = shape_pc
        rotated_data2[k, ...] = shape_pc2

        #rotation_matrix_12 = np.dot(rotation_matrix1,rotation_matrix2)
        #np.savetxt("./pca/res"+str(k)+"_oz2.txt", np.dot(x[k, ...].reshape((-1, 3)), rotation_matrix_12))

    return rotated_data, rotated_data2


def main(args):
    def log_string(str):
        logger.info(str)
        print(str)

    '''HYPER PARAMETER'''
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    '''CREATE DIR'''
    timestr = str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M'))
    experiment_dir = Path('./log/')
    experiment_dir.mkdir(exist_ok=True)
    experiment_dir = experiment_dir.joinpath('part_seg')
    experiment_dir.mkdir(exist_ok=True)
    if args.log_dir is None:
        experiment_dir = experiment_dir.joinpath(timestr)
    else:
        experiment_dir = experiment_dir.joinpath(args.log_dir)
    experiment_dir.mkdir(exist_ok=True)
    checkpoints_dir = experiment_dir.joinpath('checkpoints/')
    checkpoints_dir.mkdir(exist_ok=True)
    log_dir = experiment_dir.joinpath('logs/')
    log_dir.mkdir(exist_ok=True)

    '''LOG'''
    args = parse_args()
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('%s/%s.txt' % (log_dir, args.model))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    log_string('PARAMETER ...')
    log_string(args)

    root = 'data/circle_data/'

    TRAIN_DATASET = PartNormalDataset(root = root, npoints=args.npoint, split='train', normal_channel=args.normal)
    trainDataLoader = torch.utils.data.DataLoader(TRAIN_DATASET, batch_size=args.batch_size,shuffle=True, num_workers=4)
    TEST_DATASET = PartNormalDataset(root = root, npoints=args.npoint, split='test', normal_channel=args.normal)
    testDataLoader = torch.utils.data.DataLoader(TEST_DATASET, batch_size=args.batch_size,shuffle=False, num_workers=4)
    log_string("The number of training data is: %d" % len(TRAIN_DATASET))
    log_string("The number of test data is: %d" %  len(TEST_DATASET))

    weights = torch.Tensor([0.5,0.5]).cuda()
    num_classes = 1
    num_part = 2
    '''MODEL LOADING'''
    MODEL = importlib.import_module(args.model)
    shutil.copy('models/%s.py' % args.model, str(experiment_dir))
    shutil.copy('models/pointnet_util.py', str(experiment_dir))
    shutil.copy('train_partseg2.py', str(experiment_dir))
    shutil.copy('data_utils/CircleDataLoader.py', str(experiment_dir))
    shutil.copy('provider.py', str(experiment_dir))

    classifier = MODEL.get_model(num_part, normal_channel=args.normal).cuda()
    criterion = MODEL.get_loss().cuda()


    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv2d') != -1:
            torch.nn.init.xavier_normal_(m.weight.data)
            torch.nn.init.constant_(m.bias.data, 0.0)
        elif classname.find('Linear') != -1:
            torch.nn.init.xavier_normal_(m.weight.data)
            torch.nn.init.constant_(m.bias.data, 0.0)

    try:
        checkpoint = torch.load(str(experiment_dir) + '/checkpoints/best_model.pth')
        start_epoch = checkpoint['epoch'] + 1
        classifier.load_state_dict(checkpoint['model_state_dict'])
        log_string('Use pretrain model')
    except:
        log_string('No existing model, starting training from scratch...')
        start_epoch = 0
        #classifier = classifier.apply(weights_init)

    if args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(
            classifier.parameters(),
            lr=args.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=args.decay_rate
        )
    else:
        optimizer = torch.optim.SGD(classifier.parameters(), lr=args.learning_rate, momentum=0.9)

    def bn_momentum_adjust(m, momentum):
        if isinstance(m, torch.nn.BatchNorm2d) or isinstance(m, torch.nn.BatchNorm1d):
            m.momentum = momentum

    LEARNING_RATE_CLIP = 1e-6
    MOMENTUM_ORIGINAL = 0.1
    MOMENTUM_DECCAY = 0.5
    MOMENTUM_DECCAY_STEP = args.step_size

    best_acc = 0
    global_epoch = 0
    best_class_avg_iou = 0
    best_inctance_avg_iou = 0

    for epoch in range(start_epoch,args.epoch):
        log_string('Epoch %d (%d/%s):' % (global_epoch, epoch, args.epoch))
        '''Adjust learning rate and BN momentum'''
        lr = max(args.learning_rate * (args.lr_decay ** (epoch // args.step_size)), LEARNING_RATE_CLIP)
        log_string('Learning rate:%f' % lr)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        mean_correct = []
        momentum = MOMENTUM_ORIGINAL * (MOMENTUM_DECCAY ** (epoch // MOMENTUM_DECCAY_STEP))
        if momentum < 0.01:
            momentum = 0.01
        print('BN momentum updated to: %f' % momentum)
        classifier = classifier.apply(lambda x: bn_momentum_adjust(x,momentum))
        num_batches = len(trainDataLoader)
        '''learning one epoch'''
        loss_sum = 0
        loss1_sum = 0
        loss2_sum = 0
        rads_all = []
        rads_gt_all = []
        for i, data in tqdm(enumerate(trainDataLoader), total=len(trainDataLoader), smoothing=0.9):
            points, target, points2,rawpoints,_ = data
            cur_batch_size, NUM_POINT, _ = points.size()
            #print(points.shape, target.shape, points2.shape,rawpoints.shape)
            points = points.data.numpy()
            points2 = points2.data.numpy()

            #points[:,:, 0:3], points2[:,:, 0:3] = pca_normalize(points[:,:, 0:3], points2[:,:, 0:3])

            for j in range(points.shape[0]):
                lis = np.arange(1,500)
                np.random.seed(i*j)
                np.random.shuffle(lis)
                idx_sub = np.concatenate([[0],lis[:127]],axis = 0)
                nbrs = spatial.cKDTree(points2[j])
                dist,idxs = nbrs.query(points[j],k=500)
                
                temp_data1 = points2[j][idxs[:,:16],:] / dist[:,-1].reshape(-1,1,1)  # nearest 16 points  
                temp_data2 = points2[j][idxs[:,idx_sub],:] / dist[:,-1].reshape(-1,1,1) # random 128 points in nearest 500 points
                if(j == 0):
                    input_s = temp_data1
                    input_b = temp_data2
                else:
                    input_s = np.concatenate([input_s,temp_data1],axis = 0)  #B*NUM_POINT,16,3
                    input_b = np.concatenate([input_b,temp_data2],axis = 0)  #B*NUM_POINT,128,3
            points = input_s
            points2 = input_b

            #data augmentation
            points = provider.random_point_dropout(points)
            points2 = provider.random_point_dropout(points2)
            points[:,:, 0:3], points2[:,:, 0:3] = provider.random_scale_point_cloud(points[:,:, 0:3], points2[:,:, 0:3])
            points[:,:, 0:3], points2[:,:, 0:3] = provider.shift_point_cloud(points[:,:, 0:3], points2[:,:, 0:3])
            points[:,:, 0:3], points2[:,:, 0:3] = provider.rotate_point_cloud(points[:,:, 0:3], points2[:,:, 0:3])
            points[:,:, 0:3], points2[:,:, 0:3] = provider.rotate_point_cloud_z(points[:,:, 0:3], points2[:,:, 0:3])

            points2, points = pca_normalize(points2,points)

            points = torch.Tensor(points)
            points2 = torch.Tensor(points2)

            points, target, points2, rawpoints = points.float().cuda(), target.long().cuda(), points2.float().cuda(), rawpoints.float().cuda()
            points = points.transpose(2, 1)
            points2 = points2.transpose(2, 1)

            optimizer.zero_grad()
            classifier = classifier.train()
            seg_pred, trans_feat = classifier(points, points2)
            
            seg_pred = seg_pred.reshape(rawpoints.shape[0],args.npoint,num_part) #reshape

            seg_pred = seg_pred.contiguous().view(-1, num_part)
            target = target.view(-1, 1)[:, 0]
            pred_choice = seg_pred.data.max(1)[1]
            correct = pred_choice.eq(target.data).cpu().sum()
            mean_correct.append(correct.item() / (cur_batch_size * NUM_POINT))

            loss1,loss2,rads,rads_gt = criterion(seg_pred, target, trans_feat, weights,rawpoints)

            rads_all.append(rads)
            rads_gt_all.append(rads_gt)

            v2 = 300
            loss = loss1 +  v2 * loss2
            loss.backward()
            optimizer.step()
            loss_sum += loss
            loss1_sum += loss1
            loss2_sum += loss2
            #break
        log_string('Training mean loss1: %f mean loss2: %f' % (loss1_sum / num_batches,loss2_sum / num_batches))
        log_string('Training mean rads: %f mean rads_gt: %f' % (
            torch.cat([x.detach() for x in rads_all]).mean().item(),
            torch.cat([x.detach() for x in rads_gt_all]).mean().item()))
        log_string('Training mean loss: %f' % (loss_sum / num_batches))
        train_instance_acc = np.mean(mean_correct)
        log_string('Train accuracy is: %.5f' % train_instance_acc)

        with torch.no_grad():
            test_metrics = {}
            cur_mean_correct = []

            for batch_id, data in tqdm(enumerate(testDataLoader), total=len(testDataLoader), smoothing=0.9):
                points, target, points2,rawpoints,_ = data
                cur_batch_size, NUM_POINT, _ = points.size()
                points = points.data.numpy()
                points2 = points2.data.numpy()

                #points[:,:, 0:3], points2[:,:, 0:3] = pca_normalize(points[:,:, 0:3], points2[:,:, 0:3])

                for j in range(points.shape[0]):
                    lis = np.arange(1,500)
                    np.random.seed(i*j)
                    np.random.shuffle(lis)
                    idx_sub = np.concatenate([[0],lis[:127]],axis = 0)
                    nbrs = spatial.cKDTree(points2[j])
                    dist,idxs = nbrs.query(points[j],k=500)

                    temp_data1 = points2[j][idxs[:,:16],:] / dist[:,-1].reshape(-1,1,1)
                    temp_data2 = points2[j][idxs[:,idx_sub],:] / dist[:,-1].reshape(-1,1,1)
                    if(j == 0):
                        input_s = temp_data1
                        input_b = temp_data2
                    else:
                        input_s = np.concatenate([input_s,temp_data1],axis = 0)  #B*NUM_POINT,16,3
                        input_b = np.concatenate([input_b,temp_data2],axis = 0)  #B*NUM_POINT,128,3
                points = input_s
                points2 = input_b

                points2, points = pca_normalize(points2,points) 

                points = torch.Tensor(points)
                points2 = torch.Tensor(points2)

                points, target, points2, rawpoints = points.float().cuda(), target.long().cuda(), points2.float().cuda(), rawpoints.float().cuda()
                points = points.transpose(2, 1)
                points2 = points2.transpose(2, 1)

                classifier = classifier.eval()
                seg_pred, _ = classifier(points, points2)

                seg_pred = seg_pred.reshape(rawpoints.shape[0],args.npoint,num_part) #reshape

                seg_pred = seg_pred.contiguous().view(-1, num_part)
                target = target.view(-1, 1)[:, 0]
                cur_pred_choice = seg_pred.data.max(1)[1]
                cur_correct = cur_pred_choice.eq(target.data).cpu().sum()
                cur_mean_correct.append(cur_correct.item() / (cur_batch_size * NUM_POINT))

        test_metrics['accuracy'] = np.mean(cur_mean_correct)
        log_string('Epoch %d test Accuracy: %f' % (epoch, test_metrics['accuracy']))


        if (epoch%200 == 0):
            logger.info('Save model...')
            savepath = str(checkpoints_dir) + '/model_'+ str(epoch) +'.pth'
            log_string('Saving at %s'% savepath)
            state = {
                'epoch': epoch,
                'train_acc': train_instance_acc,
                'test_acc': test_metrics['accuracy'],
                'model_state_dict': classifier.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }
            torch.save(state, savepath)
            log_string('Saving model....')

        if test_metrics['accuracy'] > best_acc:
            best_acc = test_metrics['accuracy']
            if (True):
                logger.info('Save model...')
                savepath = str(checkpoints_dir) + '/model_'+ str(epoch) +'.pth'
                savepath2 = str(checkpoints_dir) + '/best_model' +'.pth'
                log_string('Saving at %s'% savepath)
                state = {
                    'epoch': epoch,
                    'train_acc': train_instance_acc,
                    'test_acc': test_metrics['accuracy'],
                    'model_state_dict': classifier.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }
                if(epoch > 50):
                    torch.save(state, savepath)
                torch.save(state, savepath2)
                log_string('Saving model....')
        if test_metrics['accuracy'] > best_acc:
            best_acc = test_metrics['accuracy']
        log_string('Best accuracy is: %.5f'%best_acc)
        global_epoch+=1

if __name__ == '__main__':
    args = parse_args()
    main(args)
