"""
Author: Benny
Date: Nov 2019
"""
import argparse
import os
import torch
import logging
import sys
import importlib
from tqdm import tqdm
import numpy as np
from glob import glob
from scipy import spatial

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))


def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('PointNet')
    parser.add_argument('--batch_size', type=int, default=1000, help='batch size in testing [default: 24]')
    parser.add_argument('--gpu', type=str, default='0', help='specify gpu device [default: 0]')
    parser.add_argument('--num_point', type=int, default=500, help='Point Number [default: 2048]')
    parser.add_argument('--log_dir', type=str, default='seg_model2', help='Experiment root')
    parser.add_argument('--normal', action='store_true', default=False, help='Whether to use normal information [default: False]')
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
    experiment_dir = 'log/part_seg/' + args.log_dir

    '''LOG'''
    args = parse_args()
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('%s/eval.txt' % experiment_dir)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    log_string('PARAMETER ...')
    log_string(args)

    root = 'data/circle_data/'

    log_string('Load dataset ...')
    DATA_PATH = 'data/circle_data/test_data/'
    samples = glob(DATA_PATH+"/*.txt")
    samples.sort()
    batch_size = args.batch_size
    num_part = 2


    '''MODEL LOADING'''
    model_name = os.listdir(experiment_dir+'/logs')[0].split('.')[0]
    MODEL = importlib.import_module(model_name)
    classifier = MODEL.get_model(num_part, normal_channel=args.normal).cuda()
    checkpoint = torch.load(str(experiment_dir) + '/checkpoints/best_model.pth')
    classifier.load_state_dict(checkpoint['model_state_dict'])

    for i,item in enumerate(samples):
        print(item)
        data_name = item.split('/')[-1][:-4]
        input_data = np.loadtxt(DATA_PATH + data_name + '.txt').astype(np.float32)
        if(input_data.shape[-1]>4):
            label = input_data[:,3].astype(np.int32)
        else:
            label = np.zeros(input_data.shape[0],dtype = np.int32).astype(np.int32)
        input = input_data[:,:3]
        nbrs = spatial.cKDTree(input)

        batch_num = int(input.shape[0] / batch_size)
        add_batch = 0 if(input.shape[0] % batch_size ==0)else 1
        '''MODEL LOADING'''
        model_name = os.listdir(experiment_dir+'/logs')[0].split('.')[0]
        MODEL = importlib.import_module(model_name)
        classifier = MODEL.get_model(num_part, normal_channel=args.normal).cuda()
        checkpoint = torch.load(str(experiment_dir) + '/checkpoints/best_model.pth')
        classifier.load_state_dict(checkpoint['model_state_dict'])

        with torch.no_grad():
            test_metrics = {}
            total_correct = 0
            classifier = classifier.eval()
            predss = None
            weight = None
            seg_pred_all = None
            for i in tqdm(range(batch_num+add_batch)):
                dist,idxs = nbrs.query(input[i*batch_size:(i+1)*batch_size,:],k = 500)

                #################cad 500 ring 1000+++++++++++++++++++++++
                lis = np.arange(1,500)
                np.random.seed(i)
                np.random.shuffle(lis)
                idx_sub = np.concatenate([[0],lis[:127]],axis = 0)

                points = input[idxs[:,:16],:]
                points2 = input[idxs[:,idx_sub],:]

                points2, points = pca_normalize(points2,points) 

                points = points / dist[:,-1].reshape(-1,1,1)
                points2 = points2 / dist[:,-1].reshape(-1,1,1)

                points = torch.Tensor(points)
                points2 = torch.Tensor(points2)
                print(points.shape)
                print(points2.shape)
                points,  points2 = points.float().cuda(), points2.float().cuda()

                points = points.transpose(2, 1)
                points2 = points2.transpose(2, 1)

                seg_pred, trans_feat = classifier(points, points2)

                seg_pred = seg_pred.contiguous().view(-1, num_part)
                trans_feat = trans_feat.contiguous().view(-1)

                if(i == 0):
                    predss = seg_pred.data.max(1)[1]
                    weight = trans_feat
                    seg_pred_all = seg_pred
                else:
                    predss = torch.cat((predss,seg_pred.data.max(1)[1]),axis = 0)
                    weight = torch.cat((weight,trans_feat),axis = 0)
                    seg_pred_all = torch.cat((seg_pred_all,seg_pred),axis = 0)
            predss = predss.cpu().numpy().astype(np.int32)
            weight = weight.cpu().numpy().astype(np.float32).reshape(-1,1)
            seg_pred_all = seg_pred_all.cpu().numpy().astype(np.float32).reshape(-1,num_part)

            compare = np.where(predss == label.astype(np.int32))
            correct = compare[0].shape[0]
            instance_acc = correct/float(input.shape[0])
            log_string('%s Test Instance Accuracy: %f' % (data_name,instance_acc))
            input_weight = np.concatenate((input_data[:,:3],weight),axis = -1)
            input_seg = np.concatenate((input_data[:,:3],seg_pred_all),axis = -1)

            np.savetxt(experiment_dir+"/results/"+ data_name +"_weight.txt",input_weight,fmt = '%.6f')
            np.savetxt(experiment_dir+"/results/"+ data_name +"_weight_circles.txt",input_weight[predss > 0.5],fmt = '%.6f')
            output = np.concatenate((input_data[:,:3],predss.reshape(-1, 1)),axis = -1)
            np.savetxt(experiment_dir+"/results/"+ data_name +"_output.txt",output,fmt = '%.6f')

            #np.savetxt(experiment_dir+"/results/"+ data_name +"_seg_pred.txt",input_seg,fmt = '%.6f')
            #np.savetxt(experiment_dir+"/results/"+ data_name +"_seg_pred_circles.txt",input_seg[predss > 0.5],fmt = '%.6f')


if __name__ == '__main__':
    args = parse_args()
    main(args)

