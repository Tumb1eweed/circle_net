import torch.nn as nn
import torch.utils.data
import torch
import torch.nn.functional as F
import numpy as np
from einops import rearrange, repeat


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
        )
        self.net2 = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net2(F.gelu(self.net(x)))

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        self.heads = heads
        self.scale = dim ** -0.5

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, mask = None):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)

        dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale
        mask_value = -torch.finfo(dots.dtype).max

        if mask is not None:
            mask = F.pad(mask.flatten(1), (1, 0), value = True)
            assert mask.shape[-1] == dots.shape[-1], 'mask has incorrect dimensions'
            mask = mask[:, None, :] * mask[:, :, None]
            dots.masked_fill_(~mask, mask_value)
            del mask

        attn = dots.softmax(dim=-1)

        out = torch.einsum('bhij,bhjd->bhid', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out =  self.to_out(out)
        return out

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Residual(PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout))),
                Residual(PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout)))
            ]))
    def forward(self, x, mask = None):
        for attn, ff in self.layers:
            x = attn(x, mask = mask)
            x = ff(x)
        return x

def rodrigues_rot(P, n0, n1):
    # If P is only 1d array (coords of single point), fix it to be matrix
    if P.ndim == 1:
        P = P[np.newaxis, :]
 
    # Get vector of rotation k and angle theta
    n0 = n0 / np.linalg.norm(n0)
    n1 = n1 / np.linalg.norm(n1)
    k = np.cross(n0, n1)
    k = k / np.linalg.norm(k)
    theta = np.arccos(np.dot(n0, n1))
 
    # Compute rotated points
    P_rot = np.zeros((len(P), 3))
    for i in range(len(P)):
        P_rot[i] = P[i] * np.cos(theta) + np.cross(k, P[i]) * np.sin(theta) + k * np.dot(k, P[i]) * (1 - np.cos(theta))
 
    return P_rot

def fit_circle_2d(x, y, w=[]):  # using torch tensor
    A = np.array([x, y, np.ones(len(x))]).T
    b = x ** 2 + y ** 2
    #print(A,b)
    # Modify A,b for weighted least squares
    if len(w) == len(x):
        W = torch.diag(w)
        A = torch.matmul(W, torch.Tensor(A).cuda())
        b = torch.matmul(W, torch.Tensor(b).cuda())
    else:
        A = torch.Tensor(A).cuda()
        b = torch.Tensor(b).cuda()
    # Solve by method of least squares
    c = torch.linalg.lstsq(A,b)[0]

    #c = np.linalg.lstsq(A.cpu().data.numpy(), b.cpu().data.numpy(), rcond=None)[0]  #np lstsq
    #print(c)

    # Get circle parameters from solution c
    xc = c[0] / 2
    yc = c[1] / 2
    r = torch.sqrt(c[2] + xc ** 2 + yc ** 2)
    return xc, yc, r

def fit_circle_2d_numpy(x, y, w=[]):
    A = np.array([x, y, np.ones(len(x))]).T
    b = x ** 2 + y ** 2
    #print(A,b)
    # Modify A,b for weighted least squares
    if len(w) == len(x):
        W = np.diag(w)
        A = np.dot(W, A)
        b = np.dot(W, b)
    # Solve by method of least squares
    c = np.linalg.lstsq(A, b, rcond=None)[0]

    # Get circle parameters from solution c
    xc = c[0] / 2
    yc = c[1] / 2
    r = np.sqrt(c[2] + xc ** 2 + yc ** 2)
    return xc, yc, r

def fit_circle_3d(x, y, z, w=[]):
    A = np.array([x, y, z, np.ones(len(x))]).T
    b = x ** 2 + y ** 2 + z ** 2
 
    # Modify A,b for weighted least squares
    if len(w) == len(x):
        W = np.diag(w)
        A = np.dot(W, A)
        b = np.dot(W, b)
 
    # Solve by method of least squares
    c = np.linalg.lstsq(A, b, rcond=None)[0]
 
    # Get circle parameters from solution c
    xc = c[0] / 2
    yc = c[1] / 2
    zc = c[2] / 2
    r = np.sqrt(c[3] + xc ** 2 + yc ** 2)
    return xc, yc, zc, r

def generate_circle_by_vectors(t, C, r, n, u):
    n = n / np.linalg.norm(n)
    u = u / np.linalg.norm(u)
    P_circle = r * np.cos(t)[:, np.newaxis] * u + r * np.sin(t)[:, np.newaxis] * np.cross(n, u) + C
    return P_circle

def circle_segmentation(cloud,w= []): # using torch tensor
    # -------------------------------------------------------------------------------
    # (1) Fitting plane by SVD for the mean-centered data
    # Eq. of plane is <p,n> + d = 0, where p is a point on plane and n is normal vector
    # -------------------------------------------------------------------------------
    P_mean = cloud.mean(axis=0)
    P_centered = cloud - P_mean
    #P_centered_weigth = P_centered * w.reshape(-1,1)
    U, s, V = np.linalg.svd(P_centered)
    #U, s, V = np.linalg.svd(np.matmul(P_centered_weigth.transpose(1,0),P_centered_weigth))
    # Normal vector of fitting plane is given by 3rd column in V
    # Note linalg.svd returns V^T, so we need to select 3rd row from V^T
    normal = V[2, :]
    d = -np.dot(P_mean, normal)  # d = -<p,n>
 
    # -------------------------------------------------------------------------------
    # (2) Project points to coords X-Y in 2D plane
    # -------------------------------------------------------------------------------
    P_xy = rodrigues_rot(P_centered, normal, [0, 0, 1])
 
    # -------------------------------------------------------------------------------
    # (3) Fit circle in new 2D coords
    # -------------------------------------------------------------------------------
    xc, yc, r = fit_circle_2d(P_xy[:, 0], P_xy[:, 1],w = w)
 
    # --- Generate circle points in 2D
    t = np.linspace(0, 2 * np.pi, 100)
    xx = xc.cpu().data.numpy() + r.cpu().data.numpy() * np.cos(t)
    yy = yc.cpu().data.numpy() + r.cpu().data.numpy() * np.sin(t)
 
    # -------------------------------------------------------------------------------
    # (4) Transform circle center back to 3D coords
    # -------------------------------------------------------------------------------
    C = rodrigues_rot(np.array([xc.cpu().data.numpy(), yc.cpu().data.numpy(), 0]), [0, 0, 1], normal) + P_mean
    C = C.flatten()
 
    #--- Generate points for fitting circle
    t = np.linspace(0, 2*np.pi, 1000)
    u = cloud[0] - C
    P_fitcircle = generate_circle_by_vectors(t, C, r.cpu().data.numpy(), normal, u)
    return P_fitcircle, C, r

def circle_segmentation_numpy(cloud, w=[]):
    # -------------------------------------------------------------------------------
    # (1) Fitting plane by SVD for the mean-centered data
    # Eq. of plane is <p,n> + d = 0, where p is a point on plane and n is normal vector
    # -------------------------------------------------------------------------------
    P_mean = cloud.mean(axis=0)
    P_centered = cloud - P_mean
    p_centered_weight = P_centered * w.reshape(-1,1)
    #U, s, V = np.linalg.svd(np.matmul(p_centered_weight.transpose(1,0),p_centered_weight))# * w.reshape(-1,1))
    U, s, V = np.linalg.svd(p_centered_weight)
    # Normal vector of fitting plane is given by 3rd column in V
    # Note linalg.svd returns V^T, so we need to select 3rd row from V^T
    normal = V[2, :]
    d = -np.dot(P_mean, normal)  # d = -<p,n>
    #print(normal)
    # -------------------------------------------------------------------------------
    # (2) Project points to coords X-Y in 2D plane
    # -------------------------------------------------------------------------------
    P_xy = rodrigues_rot(P_centered, normal, [0, 0, 1])
    # -------------------------------------------------------------------------------
    # (3) Fit circle in new 2D coords
    # -------------------------------------------------------------------------------
    xc, yc, r = fit_circle_2d(P_xy[:, 0], P_xy[:, 1],w)

    # --- Generate circle points in 2D
    t = np.linspace(0, 2 * np.pi, 100)
    xx = xc + r * np.cos(t)
    yy = yc + r * np.sin(t)
 
    # -------------------------------------------------------------------------------
    # (4) Transform circle center back to 3D coords
    # -------------------------------------------------------------------------------
    C = rodrigues_rot(np.array([xc, yc, 0]), [0, 0, 1], normal) + P_mean
    C = C.flatten()
 
    #--- Generate points for fitting circle
    t = np.linspace(0, 2*np.pi, 1000)
    u = cloud[0] - C
    P_fitcircle = generate_circle_by_vectors(t, C, r, normal, u)
    return P_fitcircle, C, r


def circle_segmentation_3d(cloud, w=[]):
    # -------------------------------------------------------------------------------
    # (1) Fitting plane by SVD for the mean-centered data
    # Eq. of plane is <p,n> + d = 0, where p is a point on plane and n is normal vector
    # -------------------------------------------------------------------------------
    P_mean = cloud.mean(axis=0)
    P_centered = cloud - P_mean
    U, s, V = np.linalg.svd(P_centered)# * w.reshape(-1,1))
 
    # Normal vector of fitting plane is given by 3rd column in V
    # Note linalg.svd returns V^T, so we need to select 3rd row from V^T
    normal = V[2, :]
    d = -np.dot(P_mean, normal)  # d = -<p,n>
 
    # -------------------------------------------------------------------------------
    # (2) Project points to coords X-Y in 2D plane
    # -------------------------------------------------------------------------------
    P_xy = rodrigues_rot(P_centered, normal, [0, 0, 1])
 
    # -------------------------------------------------------------------------------
    # (3) Fit circle in new 2D coords
    # -------------------------------------------------------------------------------
    #xc, yc, r = fit_circle_2d(P_xy[:, 0], P_xy[:, 1])
    xc, yc, zc, r = fit_circle_3d(P_xy[:, 0], P_xy[:, 1], P_xy[:, 2],w)

    # -------------------------------------------------------------------------------
    # (4) Transform circle center back to 3D coords
    # -------------------------------------------------------------------------------
    C = rodrigues_rot(np.array([xc, yc, xc]), [0, 0, 1], normal) + P_mean
    C = C.flatten()
 
    #--- Generate points for fitting circle
    t = np.linspace(0, 2*np.pi, 1000)
    u = cloud[0] - C
    P_fitcircle = generate_circle_by_vectors(t, C, r, normal, u)
    return P_fitcircle, C, r

def softmax(x):
    x_row_max = x.max(axis=-1)
    x_row_max = x_row_max.reshape(list(x.shape)[:-1]+[1])
    x = x - x_row_max
    x_exp = np.exp(x)
    x_exp_row_sum = x_exp.sum(axis=-1).reshape(list(x.shape)[:-1]+[1])
    softmax = x_exp / x_exp_row_sum
    return softmax


class get_model(nn.Module):
    def __init__(self, num_class, normal_channel=False):
        super(get_model, self).__init__()
        channel = 6 if normal_channel else 3
        self.mlp1 = np.array([64,64,128,128,256,512])
        self.mlp2 = np.array([64,64,128,128,256,512])

        self.mlp_convs1 = nn.ModuleList()
        self.mlp_convs2 = nn.ModuleList()

        self.mlp_bns1 = nn.ModuleList()
        self.mlp_bns2 = nn.ModuleList()

        last_channel1 = channel
        last_channel2 = channel
        for i in range(self.mlp1.shape[0]):
            self.mlp_convs1.append(nn.Conv1d(last_channel1, self.mlp1[i], 1))
            self.mlp_bns1.append(nn.BatchNorm1d(self.mlp1[i]))

            self.mlp_convs2.append(nn.Conv1d(last_channel2, self.mlp2[i], 1))
            self.mlp_bns2.append(nn.BatchNorm1d(self.mlp2[i]))

            last_channel1 = self.mlp1[i]
            last_channel2 = self.mlp2[i]

        self.transformer = Transformer(512, 4, 8, 512, 512, 0)

        self.fc1 = nn.Linear(512, 128)  ###############
        self.fc2 = nn.Linear(128, 32)
        self.fc3 = nn.Linear(32, num_class)

        self.bn1 = nn.BatchNorm1d(128)
        self.bn2 = nn.BatchNorm1d(32)
        self.drop1 = nn.Dropout(0.4)
        self.drop2 = nn.Dropout(0.4)

        self.fc4 = nn.Linear(512, 128)  #############33
        self.fc5 = nn.Linear(128, 32)
        self.fc6 = nn.Linear(32, 1)

        self.bn3 = nn.BatchNorm1d(128)
        self.bn4 = nn.BatchNorm1d(32)
        self.drop3 = nn.Dropout(0.4)
        self.drop4 = nn.Dropout(0.4)

    def forward(self, x1,x2):
        #x1 = x1 - torch.mean(x1,axis = -1,keepdims = True)
        x1 = x1 - x1[:,:,0:1]
        #print(x1.shape)
        x2 = x2 - x1[:,:,0:1]

        for i in range(self.mlp1.shape[0]):
            x1 = F.relu(self.mlp_bns1[i](self.mlp_convs1[i](x1)))
            x2 = F.relu(self.mlp_bns2[i](self.mlp_convs2[i](x2)))


        x1 = torch.max(x1, axis = 2)[0]
        x2 = torch.max(x2, axis = 2)[0]


        #transformer
        x = torch.cat((x2.reshape(-1,1,512),x1.reshape(-1,1,512)), axis = 1) #b,2,512
        x = self.transformer(x)
        x = x.mean(axis = 1) # x[:, 0]

        xa = self.drop1(F.relu(self.bn1(self.fc1(x))))
        xa = self.drop2(F.relu(self.bn2(self.fc2(xa))))
        xa = self.fc3(xa)
        #xb = F.softmax(xa, dim=-1)
        xa = F.log_softmax(xa, dim=-1)

        xb = self.drop3(F.relu(self.bn3(self.fc4(x))))
        xb = self.drop4(F.relu(self.bn4(self.fc5(xb))))
        xb = self.fc6(xb)
        #xb = F.log_softmax(xb, dim=-1)
        return xa, xb#[:,-1]


class get_loss(nn.Module):
    def __init__(self):
        super(get_loss, self).__init__()
        self.radtrans = torch.Tensor([0.12,0.1,0.08,0.06,0.04,0.025,0.015,0.01]).cuda()/2.0  #gt radius

    def fitting(self,pred, rawpoints, trans_feat,weighting = True):
        pred = pred.cpu().data.numpy()
        rawpoints = rawpoints.cpu().data.numpy()
        rads = []
        rads_gt = []
        #print(pred.shape)
        for i in range(pred.shape[0]):
            idx_max = np.argmax(pred[i], axis = -1)
            if(np.max(idx_max) == 0):   #if no pred circle boundary points
                rads.append(torch.Tensor([0]).cuda())
                rads_gt.append(torch.Tensor([0.06]).cuda())
                continue

            idx = np.where(idx_max == 1)[0]

            if(idx.shape[0]<10):         #if less than 10 pred circle boundary points
                rads.append(torch.Tensor([0]).cuda())
                rads_gt.append(torch.Tensor([0.06]).cuda())
                continue

            cicle_pionts = rawpoints[i][idx] 

            w = trans_feat[i][idx]
            w = F.softmax(w,dim=-1)
            w = w#.cpu().data.numpy()
            idx2 = cicle_pionts[:,-1]  #gt radius idx

            if(np.max(idx2) < -0.5):   #if no boundary points in gt
                rads.append(torch.Tensor([0]).cuda())
                rads_gt.append(torch.Tensor([0.06]).cuda())
                continue
            if(weighting):
                _,_,rad = circle_segmentation(cicle_pionts[:,:3],w)
            else:
                #print('no weighting')
                _,_,rad = circle_segmentation(cicle_pionts[:,:3])
            rads.append(rad.unsqueeze(0))
            rads_gt.append(self.radtrans[np.max(idx2).astype(np.int32)].unsqueeze(0))

        rads = torch.cat(rads,axis = 0)
        rads_gt = torch.cat(rads_gt,axis = 0)
            
        return rads,rads_gt
        
    def forward(self, pred, target, trans_feat, weight, rawpoints):
        #print(pred.shape, target.shape, rawpoints.shape)
        loss1 = F.nll_loss(pred, target , weight = weight)
        rads,rads_gt = self.fitting(pred.reshape(rawpoints.shape[0],500,2), rawpoints, trans_feat.reshape(rawpoints.shape[0],500), True)

        loss2 = torch.mean((rads-rads_gt)**2)

        return loss1,loss2,rads,rads_gt




