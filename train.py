from __future__ import print_function
import argparse, sys, time, torch
import torch.optim as optim
import torch.utils.data as data
import torch.nn.functional as F
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
from utils import *
from data_manager import *
from data_loader import SYSUData, RegDBData, Dataloader_MEM, TestData, ChannelExchange
from memory import ClusterMemory
from model import Model
from eval_metrics import eval_sysu, eval_regdb
from tensorboardX import SummaryWriter

parser = argparse.ArgumentParser(description='PyTorch Cross-Modality Training')
parser.add_argument('--phase', default='debug', type=str, help='debug or train')
parser.add_argument('--dataset', default='sysu', help='dataset name: regdb or sysu]')
parser.add_argument('--lr', default=0.00035 , type=float, help='learning rate, 0.00035 for adam')
parser.add_argument('--optim', default='adam', type=str, help='optimizer')
parser.add_argument('--resume', '-r', default='', type=str,help='resume from checkpoint')
parser.add_argument('--save_epoch', default=20, type=int,metavar='s', help='save model every 10 epochs')
parser.add_argument('--testonly', action='store_true', help='test only')        
parser.add_argument('--model_path', default='save_model/mpa_model/', type=str, help='model save path')
parser.add_argument('--log_path', default='log/', type=str, help='log save path')
parser.add_argument('--vis_log_path', default='log/vis_mpa_log/', type=str, help='log save path')
parser.add_argument('--workers', default=4, type=int, metavar='N', help='number of data loading workers (default: 4)')
parser.add_argument('--img_w', default=144, type=int,metavar='imgw', help='img width')
parser.add_argument('--img_h', default=288, type=int, metavar='imgh', help='img height')
parser.add_argument('--batch-size', default=8, type=int,metavar='B', help='training batch size')
parser.add_argument('--test-batch', default=64, type=int, metavar='tb', help='testing batch size')
parser.add_argument('--rerank', default= "no" , type=str, metavar='rerank', help='gamma for the hard mining')
parser.add_argument('--num_pos', default=4, type=int,help='num of pos per identity in each modality')
parser.add_argument('--trial', default=1, type=int, metavar='t', help='trial (only for RegDB dataset)')
parser.add_argument('--gpu', default='3', type=str, help='gpu device ids for CUDA_VISIBLE_DEVICES')
parser.add_argument('--mode', default='all', type=str, help='all or indoor')
parser.add_argument('--shot', default=1, type=int, help='select single-shot or multi-shot')
parser.add_argument('--channel', default= 3 , type=int, metavar='channel', help='gamma for the hard mining')      
parser.add_argument('--ml', default= 1 , type=int, metavar='ml', help='gamma for the hard mining') 
parser.add_argument('--kl', default= 1.2 , type=float, metavar='kl', help='use kl loss and the weight')
parser.add_argument('--sid', default= 1 , type=float, metavar='kl', help='use kl loss and the weight')  
parser.add_argument('--mem', default= 1 , type=float, metavar='mem', help='memory')
parser.add_argument('--nhard', action='store_false', help='not use hard memory')
parser.add_argument('--nce', action='store_true', help='not use cross entropy loss')
parser.add_argument('--gc', default=1, type=float, help='global center loss')
parser.add_argument('--caid', default= 0.4 , type=float, metavar='id loss for ca', help='use kl loss and the weight')
parser.add_argument('--mem_up', default= 0.25 , type=float, metavar='id loss for ca', help='use kl loss and the weight')
parser.add_argument('--gx', default= 0 , type=int, help='if use updated memory')
args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
set_seed(0)
cudnn.deterministic = True
cudnn.benchmark = False
dataset = args.dataset
if dataset == 'sysu':
    data_path = '../datasets/sysu/ori_data/'
    log_path = args.log_path + 'sysu_mpa_log//'
    test_mode = [1, 2]  # thermal to visible
    args.img_w, args.img_h = 128, 384
elif dataset == 'regdb':
    data_path = '../datasets/RegDB/'
    log_path = args.log_path + 'regdb_mpa_log/'
    test_mode = [2, 1]  # visible to thermal-[2,1]
    args.img_w, args.img_h = 144, 288 
    args.num_pos = 5
checkpoint_path = args.model_path
if not os.path.isdir(log_path):
    os.makedirs(log_path)
if not os.path.isdir(checkpoint_path):
    os.makedirs(checkpoint_path)
if not os.path.isdir(args.vis_log_path):
    os.makedirs(args.vis_log_path)
suffix = args.phase+'_' + dataset
suffix += '_mem' if args.mem!=0 else ''
suffix = suffix + '_p{}_n{}_ch{}_lr_{}'.format(args.num_pos, args.batch_size,(args.channel), args.lr) 
if not args.optim == 'sgd':
    suffix = suffix + '_' + args.optim
if dataset == 'regdb':
    suffix = suffix + '_trial_{}'.format(args.trial)
sys.stdout = Logger(log_path + suffix + '.txt')
vis_log_dir = args.vis_log_path + suffix + '/'
if not os.path.isdir(vis_log_dir):
    os.makedirs(vis_log_dir)
writer = SummaryWriter(vis_log_dir)
print('========')
arg_s = "Args:{}".format(args)
for i in range(len(arg_s)//130+1):
    print(arg_s[0+130*i:130+130*i])
print('========')
device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0
print('==> Loading data..')
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
transform_center = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((args.img_h, args.img_w)),
    transforms.ToTensor(),
    normalize,
    ChannelExchange(gray = 2)]) 
transform_test = transforms.Compose( [
    transforms.ToPILImage(),
    transforms.Resize((args.img_h, args.img_w)),
    transforms.ToTensor(),
    normalize])
end = time.time()
if dataset == 'sysu':
    trainset = SYSUData(data_path, transform=None,size=(args.img_h,args.img_w))
    color_pos, thermal_pos = GenIdx(trainset.train_color_label, trainset.train_thermal_label)
    query_img, query_label, query_cam = process_query_sysu(data_path, mode=args.mode)

    gall_img, gall_label, gall_cam = process_gallery_sysu(data_path, mode=args.mode, trial=0, shot=args.shot)
elif dataset == 'regdb':
    trainset = RegDBData(data_path, args.trial, transform=None,size=(args.img_w, args.img_h))
    color_pos, thermal_pos = GenIdx(trainset.train_color_label, trainset.train_thermal_label)
    query_img, query_label = process_test_regdb(data_path, trial=args.trial, modal=test_mode[0])
    gall_img, gall_label = process_test_regdb(data_path, trial=args.trial, modal=test_mode[1])
gallset  = TestData(gall_img, gall_label, transform=transform_test, img_size=(args.img_w, args.img_h))
queryset = TestData(query_img, query_label, transform=transform_test, img_size=(args.img_w, args.img_h))
gall_loader = data.DataLoader(gallset, batch_size=args.test_batch, shuffle=False, num_workers=args.workers)
query_loader = data.DataLoader(queryset, batch_size=args.test_batch, shuffle=False, num_workers=args.workers)
n_class = len(np.unique(trainset.train_color_label))
nquery = len(query_label)
ngall = len(gall_label)
print('Dataset {} statistics:'.format(dataset))
print('  ------------------------------')
print('  subset   | # ids | # images')
print('  ------------------------------')
print('  visible  | {:5d} | {:8d}'.format(n_class, len(trainset.train_color_label)))
print('  thermal  | {:5d} | {:8d}'.format(n_class, len(trainset.train_thermal_label)))
print('  ------------------------------')
print('  query    | {:5d} | {:8d}'.format(len(np.unique(query_label)), nquery))
print('  gallery  | {:5d} | {:8d}'.format(len(np.unique(gall_label)), ngall))
print('  ------------------------------')
print('Data Loading Time:\t {:.3f}'.format(time.time() - end))
print('==> Building model..')
net = Model(n_class,
            mutual_information=(args.ml!=0),
            drop_last_stride=True,  
            weight_KL=args.kl,
            weight_sid=args.sid, 
            classification=args.nce,
            channel = args.channel,
            mem = args.mem,
            global_center =args.gc,
            weight_caid=args.caid,
            gx = args.gx
)
net.to(device)

if len(args.resume) > 0:
    model_path =  args.resume
    if os.path.isfile(model_path):
        print('==> loading checkpoint {}'.format(args.resume))
        checkpoint = torch.load(model_path)
        start_epoch = checkpoint['epoch']
        net.load_state_dict(checkpoint['net'],strict=False)
        print('==> loaded checkpoint {} (epoch {})'
              .format(args.resume, checkpoint['epoch']))
    else:
        print('==> no checkpoint found at {}'.format(args.resume))

optimizer = optim.Adam(net.parameters(), lr=args.lr, weight_decay=0.0005)

def adjust_learning_rate(optimizer, epoch): # for sysu:[20,40]  for regdb:[30,60]  for CE:[60,90]
    lr = args.lr
    lr_decay = [20, 40] if args.dataset=='sysu' else [30,60]
    if epoch < 10:
        lr = args.lr * (epoch + 1) / 10
    if epoch > lr_decay[0]:
        lr = args.lr *0.1
    if epoch >= lr_decay[1]:
        lr = args.lr * 0.01
    for i in range(len(optimizer.param_groups)):
        optimizer.param_groups[i]['lr'] = lr
    return lr

def train(epoch):
    current_lr = adjust_learning_rate(optimizer, epoch)
    data_time = AverageMeter()
    
    if args.mem!=0:
        print('    Generate classes center for memory...')
        net.eval()
        Memset = Dataloader_MEM(data_path, dataset=trainset,size=(args.img_h,args.img_w))
        memory = [ClusterMemory(net.out_dim,n_class,use_hard=args.nhard,momentum=0.1).to(device)]
        memory.append(ClusterMemory(net.out_dim,n_class,use_hard=False,momentum=args.mem_up).to(device))
        memory.append(ClusterMemory(net.out_dim,n_class,use_hard=False,momentum=args.mem_up).to(device))
        memory.append(ClusterMemory(net.out_dim,n_class,use_hard=False,momentum=args.mem_up).to(device))

        ''' Global  ||  RGB  ||  IR  ||  Aux '''
        centers = torch.zeros(args.channel + 1, n_class, net.out_dim).cuda()
        log = torch.zeros(args.channel, n_class).cuda()
        for c in range(args.channel):
            Memset.choose = c 
            memloader = data.DataLoader(Memset, batch_size=args.test_batch,sampler=None, num_workers=args.workers, drop_last=False)
            with torch.no_grad():
                for batch_idx, (input, label) in enumerate(memloader):
                    unique = label.unique()
                    input = Variable(input.cuda())
                    feat = net(input,modal=c+1-(c==2)*1)
                    for i in unique:
                        log[c][i] += label.eq(i).sum()
                        centers[c+1][i] += feat[label.eq(i)].sum(0)
                centers[c+1] /= log[c].unsqueeze(1)
                memory[c+1].features = F.normalize(centers[c+1], dim=1)
                memory[c+1].centers = memory[c+1].features.clone()
                memory[0].features += memory[c+1].features
        memory[0].features /= args.channel
        memory[0].centers = memory[0].features.clone()
        print('    Generate OK!!,  Got ', n_class, ' Classes')
        net.memory, net.memory_rgb, net.memory_ir = memory[0], memory[1], memory[2]
        
        if args.channel == 3:
            net.memory_ca = memory[3]
            net.center_ca = memory[3].features.clone()
 
    net.train()
    end = time.time()
    for batch_idx, (input10, input11, input2, label1, label2) in enumerate(trainloader):
        input2 = Variable(input2.cuda())
        input10 = Variable(input10.cuda()) 
        if(args.channel==2):
            labels = torch.cat((label1, label2), 0)
            input = torch.cat((input10, input2,),0)
        elif(args.channel==3):
            labels = torch.cat((label1, label2, label1 ), 0)
            input11 = Variable(input11.cuda())
            input = torch.cat((input10, input2, input11),0)
        labels = Variable(labels.cuda())
        
        data_time.update(time.time() - end)
        loss, metric = net(input,labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        metric.update({'Loss:-{:.4f}':loss, 'lr:-{:.6f}':current_lr})
        end = time.time()
        if batch_idx % 50 == 0:
            print('Epoch: [{}][{:03d}/{}] '.format(epoch, batch_idx, len(trainloader)),end='')
            print_dict(metric)

    writer.add_scalar('total_loss', loss.data, epoch)
    writer.add_scalar('id_loss', metric.get('ce:-{:.4f}',0), epoch)
    writer.add_scalar('tri_loss', metric.get('tri:-{:.4f}',0), epoch)
    writer.add_scalar('lr', current_lr, epoch)


def Extract_Galley():
    # switch to evaluation mode
    net.eval()
    #print('Extracting Gallery Feature...')
    start = time.time()
    ptr = 0
    gall_feat = np.zeros((ngall, net.out_dim))
    with torch.no_grad():
        for batch_idx, (input, label) in enumerate(gall_loader):
            batch_num = input.size(0)
            input = Variable(input.cuda())
            feat = net(input,modal=1)
            gall_feat[ptr:ptr + batch_num, :] = feat.detach().cpu().numpy()
            ptr = ptr + batch_num
    return gall_feat
    
def Extract_Query():
    # switch to evaluation
    net.eval()
    #print('Extracting Query Feature...')
    start = time.time()
    ptr = 0
    query_feat = np.zeros((nquery, net.out_dim))
    with torch.no_grad():
        for batch_idx, (input, label) in enumerate(query_loader):
            batch_num = input.size(0)
            input = Variable(input.cuda())
            feat = net(input,modal=2)
            query_feat[ptr:ptr + batch_num, :] = feat.detach().cpu().numpy()
            ptr = ptr + batch_num
    return query_feat
def test(epoch, query_feat=None, print_log=True):
    gall_feat = Extract_Galley()
    if query_feat is None:
        query_feat = Extract_Query()
    start = time.time()
    # compute the similarity
    distmat = np.matmul(query_feat, np.transpose(gall_feat))
    if args.rerank=="k":
        distmat = -k_reciprocal(query_feat, gall_feat)
    # evaluation
    if dataset == 'regdb':
        cmc, mAP, mINP      = eval_regdb(-distmat, query_label, gall_label)
    elif dataset == 'sysu':
        cmc, mAP, mINP = eval_sysu(-distmat, query_label, gall_label, query_cam, gall_cam)
    if print_log:
        writer.add_scalar('rank1', cmc[0], epoch)
        writer.add_scalar('mAP', mAP, epoch)
        writer.add_scalar('mINP', mINP, epoch)
    return cmc, mAP, mINP

def test_all(epoch, muti_shot=False, print_log=True):
    query_feat = Extract_Query()
    global gall_loader, query_loader
    if dataset == 'sysu':
        for trial in range(10):
            gall_img, gall_label, gall_cam = process_gallery_sysu(data_path, mode=args.mode, trial=trial, shot=args.shot)
            gallset = TestData(gall_img, gall_label, transform=transform_test, img_size=(args.img_w, args.img_h))
            gall_loader = data.DataLoader(gallset, batch_size=args.test_batch, shuffle=False, num_workers=4)
            cmc, mAP, mINP = test(epoch, query_feat=query_feat, print_log=print_log)
            if trial == 0:
                    all_cmc = cmc
                    all_mAP = mAP
                    all_mINP = mINP
            else:
                all_cmc = all_cmc + cmc
                all_mAP = all_mAP + mAP
                all_mINP = all_mINP + mINP
            if not print_log or trial == 0:
                print('Trial: {}   Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| Rank-20: {:.2%}| mAP: {:.2%}| mINP: {:.2%}'.format(
                trial, cmc[0], cmc[4], cmc[9], cmc[19], mAP, mINP))
        return all_cmc/10, all_mAP/10, all_mINP/10
    else:
        cmc, mAP, mINP = test(start_epoch,query_feat=query_feat)
        return cmc, mAP, mINP
    

if args.testonly:
    print('Test Epoch: {}'.format(start_epoch))
    if dataset == 'sysu':
        cmc, mAP, mINP = test_all(start_epoch, print_log=False)
        print('Mean:   Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| Rank-20: {:.2%}| mAP: {:.2%}| mINP: {:.2%}'.format(
            cmc[0], cmc[4], cmc[9], cmc[19], mAP, mINP))
    else:
        cmc, mAP, mINP = test_all(start_epoch, print_log=False)
        print('Trial: {}   Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| Rank-20: {:.2%}| mAP: {:.2%}| mINP: {:.2%}'.format(
            1,cmc[0], cmc[4], cmc[9], cmc[19], mAP, mINP))
        all_cmc = cmc
        all_mAP = mAP
        all_mINP = mINP
        for trial in range(2,11):
            query_img, query_label = process_test_regdb(data_path, trial=trial, modal=test_mode[0])
            gall_img, gall_label = process_test_regdb(data_path, trial=trial, modal=test_mode[1])
            gallset  = TestData(gall_img, gall_label, transform=transform_test, img_size=(args.img_w, args.img_h))
            queryset = TestData(query_img, query_label, transform=transform_test, img_size=(args.img_w, args.img_h))
            gall_loader = data.DataLoader(gallset, batch_size=args.test_batch, shuffle=False, num_workers=args.workers)
            query_loader = data.DataLoader(queryset, batch_size=args.test_batch, shuffle=False, num_workers=args.workers)
            
            model_path = model_path[:-8] + str(trial) + '_best.t'
            print('==> loading checkpoint {}'.format(model_path))
            checkpoint = torch.load(model_path)
            start_epoch = checkpoint['epoch']
            net.load_state_dict(checkpoint['net'],strict=False)
            print('==> loaded checkpoint {} (epoch {})'.format(model_path, checkpoint['epoch']))
            cmc, mAP, mINP = test_all(start_epoch, print_log=False)
            all_cmc = all_cmc + cmc
            all_mAP = all_mAP + mAP
            all_mINP = all_mINP + mINP
            print('Trial: {}   Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| Rank-20: {:.2%}| mAP: {:.2%}| mINP: {:.2%}'.format(
                trial, cmc[0], cmc[4], cmc[9], cmc[19], mAP, mINP))
        print('Mean:   Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| Rank-20: {:.2%}| mAP: {:.2%}| mINP: {:.2%}'.format(
                all_cmc[0]/10, all_cmc[4]/10, all_cmc[9]/10, all_cmc[19]/10, all_mAP/10, all_mINP/10))    
    sys.exit()
    
# training
print('==> Start Training...')
for epoch in range(start_epoch, 160 + start_epoch):
    sampler = IdentitySampler(trainset.train_color_label, \
                              trainset.train_thermal_label, color_pos, thermal_pos, args.num_pos, args.batch_size,epoch)
    trainset.cIndex = sampler.index1
    trainset.tIndex = sampler.index2
    loader_batch = args.batch_size * args.num_pos
    trainloader = data.DataLoader(trainset, batch_size=loader_batch,sampler=sampler, num_workers=args.workers, drop_last=True)
    # training
    train(epoch)

    if epoch > 10 and epoch%2==0:
        # testing
        cmc, mAP, mINP = test_all(epoch)
        # save model
        if mAP > best_acc:
            best_acc = mAP
            best_epoch = epoch
            state = {   'net': net.state_dict(),
                        'cmc': cmc,
                        'mAP': mAP,
                        'mINP': mINP,
                        'epoch': epoch,}
            torch.save(state, checkpoint_path + suffix + '_best.t')

        print('Test Epoch [{}]'.format(epoch),end='')
        print('Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| Rank-20: {:.2%}| mAP: {:.2%}| mINP: {:.2%}'.format(
            cmc[0], cmc[4], cmc[9], cmc[19], mAP, mINP))
        print('Best Epoch [{}]'.format(best_epoch),end='')
        print('Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| Rank-20: {:.2%}| mAP: {:.2%}| mINP: {:.2%}'.format(
            state['cmc'][0], state['cmc'][4], state['cmc'][9], state['cmc'][19], state['mAP'], state['mINP']))
        