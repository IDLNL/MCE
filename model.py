import torch
import torch.nn as nn
from torch.nn import functional as F
from resnet import resnet50
from utils import calc_acc
from GlobalCenterLoss import GlobalCenterTriplet
class Model(nn.Module):
    def __init__(self, num_classes=None, drop_last_stride=False, mutual_information=False, **kwargs):
        super(Model, self).__init__()

        self.drop_last_stride = drop_last_stride
        self.mutual_learning = mutual_information 
        self.backbone = resnet50(pretrained=True, drop_last_stride=drop_last_stride)
        self.base_dim = 2048
        
        if mutual_information:            
            self.KLDivLoss = nn.KLDivLoss(reduction='batchmean')
            self.weight_sid = kwargs.get('weight_sid', 0.5)
            self.weight_caid = kwargs.get('weight_caid', 0.5)
            self.weight_KL = kwargs.get('weight_KL', 2.0)

        print("output feat length:{}".format(self.base_dim))
        self.out_dim=2048
        self.bn_neck = nn.BatchNorm1d(self.base_dim)
        nn.init.constant_(self.bn_neck.bias, 0) 
        self.bn_neck.bias.requires_grad_(False)

        if kwargs.get('eval', False):
            return

        self.classification = kwargs.get('classification', False)
        self.global_center = kwargs.get('global_center', False)
        
        if self.classification:
            self.classifier = nn.Linear(self.base_dim, num_classes, bias=False)
        if self.mutual_learning or self.classification:
            self.id_loss = nn.CrossEntropyLoss(ignore_index=-1)
        if self.global_center:
            self.gc_loss = GlobalCenterTriplet(0.3,channel=3)
            
        self.channel, self.mem = kwargs.get('channel'), kwargs.get('mem')
        if self.mem != 0:
            self.memory, self.memory_rgb, self.memory_ir = None, None, None
            if self.channel == 3:
                self.memory_ca = None
        self.gx = kwargs.get('gx')
        
    def forward(self, inputs, labels=None, modal=0):

        feats = self.backbone(inputs,modal)
        if not self.training:
            feats = self.bn_neck(feats)
            return F.normalize(feats,dim=1)
        else:
            return self.train_forward(feats, labels)

    def train_forward(self, feat, labels):
        n = feat.size(0)//self.channel
        metric = {}
        loss = 0
        feat = self.bn_neck(feat) 
        if self.global_center:
            global_center_loss,_ = self.gc_loss(F.normalize(feat,dim=1), labels, self.memory.centers)
            loss += global_center_loss * self.global_center 
            metric.update({'GC:-{:.4f}': global_center_loss.data})        
    
        if  self.classification :
            logits = self.classifier(feat)
            cls_loss = self.id_loss(logits.float(), labels)
            loss += cls_loss
            metric.update({'CE:-{:.4f}': cls_loss.data})
        if  self.mem:
            feat = F.normalize(feat,dim=1)
            cls_loss, logits = self.memory(feat, labels, return_logits=True) 
            loss += cls_loss
            metric.update({'MEM:-{:.4f}': cls_loss.data})
        metric.update({'ACC:-{:.2f}': calc_acc(logits.data, labels) * 100.})
        if self.mutual_learning:
            if  self.mem:
                v_cls_loss = self.memory_rgb(feat.narrow(0,0,n), labels.narrow(0,0,n))
                i_cls_loss = self.memory_ir(feat.narrow(0,n,n), labels.narrow(0,n,n))  
                loss += (v_cls_loss + i_cls_loss) * self.weight_sid              
                metric.update({'vid:-{:.4f}': v_cls_loss.data})
                metric.update({'iid:-{:.4f}': i_cls_loss.data})
                temp = 0.05
                if not self.gx:
                    c_rgb = self.memory_rgb.centers
                    c_ir = self.memory_ir.centers
                    if self.channel==3:
                        c_ca =self.memory_ca.centers
                else:
                    c_rgb = self.memory_rgb.features
                    c_ir = self.memory_ir.features
                    if self.channel==3:
                        c_ca =self.memory_ca.features   
                
                logits_v = (feat.narrow(0,0,n).mm(c_rgb.t()))/temp
                logits_i = (feat.narrow(0,n,n).mm(c_ir.t()))/temp
                with torch.no_grad():
                    logits_v_ = (feat.narrow(0,0,n).mm(c_ir.t()))/temp
                    logits_i_ = (feat.narrow(0,n,n).mm(c_rgb.t()))/temp  
                logits_m = torch.cat([logits_v, logits_i], 0).float()
                logits_m_ = torch.cat([logits_i_, logits_v_], 0).float()
                
                if self.channel==3:
                    s_cls_loss = self.memory_ca(feat.narrow(0,2*n,n), labels.narrow(0,2*n,n))
                    loss += s_cls_loss * self.weight_sid * self.weight_caid
                    metric.update({'sid:-{:.4f}': s_cls_loss.data})
                    logits_s = (feat.narrow(0,2*n,n).mm(c_ca.t()))/temp
                    with torch.no_grad():
                        logits_s_ = (feat.narrow(0,2*n,n).mm(c_ir.t()))/temp
                    logits_m = torch.cat([logits_v, logits_i, logits_s], 0).float()
                    logits_m_ = torch.cat([logits_i_, logits_v_, logits_s_], 0).float()

                mod_loss = self.KLDivLoss(F.log_softmax(logits_m, 1), F.softmax(logits_m_, 1)) 
                loss += mod_loss * self.weight_KL
                metric.update({'KL:-{:.4f}': mod_loss.data})

        return loss, metric
