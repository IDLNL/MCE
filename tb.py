import pynvml

mean=[0.485, 0.456, 0.406]
std=[0.229, 0.224, 0.225]
def denorm(input,N=2):
    x=input.data
    if N==1:
        x[0]=x[0]*std[0]+mean[0]
        x[1]=x[1]*std[1]+mean[1]
        x[2]=x[2]*std[2]+mean[2]
        #x*=255
    else:
        for i in range(len(x)):
            x[i][0]=x[i][0]*std[0]+mean[0]
            x[i][1]=x[i][1]*std[1]+mean[1]
            x[i][2]=x[i][2]*std[2]+mean[2]
            #x[i]*=255
    #print('denorm ok!')
    return x

def print_file(filename, begin=None, stop=None):
    with open(filename) as file:
        lines, f = file.readlines(), 0
        if begin is None and stop is None:
            for line in lines:
                print(line,end='')  
            print('')    
        else:
            for line in lines:
                if line==begin or f == 1:
                    print(line,end='')
                    f=1
                if line==stop:
                    return

def auto_select_gpu(gpu_num=1, used=15):
    pynvml.nvmlInit() #初始化
    gpuDeviceCount = pynvml.nvmlDeviceGetCount()#获取Nvidia GPU块数
    print("GPU个数: ", gpuDeviceCount )
    n,gpus=0,''
    for i in range(gpuDeviceCount):
        handle = pynvml.nvmlDeviceGetHandleByIndex(i)#获取GPU i的handle，后续通过handle来处理
        memoryInfo = pynvml.nvmlDeviceGetMemoryInfo(handle)#通过handle获取GPU i的信息
        print('GPU ',i,' use memory: ',memoryInfo.used / 1024 /1024,' MB')#这里是字节bytes，所以要想得到以兆M为单位就需要除以1024**2
        if(memoryInfo.used/1024/1024<used):
            n+=1
            gpus+=',' if n!=1 else ''
            gpus+=str(i)
            if(n>=gpu_num):
                break
    if(n!=gpu_num):
        print('要求的显卡不足',gpu_num,'个，请检查！ ')
        import sys
        sys.exit()
    print("将要使用的GPU为: ",gpus)
    return gpus
    
def print_dict(dict):
    for k,v in dict.items():
        if isinstance(v,list):
            suffix, format = k.split('-')
            print(suffix,format.format(*v),end='')
        else:    
            suffix, format = k.split('-')
            print(suffix,format.format(v),end=' ')
    print('')