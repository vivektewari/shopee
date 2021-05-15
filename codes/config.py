class Config(object):
    miscDirectory='/home/pooja/PycharmProjects/pythonProject1/shopee-product-matching/miscPlots/'
    target='label_group'
    env = 'default'
    backbone ='arc'# 'arc'#'resnet18'
    classify = 'softmax'
    num_classes = 13938
    metric = None#'arc_margin'
    easy_margin = False
    use_se = False
    loss = '0'  #thetaLoss'#'focal_loss'

    display = True
    finetune = False
    preTraining=False#True
    train_root = '/data/Datasets/webface/CASIA-maxpy-clean-crop-144/'
    train_list = '/data/Datasets/webface/train_data_13938.txt'
    val_list = '/data/Datasets/webface/val_data_13938.txt'

    test_root = '/data1/Datasets/anti-spoofing/test/data_align_256'
    test_list = 'test.txt'

    lfw_root = '/data/Datasets/lfw/lfw-align-128'
    lfw_test_list = '/data/Datasets/lfw/lfw_test_pair.txt'

    checkpoints_path = '/home/pooja/PycharmProjects/pythonProject1/trainedModels'
    load_model_path = '/home/pooja/PycharmProjects/pythonProject1/trainedModels/arcretrained_95.pth'
    test_model_path = 'checkpoints/resnet18_110.pth'
    save_interval = 5

    train_batch_size = 16  # batch size
    test_batch_size = 60

    input_shape = (1, 128, 128)

    optimizer = 'sgd'

    use_gpu = True  # use GPU or not
    gpu_id = '0, 1'
    num_workers = 4  # how many workers for loading data
    print_freq = 5 # print info every N batch

    debug_file = '/tmp/debug'  # if os.path.exists(debug_file): enter ipdb
    result_file = 'result.csv'
    numBreak=10

    max_epoch = 100
    lr =1.5 # initial learning rate 1e-1
    lr_step = 30#10
    lr_decay = 0.99  # when val_loss increase, lr = lr*lr_decay 0.95
    weight_decay = 5e-4
    def customRate(self,model):
        return [{'params': model.theta1,'lr':0.01},{'params': model.weight,'lr': 0.5}]
    def change(self,epoch,p1=None,p2=None):
        pass