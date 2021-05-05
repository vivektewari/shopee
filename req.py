from datas import *
from sklearn import preprocessing
lb = preprocessing.LabelBinarizer()
args = parse_args()
split=True
if split==True:
    df, out_dim = get_df(args.kernel_type, args.data_dir, args.train_step)#, nrows=1000
    #df['label_group'] = df['label_group'].astype('category').cat.codes
    train,hold=holdOut(df, 'label_group')
    #train['label_group'] = train['label_group'].astype('category').cat.codes
    #hold['label_group'] = hold['label_group'].astype('category').cat.codes
    lb.fit(hold['label_group'])
    hold['label_group'] = hold['label_group'].apply(lambda x: np.array(lb.transform([x])).flatten())
    lb.fit(train['label_group'])
    train['label_group'] = train['label_group'].apply(lambda x: np.array(lb.transform([x])).flatten())
else:

    df= pd.read_csv('../input/shopee-product-matching/train.csv')#,nrows=500)
    tmp = df.groupby('label_group').posting_id.agg('unique').to_dict()
    df['target'] = df.label_group.map(tmp)
#         df=pd.read_csv(args.data_dir+'holdOut.csv',nrows=500)
#         df['target'] = df['target'].apply(packing.unpack)
    maxCat=len(df['label_group'].unique())
    df['label_group'] = df['label_group'].astype('category').cat.codes

model = 'imageMatch'#user_id


if model == 'modelImagePhash':
    from models import modelImagePhash
    train = modelImagePhash(df, 'pred0')
    train['f1'] = train.apply(getMetric('pred0'), axis=1)
    print('CV score for baseline =', train.f1.mean())
if model == 'imageMatch':

        #data set
    train2=train.copy()
    hold2=hold.copy()
    train2['filepath'] = train2['image'].apply(lambda x: os.path.join(args.data_dir, 'train_images', x))
    hold2['filepath'] = hold2['image'].apply(lambda x: os.path.join(args.data_dir, 'train_images', x))

    tokenizer =None# AutoTokenizer.from_pretrained('bert-base-uncased')
    dataset_train = LandmarkDataset(train2, 'train', 'train', transform=get_transforms(image_size=256),
                                   tokenizer=None)
    train_loader = DataLoader(dataset_train, batch_size=64, num_workers=2)
    dataset_hold = LandmarkDataset(hold2, 'train', 'train', transform=get_transforms(image_size=256),
                                   tokenizer=None)
    hold_loader = DataLoader(dataset_hold, batch_size=256, num_workers=2)
    print(len(dataset_train), len(dataset_train[0][0][0]))
    print(len(dataset_hold), len(dataset_hold[0][0][0]))
    # v=dataset_test.trialImage(0)
    # f=dataset_test[0]
    # f2=dataset_test[0][0]
    # f3=dataset_test[0][0][0]

    training =False
    if training == True:pass
    elif training == False:
        model_ft = torchvision.models.resnet18(pretrained=True)
        for param in model_ft.parameters():param.requires_grad = False
#         num_ftrs = model_ft.fc.in_features
#         model_ft.fc = nn.Linear(num_ftrs, maxCat)
        #model_ft=DistributedDataParallel(model_ft, delay_allreduce=True)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model_ft = model_ft.to(device)