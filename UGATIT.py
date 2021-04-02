import time, itertools
from dataset import ImageFolder
from torchvision import transforms
from torch.utils.data import DataLoader
from networks import *
from utils import *
from glob import glob
import neptune
import ADA
from matplotlib import pyplot as plt

class UGATIT(object) :
    def __init__(self, args):
        self.params = vars(args)
        self.light = args.light

        if self.light :
            self.model_name = 'UGATIT_light'
        else :
            self.model_name = 'UGATIT'

        self.result_dir = args.result_dir
        self.dataset = args.dataset

        self.iteration = args.iteration
        self.decay_flag = args.decay_flag
        self.notebook = args.notebook
        self.batch_size = args.batch_size
        self.print_freq = args.print_freq
        self.save_freq = args.save_freq
        self.metric_freq = args.metric_freq

        self.lr = args.lr
        self.weight_decay = args.weight_decay
        self.ch = args.ch

        """ Weight """
        self.adv_weight = args.adv_weight
        self.cycle_weight = args.cycle_weight
        self.identity_weight = args.identity_weight
        self.cam_weight = args.cam_weight
        self.group_weight = args.group_weight

        """ Generator """
        self.n_res = args.n_res

        """ Discriminator """
        self.n_dis = args.n_dis

        self.img_size = args.img_size
        self.img_ch = args.img_ch

        self.device = args.device
        self.benchmark_flag = args.benchmark_flag
        self.resume = args.resume

        self.logit_list_real = []
        self.logit_list_fake = []
        self.group_loss_D_Global = []
        self.group_loss_D_Local = []
        self.group_loss_G_Global = []
        self.group_loss_G_Local = []
        self.kid_list = []
        self.start_iter = 1
        self.current_step = 1
        self.act_true = None#Inception activation of train data
        self.ada = ADA.ADA()
        self.ada.strength = args.strength
        self.ada.tune_kimg = args.kimg
        self.ada2 = ADA.ADA()
        self.ada2.strength = args.strength
        self.ada2.tune_kimg = args.kimg
        self.useADA = args.useADA
        self.group_list = sorted([int(x) for x in args.group.split(',')])
        self.neptune = args.neptune
        self.use_grouploss = args.use_grouploss
        self.datasetA_list = []
        self.loaderA_list = []
        self.datasetB_list = []
        self.loaderB_list = []

        if torch.backends.cudnn.enabled and self.benchmark_flag:
            print('set benchmark !')
            torch.backends.cudnn.benchmark = True

        print()

        print("##### Information #####")
        print("# light : ", self.light)
        print("# dataset : ", self.dataset)
        print("# batch_size : ", self.batch_size)
        print("# iteration per epoch : ", self.iteration)

        print()

        print("##### Generator #####")
        print("# residual blocks : ", self.n_res)

        print()

        print("##### Discriminator #####")
        print("# discriminator layer : ", self.n_dis)

        print()

        print("##### Weight #####")
        print("# adv_weight : ", self.adv_weight)
        print("# cycle_weight : ", self.cycle_weight)
        print("# identity_weight : ", self.identity_weight)
        print("# cam_weight : ", self.cam_weight)

    ##################################################################################
    # Model
    ##################################################################################

    def build_model(self):
        """ DataLoader """
        train_transform = transforms.Compose([


            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ])
        test_transform = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ])

        self.trainA_0 = ImageFolder(os.path.join('dataset', self.dataset, 'trainA', '0'), 0, train_transform)
        self.trainA_1 = ImageFolder(os.path.join('dataset', self.dataset, 'trainA', '1'), 1, train_transform)
        self.trainA_2 = ImageFolder(os.path.join('dataset', self.dataset, 'trainA', '2'), 2, train_transform)
        self.trainA_3 = ImageFolder(os.path.join('dataset', self.dataset, 'trainA', '3'), 3, train_transform)
        self.datasetA_list = [self.trainA_0, self.trainA_1,self.trainA_2,self.trainA_3]

        self.trainB_0 = ImageFolder(os.path.join('dataset', self.dataset, 'trainB', '0'), 0, train_transform)
        self.trainB_1 = ImageFolder(os.path.join('dataset', self.dataset, 'trainB', '1'), 1, train_transform)
        self.trainB_2 = ImageFolder(os.path.join('dataset', self.dataset, 'trainB', '2'), 2, train_transform)
        self.trainB_3 = ImageFolder(os.path.join('dataset', self.dataset, 'trainB', '3'), 3, train_transform)
        self.datasetB_list = [self.trainB_0, self.trainB_1, self.trainB_2, self.trainB_3]

        self.testA = ImageFolder(os.path.join('dataset', self.dataset, 'testA'), -1,test_transform)
        self.testB = ImageFolder(os.path.join('dataset', self.dataset, 'testB'), -1,test_transform)

        self.trainA_0_loader = DataLoader(self.trainA_0, batch_size=self.batch_size, shuffle=True)
        self.trainA_1_loader = DataLoader(self.trainA_1, batch_size=self.batch_size, shuffle=True)
        self.trainA_2_loader = DataLoader(self.trainA_2, batch_size=self.batch_size, shuffle=True)
        self.trainA_3_loader = DataLoader(self.trainA_3, batch_size=self.batch_size, shuffle=True)
        self.loaderA_list = [self.trainA_0_loader, self.trainA_1_loader, self.trainA_2_loader, self.trainA_3_loader]

        self.trainB_0_loader = DataLoader(self.trainB_0, batch_size=self.batch_size, shuffle=True)
        self.trainB_1_loader = DataLoader(self.trainB_1, batch_size=self.batch_size, shuffle=True)
        self.trainB_2_loader = DataLoader(self.trainB_2, batch_size=self.batch_size, shuffle=True)
        self.trainB_3_loader = DataLoader(self.trainB_3, batch_size=self.batch_size, shuffle=True)
        self.loaderB_list = [self.trainB_0_loader, self.trainB_1_loader, self.trainB_2_loader, self.trainB_3_loader]

        self.testA_loader = DataLoader(self.testA, batch_size=1, shuffle=True)
        self.testB_loader = DataLoader(self.testB, batch_size=1, shuffle=True)

        """ Adjust dataset list and dataloader list """
        for i in range(4):
            if not(i in self.group_list):
                self.datasetA_list[i] = None
                self.datasetB_list[i] = None
                self.loaderA_list[i] = None
                self.loaderB_list[i] = None
        self.datasetA_list = removeNone(self.datasetA_list)
        self.datasetB_list = removeNone(self.datasetB_list)
        self.loaderA_list = removeNone(self.loaderA_list)
        self.loaderB_list = removeNone(self.loaderB_list)

        # Sanity check
        assert len(self.loaderA_list) == len(self.loaderB_list) == len(self.datasetA_list) == \
               len(self.datasetB_list) == len(self.group_list), 'Group num differs'
        self.iterA_list = [iter(x) for x in self.loaderA_list]
        self.iterB_list = [iter(x) for x in self.loaderB_list]

        """ Define Generator, Discriminator """
        self.genA2B = ResnetGenerator(input_nc=3, output_nc=3, ngf=self.ch, n_blocks=self.n_res, img_size=self.img_size, light=self.light).to(self.device)
        self.genB2A = ResnetGenerator(input_nc=3, output_nc=3, ngf=self.ch, n_blocks=self.n_res, img_size=self.img_size, light=self.light).to(self.device)
        #Global attention
        self.disGA = Discriminator(input_nc=3, ndf=self.ch, n_layers=7).to(self.device)
        self.disGB = Discriminator(input_nc=3, ndf=self.ch, n_layers=7).to(self.device)
        #Local attention
        self.disLA = Discriminator(input_nc=3, ndf=self.ch, n_layers=5).to(self.device)
        self.disLB = Discriminator(input_nc=3, ndf=self.ch, n_layers=5).to(self.device)

        """ Define Loss """
        self.L1_loss = nn.L1Loss().to(self.device)
        self.MSE_loss = nn.MSELoss().to(self.device)
        self.BCE_loss = nn.BCEWithLogitsLoss().to(self.device)
        self.CE_loss = nn.CrossEntropyLoss().to(self.device)

        """ Trainer """
        # Module 안에는 _parameters, _modules라는 OrderedDict가 있다.
        # _parameters 안에는 Parameters(tensor의 자식, weight, bias etc...) 객체가 들어있다.
        # _modules 안에는 하위 module들이 들어있다.
        # Module.parameters()를 호출하면, _modules를 돌면서 자신을 포함한 모든 모듈들의 _parameters 안의 Parameters 객체를 list로 반환한다.
        # 즉, optimizer는 list of Parameters(or dict)를 받는다.

        # Freeze the generator
        self.ToggleBottleNeck(freeze=True)
        # Trainable params
        genA2B_params = []
        genB2A_params = []
        for param in self.genA2B.parameters():
            if param.requires_grad == True:
                genA2B_params.append(param)
        for param in self.genB2A.parameters():
            if param.requires_grad == True:
                genB2A_params.append(param)

        #self.G_optim = torch.optim.Adam(itertools.chain(self.genA2B.parameters(), self.genB2A.parameters()), lr=self.lr, betas=(0.5, 0.999), weight_decay=self.weight_decay)
        self.G_optim = torch.optim.Adam(itertools.chain(genA2B_params, genB2A_params), lr=self.lr,
                                        betas=(0.5, 0.999), weight_decay=self.weight_decay)

        self.D_optim = torch.optim.Adam(itertools.chain(self.disGA.parameters(), self.disGB.parameters(), self.disLA.parameters(), self.disLB.parameters()), lr=self.lr, betas=(0.5, 0.999), weight_decay=self.weight_decay)

        """ Define Rho clipper to constraint the value of rho in AdaILN and ILN"""
        self.Rho_clipper = RhoClipper(0, 1)
    def ToggleBottleNeck(self, freeze=True):#Freeze the bottleneck blocks of generator.
        for seq, block in enumerate(self.genA2B.DownBlock.children()):
            if isinstance(block, ResnetBlock):
                for param in block.parameters():
                    param.requires_grad = (not freeze)
        for seq, block in enumerate(self.genB2A.DownBlock.children()):
            if isinstance(block, ResnetBlock):
                for param in block.parameters():
                    param.requires_grad = (not freeze)
    def next_iter(self, step):#return real_A, real_B, real_A_label, real_B_label of train dataset
        group_num = len(self.group_list)
        seq = step%group_num

        try:
            real_A, real_A_label = self.iterA_list[seq].next()
        except:
            self.iterA_list[seq] = iter(self.loaderA_list[seq])
            real_A, real_A_label = self.iterA_list[seq].next()

        try:
            real_B, real_B_label = self.iterB_list[seq].next()
        except:
            self.iterB_list[seq] = iter(self.loaderB_list[seq])
            real_B, real_B_label = self.iterB_list[seq].next()
        assert type(real_A_label.item()) is int and type(real_B_label.item()) is int, 'next_iter exception'

        return real_A, real_B, real_A_label, real_B_label
    def train(self):

        self.genA2B.train(), self.genB2A.train(), self.disGA.train(), self.disGB.train(), self.disLA.train(), self.disLB.train()
        best_kid = 35
        if self.neptune:
            neptune.init(project_qualified_name='ml.swlee/face2webtoon',
                        api_token = '')
            neptune.create_experiment(params=self.params)

        start_iter = 1
        if self.resume:
            model_list = glob(os.path.join(self.result_dir, self.dataset, 'model', '*.pt'))
            if not len(model_list) == 0:
                model_list.sort()
                start_iter = int(model_list[-1].split('_')[-1].split('.')[0])
                self.load(os.path.join(self.result_dir, self.dataset, 'model'), start_iter)
                print(" [*] Load SUCCESS")
                if self.decay_flag and start_iter > (self.iteration // 2):
                    self.G_optim.param_groups[0]['lr'] -= (self.lr / (self.iteration // 2)) * (start_iter - self.iteration // 2)
                    self.D_optim.param_groups[0]['lr'] -= (self.lr / (self.iteration // 2)) * (start_iter - self.iteration // 2)

        self.start_iter = start_iter
        self.logit_list_fake = []
        self.logit_list_real = []

        # training loop
        print('training start !')
        start_time = time.time()
        for step in range(start_iter, self.iteration + 1):
            self.current_step = step
            t = time.time()
            #총 iteration의 절반까지는 decay 안하고, 이후부터 적용한다
            if self.decay_flag and step > (self.iteration // 2):
            # param_groups은 list이고, 원소로 dict를 가진다.
            # 각 dict는 각 module에 해당한다. dict['params']의 value는 learnable parameter의 list이다.
            # dict['lr'], dict['beta']등 한 모듈에 해당하는 dict에는 많은 파라미터가 있고
            # param_groups는 그런 dict의 list이다.
            # 여기서는 param_groups에 각 원소(dict)에 접근해서, 'lr'이라는 key에 해당하는 value값에 접근하고 있다.
                self.G_optim.param_groups[0]['lr'] -= (self.lr / (self.iteration // 2))
                self.D_optim.param_groups[0]['lr'] -= (self.lr / (self.iteration // 2))

            """
            try:
                #첫 루프라 trainA_iter 없으면 밑으로 가서 선언
                if step % 4 == 0:
                    real_A, real_A_label = trainA_0_iter.next()
                elif step % 4 == 1:
                    real_A, real_A_label = trainA_1_iter.next()
                elif step % 4 == 2:
                    real_A, real_A_label = trainA_2_iter.next()
                elif step % 4 == 3:
                    real_A, real_A_label = trainA_3_iter.next()



            except:
                #Dataloader의 iterator를 가져오고 원소 하나를 real_A에 줌. 마치 실시간 for 문.
                #Shuffle을 매 epoch마다 해야하는데, epoch을 안쓰면 어떡하냐?
                #Iterator가 끝나면, 위에서 예외가 나면서 여기로 올거임.
                #그 때 여기서 iter를 다시 생성하면 다시 shuffle됨.
                #어차피 for문도 한 바퀴 돌고 iter 다시 생성하고의 반복임. 결국 같음.
                trainA_0_iter = iter(self.trainA_0_loader)
                trainA_1_iter = iter(self.trainA_1_loader)
                trainA_2_iter = iter(self.trainA_2_loader)
                trainA_3_iter = iter(self.trainA_3_loader)

                real_A, real_A_label = trainA_0_iter.next()

            try:
                if step % 4== 0:
                    real_B, real_B_label = trainB_0_iter.next()
                elif step % 4 == 1:
                    real_B, real_B_label = trainB_1_iter.next()
                elif step % 4 == 2:
                    real_B, real_B_label = trainB_2_iter.next()
                elif step % 4 == 3:
                    real_B, real_B_label = trainB_3_iter.next()


            except:
                trainB_0_iter = iter(self.trainB_0_loader)
                trainB_1_iter = iter(self.trainB_1_loader)
                trainB_2_iter = iter(self.trainB_2_loader)
                trainB_3_iter = iter(self.trainB_3_loader)

                real_B, real_B_label = trainB_0_iter.next()
            """

            # Selective backpropagation
            """
            if real_B_label == 0:
                self.ToggleBottleNeck(freeze=False)
            else:
                self.ToggleBottleNeck(freeze=True)
            """

            real_A, real_B, real_A_label, real_B_label = self.next_iter(step)
            print(f"curr label : {real_A_label}, {real_B_label}")

            real_A, real_B = real_A.to(self.device), real_B.to(self.device)
            real_A_label, real_B_label = real_A_label.to(self.device), real_B_label.to(self.device)


            # Update D
            self.D_optim.zero_grad()

            fake_A2B, _, _ = self.genA2B(real_A)
            fake_B2A, _, _ = self.genB2A(real_B)


            if self.useADA:
                #ADA
                real_A_aug = torch.unsqueeze(self.ada2.augment_pipeline(torch.squeeze(real_A, 0)), 0)
                real_B_aug = torch.unsqueeze(self.ada.augment_pipeline(torch.squeeze(real_B, 0)), 0)

                fake_B2A_aug = torch.unsqueeze(self.ada2.augment_pipeline(torch.squeeze(fake_B2A, 0)), 0)
                fake_A2B_aug = torch.unsqueeze(self.ada.augment_pipeline(torch.squeeze(fake_A2B, 0)), 0)

                real_GA_logit, real_GA_cam_logit, _, __ = self.disGA(real_A_aug)
                real_LA_logit, real_LA_cam_logit, _, __ = self.disLA(real_A_aug)
                real_GB_logit, real_GB_cam_logit, _, real_GB_group_logit = self.disGB(real_B_aug)
                real_LB_logit, real_LB_cam_logit, _, real_LB_group_logit = self.disLB(real_B_aug)


                self.ada.feed(torch.squeeze(real_GB_logit,0))
                self.ada2.feed(torch.squeeze(real_GA_logit,0))
                fake_GA_logit, fake_GA_cam_logit, _, __ = self.disGA(fake_B2A_aug)
                fake_LA_logit, fake_LA_cam_logit, _, __ = self.disLA(fake_B2A_aug)
                fake_GB_logit, fake_GB_cam_logit, _, fake_GB_group_logit = self.disGB(fake_A2B_aug)
                fake_LB_logit, fake_LB_cam_logit, _, fake_LB_group_logit = self.disLB(fake_A2B_aug)
            else:
                real_GA_logit, real_GA_cam_logit, _, __ = self.disGA(real_A)
                real_LA_logit, real_LA_cam_logit, _, __ = self.disLA(real_A)
                real_GB_logit, real_GB_cam_logit, _, real_GB_group_logit = self.disGB(real_B)
                real_LB_logit, real_LB_cam_logit, _, real_LB_group_logit = self.disLB(real_B)


                fake_GA_logit, fake_GA_cam_logit, _, __ = self.disGA(fake_B2A)
                fake_LA_logit, fake_LA_cam_logit, _, __ = self.disLA(fake_B2A)
                fake_GB_logit, fake_GB_cam_logit, _, fake_GB_group_logit = self.disGB(fake_A2B)
                fake_LB_logit, fake_LB_cam_logit, _, fake_LB_group_logit = self.disLB(fake_A2B)


            # Save the logits to plot after training finishes.






            D_ad_loss_GA = self.MSE_loss(real_GA_logit, torch.ones_like(real_GA_logit).to(self.device)) + self.MSE_loss(fake_GA_logit, torch.zeros_like(fake_GA_logit).to(self.device))
            D_ad_cam_loss_GA = self.MSE_loss(real_GA_cam_logit, torch.ones_like(real_GA_cam_logit).to(self.device)) + self.MSE_loss(fake_GA_cam_logit, torch.zeros_like(fake_GA_cam_logit).to(self.device))
            D_ad_loss_LA = self.MSE_loss(real_LA_logit, torch.ones_like(real_LA_logit).to(self.device)) + self.MSE_loss(fake_LA_logit, torch.zeros_like(fake_LA_logit).to(self.device))
            D_ad_cam_loss_LA = self.MSE_loss(real_LA_cam_logit, torch.ones_like(real_LA_cam_logit).to(self.device)) + self.MSE_loss(fake_LA_cam_logit, torch.zeros_like(fake_LA_cam_logit).to(self.device))

            D_ad_loss_GB = self.MSE_loss(real_GB_logit, torch.ones_like(real_GB_logit).to(self.device)) + self.MSE_loss(fake_GB_logit, torch.zeros_like(fake_GB_logit).to(self.device))
            D_ad_cam_loss_GB = self.MSE_loss(real_GB_cam_logit, torch.ones_like(real_GB_cam_logit).to(self.device)) + self.MSE_loss(fake_GB_cam_logit, torch.zeros_like(fake_GB_cam_logit).to(self.device))
            D_group_loss_GB = self.CE_loss(real_GB_group_logit, real_B_label)
            D_ad_loss_LB = self.MSE_loss(real_LB_logit, torch.ones_like(real_LB_logit).to(self.device)) + self.MSE_loss(fake_LB_logit, torch.zeros_like(fake_LB_logit).to(self.device))
            D_ad_cam_loss_LB = self.MSE_loss(real_LB_cam_logit, torch.ones_like(real_LB_cam_logit).to(self.device)) + self.MSE_loss(fake_LB_cam_logit, torch.zeros_like(fake_LB_cam_logit).to(self.device))
            D_group_loss_LB = self.CE_loss(real_LB_group_logit, real_B_label)


            self.group_loss_D_Global.append(D_group_loss_GB.cpu().detach().numpy())
            self.group_loss_D_Local.append(D_group_loss_LB.cpu().detach().numpy())
            if self.use_grouploss == False:
                D_group_loss_LB = torch.zeros_like(D_group_loss_LB)
                D_group_loss_GB = torch.zeros_like(D_group_loss_GB)
            D_loss_A = self.adv_weight * (D_ad_loss_GA + D_ad_cam_loss_GA + D_ad_loss_LA + D_ad_cam_loss_LA)
            D_loss_B = self.adv_weight * (D_ad_loss_GB + D_ad_cam_loss_GB + D_ad_loss_LB + D_ad_cam_loss_LB + D_group_loss_GB + D_group_loss_LB)



            Discriminator_loss = D_loss_A + D_loss_B


            Discriminator_loss.backward()
            self.D_optim.step()


            # Update G
            self.G_optim.zero_grad()

            t_tmp = time.time()
            fake_A2B, fake_A2B_cam_logit, _ = self.genA2B(real_A)
            fake_B2A, fake_B2A_cam_logit, _ = self.genB2A(real_B)

            fake_A2B2A, _, _ = self.genB2A(fake_A2B)
            fake_B2A2B, _, _ = self.genA2B(fake_B2A)

            fake_A2A, fake_A2A_cam_logit, _ = self.genB2A(real_A)
            fake_B2B, fake_B2B_cam_logit, _ = self.genA2B(real_B)

            if self.useADA:
                #ADA
                fake_A2B_aug = torch.unsqueeze(self.ada.augment_pipeline(torch.squeeze(fake_A2B, 0)), 0)
                fake_B2A_aug = torch.unsqueeze(self.ada2.augment_pipeline(torch.squeeze(fake_B2A, 0)), 0)

                fake_GA_logit, fake_GA_cam_logit, _, __ = self.disGA(fake_B2A_aug)
                fake_LA_logit, fake_LA_cam_logit, _, __ = self.disLA(fake_B2A_aug)
                fake_GB_logit, fake_GB_cam_logit, _, fake_GB_group_logit = self.disGB(fake_A2B_aug)
                fake_LB_logit, fake_LB_cam_logit, _, fake_LB_group_logit = self.disLB(fake_A2B_aug)

            else:
                fake_GA_logit, fake_GA_cam_logit, _, __ = self.disGA(fake_B2A)
                fake_LA_logit, fake_LA_cam_logit, _, __ = self.disLA(fake_B2A)
                fake_GB_logit, fake_GB_cam_logit, _, fake_GB_group_logit = self.disGB(fake_A2B)
                fake_LB_logit, fake_LB_cam_logit, _, fake_LB_group_logit = self.disLB(fake_A2B)

            G_ad_loss_GA = self.MSE_loss(fake_GA_logit, torch.ones_like(fake_GA_logit).to(self.device))
            G_ad_cam_loss_GA = self.MSE_loss(fake_GA_cam_logit, torch.ones_like(fake_GA_cam_logit).to(self.device))
            G_ad_loss_LA = self.MSE_loss(fake_LA_logit, torch.ones_like(fake_LA_logit).to(self.device))
            G_ad_cam_loss_LA = self.MSE_loss(fake_LA_cam_logit, torch.ones_like(fake_LA_cam_logit).to(self.device))
            G_ad_loss_GB = self.MSE_loss(fake_GB_logit, torch.ones_like(fake_GB_logit).to(self.device))
            G_ad_cam_loss_GB = self.MSE_loss(fake_GB_cam_logit, torch.ones_like(fake_GB_cam_logit).to(self.device))
            G_ad_loss_LB = self.MSE_loss(fake_LB_logit, torch.ones_like(fake_LB_logit).to(self.device))
            G_ad_cam_loss_LB = self.MSE_loss(fake_LB_cam_logit, torch.ones_like(fake_LB_cam_logit).to(self.device))

            G_recon_loss_A = self.L1_loss(fake_A2B2A, real_A)
            G_recon_loss_B = self.L1_loss(fake_B2A2B, real_B)

            G_identity_loss_A = self.L1_loss(fake_A2A, real_A)
            G_identity_loss_B = self.L1_loss(fake_B2B, real_B)


            G_cam_loss_A = self.BCE_loss(fake_B2A_cam_logit, torch.ones_like(fake_B2A_cam_logit).to(self.device)) + self.BCE_loss(fake_A2A_cam_logit, torch.zeros_like(fake_A2A_cam_logit).to(self.device))
            G_cam_loss_B = self.BCE_loss(fake_A2B_cam_logit, torch.ones_like(fake_A2B_cam_logit).to(self.device)) + self.BCE_loss(fake_B2B_cam_logit, torch.zeros_like(fake_B2B_cam_logit).to(self.device))


            G_group_loss_GB = self.CE_loss(fake_GB_group_logit, real_A_label)
            G_group_loss_LB = self.CE_loss(fake_LB_group_logit, real_A_label)
            if self.use_grouploss == False:
                G_group_loss_LB = torch.zeros_like(G_group_loss_LB)
                G_group_loss_GB = torch.zeros_like(G_group_loss_GB)

            G_loss_A =  self.adv_weight * (G_ad_loss_GA + G_ad_cam_loss_GA + G_ad_loss_LA + G_ad_cam_loss_LA) + self.cycle_weight * G_recon_loss_A + self.identity_weight * G_identity_loss_A + self.cam_weight * G_cam_loss_A
            G_loss_B = self.adv_weight * (G_ad_loss_GB + G_ad_cam_loss_GB + G_ad_loss_LB + G_ad_cam_loss_LB) + self.cycle_weight * G_recon_loss_B + self.identity_weight * G_identity_loss_B + self.cam_weight * G_cam_loss_B + self.group_weight*(G_group_loss_GB + G_group_loss_LB)

            print("==time_Gforward : ", time.time() - t_tmp)
            t_tmp = time.time()
            Generator_loss = G_loss_A + G_loss_B
            Generator_loss.backward()
            self.G_optim.step()

            # clip parameter of AdaILN and ILN, applied after optimizer step
            self.genA2B.apply(self.Rho_clipper)
            self.genB2A.apply(self.Rho_clipper)

            print("==time_Gbackward : ", time.time() - t_tmp)

            #Neptune
            if self.neptune:
                t_tmp = time.time()
                neptune.log_metric('Generator global group loss', G_group_loss_GB)
                neptune.log_metric('Generator local group loss', G_group_loss_LB)
                neptune.log_metric('GeneratorA2B ad loss', G_ad_loss_GB)
                neptune.log_metric('GeneratorB2A ad loss', G_ad_loss_GA)
                neptune.log_metric('Generator loss', Generator_loss)
                neptune.log_metric('Generator cycle A2B2A loss', G_recon_loss_A)
                neptune.log_metric('Generator cycle B2A2B loss', G_recon_loss_B)

                neptune.log_metric('Generator cam loss B', G_cam_loss_B)
                neptune.log_metric('Generator cam loss A', G_cam_loss_A)

                neptune.log_metric('Discriminator_B global real output', torch.mean(real_GB_logit))
                neptune.log_metric('Discriminator_B local real output', torch.mean(real_LB_logit))
                neptune.log_metric('Discriminator_B global fake output', torch.mean(fake_GB_logit))
                neptune.log_metric('Discriminator_B local fake output', torch.mean(fake_LB_logit))
                neptune.log_metric('Discriminator_B global ad loss', D_ad_loss_GB)
                neptune.log_metric('Discriminator_B global cam loss', D_ad_cam_loss_GB)
                neptune.log_metric('Discriminator_B local ad loss', D_ad_loss_LB)
                neptune.log_metric('Discriminator_B local cam loss', D_ad_cam_loss_LB)

                neptune.log_metric('Discriminator_A global ad loss', D_ad_loss_GA)
                neptune.log_metric('Discriminator_A global cam loss', D_ad_cam_loss_GA)
                neptune.log_metric('Discriminator_A local ad loss', D_ad_loss_LA)
                neptune.log_metric('Discriminator_A local cam loss', D_ad_cam_loss_LA)

                neptune.log_metric('Discriminator global group loss', D_group_loss_GB)
                neptune.log_metric('Discriminator local group loss', D_group_loss_LB)

                neptune.log_metric('Discriminator total loss', Discriminator_loss)

                neptune.log_metric('Augmentation strength_DB', self.ada.strength)
                neptune.log_metric('Augmentation strength_DA', self.ada2.strength)

                neptune.log_metric('rt_DB', self.ada.rt)
                neptune.log_metric('rt_DA', self.ada2.rt)
                print("==neptune time : ", time.time()-t_tmp)

            print("[%5d/%5d] time: %4.4f d_loss: %.8f, g_loss: %.8f" % (step, self.iteration, time.time() - start_time, Discriminator_loss, Generator_loss))

            if step % self.print_freq == 0:
                train_sample_num = 5
                test_sample_num = 5
                A2B = np.zeros((self.img_size * 7, 0, 3))
                B2A = np.zeros((self.img_size * 7, 0, 3))

                self.genA2B.eval(), self.genB2A.eval(), self.disGA.eval(), self.disGB.eval(), self.disLA.eval(), self.disLB.eval()
                for print_step in range(train_sample_num):

                    real_A, real_B, real_A_label, real_B_label = self.next_iter(print_step)
                    real_A, real_B = real_A.to(self.device), real_B.to(self.device)

                    fake_A2B, _, fake_A2B_heatmap = self.genA2B(real_A)
                    fake_B2A, _, fake_B2A_heatmap = self.genB2A(real_B)

                    fake_A2B2A, _, fake_A2B2A_heatmap = self.genB2A(fake_A2B)
                    fake_B2A2B, _, fake_B2A2B_heatmap = self.genA2B(fake_B2A)

                    fake_A2A, _, fake_A2A_heatmap = self.genB2A(real_A)
                    fake_B2B, _, fake_B2B_heatmap = self.genA2B(real_B)

                    A2B = np.concatenate((A2B, np.concatenate((RGB2BGR(tensor2numpy(denorm(real_A[0]))),
                                                               cam(tensor2numpy(fake_A2A_heatmap[0]), self.img_size),
                                                               RGB2BGR(tensor2numpy(denorm(fake_A2A[0]))),
                                                               cam(tensor2numpy(fake_A2B_heatmap[0]), self.img_size),
                                                               RGB2BGR(tensor2numpy(denorm(fake_A2B[0]))),
                                                               cam(tensor2numpy(fake_A2B2A_heatmap[0]), self.img_size),
                                                               RGB2BGR(tensor2numpy(denorm(fake_A2B2A[0])))), 0)), 1)

                    B2A = np.concatenate((B2A, np.concatenate((RGB2BGR(tensor2numpy(denorm(real_B[0]))),
                                                               cam(tensor2numpy(fake_B2B_heatmap[0]), self.img_size),
                                                               RGB2BGR(tensor2numpy(denorm(fake_B2B[0]))),
                                                               cam(tensor2numpy(fake_B2A_heatmap[0]), self.img_size),
                                                               RGB2BGR(tensor2numpy(denorm(fake_B2A[0]))),
                                                               cam(tensor2numpy(fake_B2A2B_heatmap[0]), self.img_size),
                                                               RGB2BGR(tensor2numpy(denorm(fake_B2A2B[0])))), 0)), 1)

                for _ in range(test_sample_num):
                    try:
                        real_A, _ = testA_iter.next()
                    except:
                        testA_iter = iter(self.testA_loader)
                        real_A, _ = testA_iter.next()

                    try:
                        real_B, _ = testB_iter.next()
                    except:
                        testB_iter = iter(self.testB_loader)
                        real_B, _ = testB_iter.next()
                    real_A, real_B = real_A.to(self.device), real_B.to(self.device)

                    fake_A2B, _, fake_A2B_heatmap = self.genA2B(real_A)
                    fake_B2A, _, fake_B2A_heatmap = self.genB2A(real_B)

                    fake_A2B2A, _, fake_A2B2A_heatmap = self.genB2A(fake_A2B)
                    fake_B2A2B, _, fake_B2A2B_heatmap = self.genA2B(fake_B2A)

                    fake_A2A, _, fake_A2A_heatmap = self.genB2A(real_A)

                    fake_B2B, _, fake_B2B_heatmap = self.genA2B(real_B)

                    A2B = np.concatenate((A2B, np.concatenate((RGB2BGR(tensor2numpy(denorm(real_A[0]))),
                                                               cam(tensor2numpy(fake_A2A_heatmap[0]), self.img_size),
                                                               RGB2BGR(tensor2numpy(denorm(fake_A2A[0]))),
                                                               cam(tensor2numpy(fake_A2B_heatmap[0]), self.img_size),
                                                               RGB2BGR(tensor2numpy(denorm(fake_A2B[0]))),
                                                               cam(tensor2numpy(fake_A2B2A_heatmap[0]), self.img_size),
                                                               RGB2BGR(tensor2numpy(denorm(fake_A2B2A[0])))), 0)), 1)

                    B2A = np.concatenate((B2A, np.concatenate((RGB2BGR(tensor2numpy(denorm(real_B[0]))),
                                                               cam(tensor2numpy(fake_B2B_heatmap[0]), self.img_size),
                                                               RGB2BGR(tensor2numpy(denorm(fake_B2B[0]))),
                                                               cam(tensor2numpy(fake_B2A_heatmap[0]), self.img_size),
                                                               RGB2BGR(tensor2numpy(denorm(fake_B2A[0]))),
                                                               cam(tensor2numpy(fake_B2A2B_heatmap[0]), self.img_size),
                                                               RGB2BGR(tensor2numpy(denorm(fake_B2A2B[0])))), 0)), 1)

                cv2.imwrite(os.path.join(self.result_dir, self.dataset, 'img', 'A2B_%07d.png' % step), A2B * 255.0)
                cv2.imwrite(os.path.join(self.result_dir, self.dataset, 'img', 'B2A_%07d.png' % step), B2A * 255.0)
                self.genA2B.train(), self.genB2A.train(), self.disGA.train(), self.disGB.train(), self.disLA.train(), self.disLB.train()


            if step % self.save_freq == 0 and start_iter != step:
                self.save(os.path.join(self.result_dir, self.dataset, 'model'), step)

            if step % self.metric_freq == 0 and step % self.print_freq != 0:
                result = self.kid()
                for p, m, s, act_true in result:
                    if not isinstance(self.act_true, np.ndarray):
                        self.act_true = act_true.copy()
                    print('KID (%s): %.3f (%.3f)' % (p, m, s))

                if self.neptune : neptune.log_metric('KID*1000', m*1000)
                self.kid_list.append(m*1000)
                if m*1000<best_kid:
                    print("--------------------best_kid saving --------------", m*1000)
                    best_kid = m*1000
                    params = {}
                    params['genA2B'] = self.genA2B.state_dict()
                    params['genB2A'] = self.genB2A.state_dict()
                    params['disGA'] = self.disGA.state_dict()
                    params['disGB'] = self.disGB.state_dict()
                    params['disLA'] = self.disLA.state_dict()
                    params['disLB'] = self.disLB.state_dict()
                    torch.save(params, os.path.join(self.result_dir, self.dataset + f'{step}_{m*1000}.pt'))


            if step % 2500 == 0:
                #Model saving
                params = {}
                params['genA2B'] = self.genA2B.state_dict()
                params['genB2A'] = self.genB2A.state_dict()
                params['disGA'] = self.disGA.state_dict()
                params['disGB'] = self.disGB.state_dict()
                params['disLA'] = self.disLA.state_dict()
                params['disLB'] = self.disLB.state_dict()
                torch.save(params, os.path.join(self.result_dir, self.dataset + '_params_latest.pt'))

            print("==time_endofloop : ", time.time() - t)
            if self.neptune : neptune.log_metric("Sec per loop", time.time()-t)

    def kid(self):

        self.test(show_real = False, load_ckp = False, same_path=True)
        assert os.path.isdir(os.path.join(self.result_dir, self.dataset, 'test', 'kid')) == True, "Test path doesn't exist."
        from gan_metrics_pytorch import kid_score
        real_path = r'C:\ML\face2webtoon\UGATIT-pytorch\dataset\video2anime\trainB'
        fake_path = os.path.join(self.result_dir, self.dataset, 'test', 'kid')
        batch_size = 50
        dims = 2048

        os.environ['CUDA_VISIBLE_DEVICES'] = '0'
        print("Calculating KID")
        result = kid_score.calculate_kid_given_paths([real_path, fake_path], batch_size, True, dims, 'inception', self.img_size, act_true=self.act_true)
        return result
    def save(self, dir, step):
        params = {}
        params['genA2B'] = self.genA2B.state_dict()
        params['genB2A'] = self.genB2A.state_dict()
        params['disGA'] = self.disGA.state_dict()
        params['disGB'] = self.disGB.state_dict()
        params['disLA'] = self.disLA.state_dict()
        params['disLB'] = self.disLB.state_dict()
        torch.save(params, os.path.join(dir, self.dataset + '_params_%07d.pt' % step))
        try:
            None
            #neptune.log_artifact(os.path.join(dir, self.dataset + '_params_%07d.pt' % step))
        except:
            None


    def load(self, dir, step):
        params = torch.load(os.path.join(dir, self.dataset + '_params_%07d.pt' % step))
        self.genA2B.load_state_dict(params['genA2B'])
        self.genB2A.load_state_dict(params['genB2A'])
        self.disGA.load_state_dict(params['disGA'], strict=False)
        self.disGB.load_state_dict(params['disGB'], strict=False)
        self.disLA.load_state_dict(params['disLA'], strict=False)
        self.disLB.load_state_dict(params['disLB'], strict=False)

    def test(self, show_real = True, load_ckp = True, same_path = False):
        if load_ckp:
            model_list = glob(os.path.join(self.result_dir, self.dataset, 'model', '*.pt'))#디렉토리에 *.pt와 패턴 매칭되는 모든 파일
            if not len(model_list) == 0:
                model_list.sort()
                iter = int(model_list[-1].split('_')[-1].split('.')[0])
                self.load(os.path.join(self.result_dir, self.dataset, 'model'), iter)
                print(iter, " [*] Load SUCCESS")
            else:
                print(" [*] Load FAILURE")
                return
        else:
            iter = self.current_step
        self.genA2B.eval(), self.genB2A.eval()
        for n, (real_A, _) in enumerate(self.testA_loader):
            real_A = real_A.to(self.device)

            fake_A2B, _, fake_A2B_heatmap = self.genA2B(real_A)

            fake_A2B2A, _, fake_A2B2A_heatmap = self.genB2A(fake_A2B)

            fake_A2A, _, fake_A2A_heatmap = self.genB2A(real_A)

            A2B = np.concatenate((RGB2BGR(tensor2numpy(denorm(real_A[0]))),
                                  cam(tensor2numpy(fake_A2A_heatmap[0]), self.img_size),
                                  RGB2BGR(tensor2numpy(denorm(fake_A2A[0]))),
                                  cam(tensor2numpy(fake_A2B_heatmap[0]), self.img_size),
                                  RGB2BGR(tensor2numpy(denorm(fake_A2B[0]))),
                                  cam(tensor2numpy(fake_A2B2A_heatmap[0]), self.img_size),
                                  RGB2BGR(tensor2numpy(denorm(fake_A2B2A[0])))), 0)
            #real, fake_A2B만 출력하게 변경
            if show_real:
                result = np.concatenate(((RGB2BGR(tensor2numpy(denorm(real_A[0]))),RGB2BGR(tensor2numpy(denorm(fake_A2B[0]))))),0)
            #fake_A2B만 출력하게 변경
            else:
                result = RGB2BGR(tensor2numpy(denorm(fake_A2B[0])))

            if same_path==False:
                check_folder(os.path.join(self.result_dir, self.dataset, 'test', str(iter)))
                cv2.imwrite(os.path.join(self.result_dir, self.dataset, 'test', str(iter), 'A2B_%d.png' % (n + 1)), result * 255.0)
            else:
                check_folder(os.path.join(self.result_dir, self.dataset, 'test', 'kid'))
                cv2.imwrite(os.path.join(self.result_dir, self.dataset, 'test', 'kid', 'A2B_%d.png' % (n + 1)), result * 255.0)
        for n, (real_B, _) in enumerate(self.testB_loader):
            real_B = real_B.to(self.device)

            fake_B2A, _, fake_B2A_heatmap = self.genB2A(real_B)

            fake_B2A2B, _, fake_B2A2B_heatmap = self.genA2B(fake_B2A)

            fake_B2B, _, fake_B2B_heatmap = self.genA2B(real_B)

            B2A = np.concatenate((RGB2BGR(tensor2numpy(denorm(real_B[0]))),
                                  cam(tensor2numpy(fake_B2B_heatmap[0]), self.img_size),
                                  RGB2BGR(tensor2numpy(denorm(fake_B2B[0]))),
                                  cam(tensor2numpy(fake_B2A_heatmap[0]), self.img_size),
                                  RGB2BGR(tensor2numpy(denorm(fake_B2A[0]))),
                                  cam(tensor2numpy(fake_B2A2B_heatmap[0]), self.img_size),
                                  RGB2BGR(tensor2numpy(denorm(fake_B2A2B[0])))), 0)
            #real, fake_B2A만 출력하게 변경
            result = np.concatenate(((RGB2BGR(tensor2numpy(denorm(real_B[0]))),RGB2BGR(tensor2numpy(denorm(fake_B2A[0]))))),0)
            cv2.imwrite(os.path.join(self.result_dir, self.dataset, 'test', 'B2A_%d.png' % (n + 1)), result * 255.0)
        self.genA2B.train(), self.genB2A.train()
