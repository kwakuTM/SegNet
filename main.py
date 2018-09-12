import argparse
import torch
import torch.nn.init
import torch.nn as nn
import torch.nn.functional as F 
import torch.optim as optim
import torchvision.datasets as dset
import torch.backends.cudnn as cudnn 
import torchvision.models as models

from tensorboard_logger import configure, log_value

from convNet import *
from dataLoader import *
from tqdm import tqdm

import cv2
import pdb

# Training settings
parser = argparse.ArgumentParser(description='dida Roof detection')
# Model options
parser.add_argument('--dataroot', type=str, default='/home/kwakutm/Documents/hobby/roofDetection/data',
                    help='path to dataset')
parser.add_argument('--train-data', type=str, default='/train_data/',
                    help='input training data')
parser.add_argument('--label-data', type=str, default='/labelsX/',
                    help='input label data')
parser.add_argument('--test-data', type=str, default='/test_data/553.png',
                    help='input label data')
parser.add_argument('--log-dir', default='./logsDida',
                    help='folder to output model checkpoints')
parser.add_argument('--imageSize', type=int, default=256,
                    help='the height / width of the input image to network')
parser.add_argument('--resume', default='logsDida/stateModel/', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--epochs', type=int, default=10, metavar='E',
                    help='number of epochs to train (default: 10)')

# Training options
parser.add_argument('--batch-size', type=int, default=5, metavar='BS',
                    help='input batch size for training (default: 128)')
parser.add_argument('--test-batch-size', type=int, default=5, metavar='BST',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--margin', type=float, default=2.0, metavar='MARGIN',
                    help='the margin value for the loss function (default: 2.0')
parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                    help='learning rate (default: 0.1)')
parser.add_argument('--lr-decay', default=1e-6, type=float, metavar='LRD',
                    help='learning rate decay ratio (default: 1e-6')
parser.add_argument('--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--optimizer', default='adam', type=str,
                    metavar='OPT', help='The optimizer to use (default: SGD)')
# Device options
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--gpu-id', default='0', type=str,
                    help='id(s) for CUDA_VISIBLE_DEVICES')
parser.add_argument('--seed', type=int, default=0, metavar='S',
                    help='random seed (default: 0)')
parser.add_argument('--log-interval', type=int, default=1, metavar='LI',
                    help='how many batches to wait before logging training status')

#passing all arguments 
args = parser.parse_args()
# set the device to use by setting CUDA_VISIBLE_DEVICES env variable in
# order to prevent any memory allocation on unused GPUs
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id

# Using cuda
args.cuda = not args.no_cuda and torch.cuda.is_available()
torch.manual_seed(args.seed)
np.random.seed(args.seed)
cv2.setRNGSeed(args.seed)

# Directory for models
if not os.path.exists(args.log_dir):
    os.makedirs(args.log_dir)

if args.cuda:
    cudnn.benchmark = True
    torch.cuda.manual_seed_all(args.seed)

LOG_DIR = args.log_dir + '/stateModel'

# logger for models data
class Logger(object):
    def __init__(self, log_dir):
        # clean previous logged data under the same directory name
        self._remove(log_dir)

        # configure the project
        configure(log_dir)

        self.global_step = 0

    def log_value(self, name, value):
        log_value(name, value, self.global_step)
        return self

    def step(self):
        self.global_step += 1

    @staticmethod
    def _remove(path):
        """ param <path> could either be relative or absolute. """
        if os.path.isfile(path):
            os.remove(path)  # remove the file
        elif os.path.isdir(path):
            import shutil
            shutil.rmtree(path)  # remove dir and all contains

# create logger
logger = Logger(LOG_DIR)

# Encoder-Decoder Network
class EDNet(nn.Module):
    """autoencoder definition
    """
    def __init__(self, n_classes=1, in_channels=3, is_unpooling=True):
        super(EDNet, self).__init__()
        
        self.in_channels = in_channels
        self.is_unpooling = is_unpooling

        self.enc1 = encoder1(self.in_channels, 64)
        self.enc2 = encoder1(64, 128)
        self.enc3 = encoder2(128, 256)
        self.enc4 = encoder2(256, 512)
        self.enc5 = encoder2(512, 512)

        self.dec5 = decoder2(512, 512)
        self.dec4 = decoder2(512, 256)
        self.dec3 = decoder2(256, 128)
        self.dec2 = decoder1(128, 64)
        self.dec1 = decoder1(64, n_classes)


    def forward(self, x):

        enc1, ind_1, unpool_sh1 = self.enc1(x)
        enc2, ind_2, unpool_sh2 = self.enc2(enc1)
        enc3, ind_3, unpool_sh3 = self.enc3(enc2)
        enc4, ind_4, unpool_sh4 = self.enc4(enc3)
        enc5, ind_5, unpool_sh5 = self.enc5(enc4)

        dec5 = self.dec5(enc5, ind_5, unpool_sh5)
        dec4 = self.dec4(dec5, ind_4, unpool_sh4)
        dec3 = self.dec3(dec4, ind_3, unpool_sh3)
        dec2 = self.dec2(dec3, ind_2, unpool_sh2)
        dec1 = self.dec1(dec2, ind_1, unpool_sh1)

        return dec1

    def vgg16_init(self, vgg16):
        covNets = [self.enc1, self.enc2, self.enc3, self.enc4, self.enc5]

        vgg16Features = list(vgg16.features.children())

        vgg16Layers = []
        for _layer in vgg16Features:
            if isinstance(_layer, nn.Conv2d):
                vgg16Layers.append(_layer)

        outLayers = []
        for idx, conv in enumerate(covNets):
            if idx < 2:
                layers = [conv.conv1.cbr_unit, conv.conv2.cbr_unit]
            else:
                layers = [
                    conv.conv1.cbr_unit,
                    conv.conv2.cbr_unit,
                    conv.conv3.cbr_unit
                ]
            for lay in layers:
                for _layer in lay:
                    if isinstance(_layer, nn.Conv2d):
                        outLayers.append(_layer)

        assert len(vgg16Layers) == len(outLayers)

        for xx, yy in zip(vgg16Layers, outLayers):
            if isinstance(xx, nn.Conv2d) and isinstance(yy, nn.Conv2d):
                assert xx.weight.size() == yy.weight.size()
                assert xx.bias.size() == yy.bias.size()
                yy.weight.data = xx.weight.data
                yy.bias.data = xx.bias.data


# augmenting training data
augArr = Compose(augmentationsArray)

train_loader = Dataloader(args.dataroot, input_data=args.train_data, labels=args.label_data, is_transform=True, augmentations=augArr)
test_loader = Dataloader(args.dataroot, input_data=args.train_data, labels=args.label_data, is_transform=True)


def main():

    # print the experiment configuration
    print('\nparsed options:\n{}\n'.format(vars(args)))

    model = EDNet()

    # initialize model with weights of VGG-net
    vgg16 = models.vgg16(pretrained=True)
    model.vgg16_init(vgg16)

    if args.cuda:
        model.cuda()

    optimizer = create_optimizer(model, args.lr)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume): 
            print('=> loading checkpoint {}'.format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            checkpoint = torch.load(args.resume)
            model.load_state_dict(checkpoint['state_dict'])
        else:
            print('=> no checkpoint found at {}'.format(args.resume))

    start = args.start_epoch
    end = start + args.epochs
    for epoch in range(start, end):
        train(train_loader, model, optimizer, epoch)
        test(test_loader, model, epoch)


def train(train_loader, model, optimizer, epoch):
    #switch to train mode
    model.train
    pbar = tqdm(range(len(train_loader)))
    for i in pbar:
        data, labels = train_loader[i]

        if args.cuda:
            data, labels = data.cuda(), labels.cuda()
        data, labels = Variable(data, requires_grad=True), Variable(labels, requires_grad=False)

        optimizer.zero_grad()

        #compute output
        out = model(data)
        
        # dice loss used
        loss_fn = dice_loss
        loss = loss_fn(input=out, target=labels)

        loss.backward()
        optimizer.step()

        # update the optimizer learning rate
        adjust_learning_rate(optimizer)

        # log loss value
        logger.log_value('loss', loss.data[0]).step()

        if i % args.log_interval == 0:
            pbar.set_description(
                'Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, i * len(data), len(train_loader),
                    100. * i / len(train_loader),
                    loss.data[0]))

    # do checkpointing
    torch.save({'epoch': epoch + 1, 'state_dict': model.state_dict()},
               '{}/checkpoint_{}.pth'.format(LOG_DIR, epoch))


def test(test_loader, model, epoch):
    #sitch to evauate mode
    model.eval()
    #setup image
    print("Read Input Image from : {}".format(args.dataroot))
    
    # read in Test Image
    img = cv2.imread(args.dataroot + args.test_data)
    img = np.array(img, dtype=np.uint8)

    n_classes = test_loader.n_classes

    # resizing the image with proper dimensions
    img = m.imresize(img, (test_loader.img_size[0], test_loader.img_size[1]))
    
    #
    img = img[:, :, ::-1] # RGB -> BGR
    img = img.astype(np.float64)

    # normalization
    img -= test_loader.mean
    img = img.astype(float) / 255.0

    # NHWC -> NCHW
    img = img.transpose(2, 0, 1)
    img = np.expand_dims(img, 0)
    img = torch.from_numpy(img).float()

    # Setup Model
    state = torch.load(args.resume + "checkpoint_" + str(epoch) + ".pth")["state_dict"]
    model.load_state_dict(state)

    if args.cuda:
        img = img.cuda()

    img = Variable(img, requires_grad=False)

    # modelling test images
    outputs = model(img)

    # getting HW dims
    pred = np.squeeze(outputs.data.cpu().numpy(), axis=0)

    #decoding resulting image
    decoded = test_loader.decode_segmap(pred)

    cv2.imwrite("data/img.png", decoded)
    print("Segmentation Mask Saved ")


def adjust_learning_rate(optimizer):
    """Updates the learning rate given the learning rate decay.
    The routine has been implemented according to the original Lua SGD optimizer
    """
    for group in optimizer.param_groups:
        if 'step' not in group:
            group['step'] = 0
        group['step'] += 1

        group['lr'] = args.lr / (1 + group['step'] * args.lr_decay)


def create_optimizer(model, new_lr):
    # setup optimizer
    if args.optimizer == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=new_lr,
                              momentum=0.9, dampening=0.9,
                              weight_decay=args.wd)
    elif args.optimizer == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=new_lr,
                               weight_decay=args.wd)
    else:
        raise Exception('Not supported optimizer: {0}'.format(args.optimizer))
    return optimizer


def dice_loss(input, target):
    smooth = 1.

    input = input.view(-1)
    target = target.float().view(-1)
  
    intersection = (input * target).sum()
    
    return 1 - ((2. * intersection + smooth) /
              (input.sum() + target.sum() + smooth))


def cross_entropy2d(input, target, weight=None, size_average=True):

    diceLoss = dice_loss(input, target)

    input = input.transpose(1, 2).transpose(2, 3).contiguous().view(-1, train_loader.n_classes)
    target = target.view(-1)
    loss = F.cross_entropy(
        input, target, weight=weight, size_average=size_average, ignore_index=250
    ) + diceLoss
    return loss


if __name__ == '__main__':
    main()