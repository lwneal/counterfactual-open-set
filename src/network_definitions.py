import torch
from torch import nn
from torch.nn.functional import log_softmax


def weights_init(m):
    classname = m.__class__.__name__
    # TODO: what about fully-connected layers?
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.05)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


class encoder32(nn.Module):
    def __init__(self, latent_size=100, num_classes=2, batch_size=64, **kwargs):
        super(self.__class__, self).__init__()
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.conv1 = nn.Conv2d(3,       64,     3, 1, 1, bias=False)
        self.conv2 = nn.Conv2d(64,      64,     3, 1, 1, bias=False)
        self.conv3 = nn.Conv2d(64,     128,     3, 2, 1, bias=False)

        self.conv4 = nn.Conv2d(128,    128,     3, 1, 1, bias=False)
        self.conv5 = nn.Conv2d(128,    128,     3, 1, 1, bias=False)
        self.conv6 = nn.Conv2d(128,    128,     3, 2, 1, bias=False)

        self.conv7 = nn.Conv2d(128,    128,     3, 1, 1, bias=False)
        self.conv8 = nn.Conv2d(128,    128,     3, 1, 1, bias=False)
        self.conv9 = nn.Conv2d(128,    128,     3, 2, 1, bias=False)

        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)

        self.bn4 = nn.BatchNorm2d(128)
        self.bn5 = nn.BatchNorm2d(128)
        self.bn6 = nn.BatchNorm2d(128)

        self.bn7 = nn.BatchNorm2d(128)
        self.bn8 = nn.BatchNorm2d(128)
        self.bn9 = nn.BatchNorm2d(128)

        self.conv10 = nn.Conv2d(   128,      int(latent_size / 16), 3, stride=1, padding=1)
        #self.fc1 = nn.Linear(128*4*4, latent_size)
        self.dr1 = nn.Dropout2d(0.2)
        self.dr2 = nn.Dropout2d(0.2)
        self.dr3 = nn.Dropout2d(0.2)

        self.apply(weights_init)
        self.cuda()

    def forward(self, x, return_features=False):
        batch_size = len(x)

        x = self.dr1(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = nn.LeakyReLU(0.2, inplace=True)(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = nn.LeakyReLU(0.2, inplace=True)(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = nn.LeakyReLU(0.2, inplace=True)(x)

        x = self.dr2(x)
        x = self.conv4(x)
        x = self.bn4(x)
        x = nn.LeakyReLU(0.2, inplace=True)(x)
        x = self.conv5(x)
        x = self.bn5(x)
        x = nn.LeakyReLU(0.2, inplace=True)(x)
        x = self.conv6(x)
        x = self.bn6(x)
        x = nn.LeakyReLU(0.2, inplace=True)(x)

        x = self.dr3(x)
        x = self.conv7(x)
        x = self.bn7(x)
        x = nn.LeakyReLU(0.2, inplace=True)(x)
        x = self.conv8(x)
        x = self.bn8(x)
        x = nn.LeakyReLU(0.2, inplace=True)(x)
        x = self.conv9(x)
        x = self.bn9(x)
        x = nn.LeakyReLU(0.2, inplace=True)(x)
        x = self.conv10(x)
        x = x.view(batch_size, -1)
        #x = self.fc1(x)
        return x


class generator32(nn.Module):
    def __init__(self, latent_size=100, batch_size=64, **kwargs):
        super(self.__class__, self).__init__()
        self.latent_size = latent_size
        #self.fc1 = nn.Linear(latent_size, 4*4*512)
        self.conv0 = nn.ConvTranspose2d(   int(latent_size/16),      512, 3, stride=1, padding=1, bias=False)
        self.conv1 = nn.ConvTranspose2d(   512,      256, 4, stride=2, padding=1, bias=False)
        self.conv2 = nn.ConvTranspose2d(   256,      128, 4, stride=2, padding=1, bias=False)
        self.conv3 = nn.ConvTranspose2d(   128,        3, 4, stride=2, padding=1)
        self.bn0 = nn.BatchNorm2d(512)
        self.bn1 = nn.BatchNorm2d(256)
        self.bn2 = nn.BatchNorm2d(128)

        self.batch_size = batch_size
        self.apply(weights_init)
        self.cuda()

    def forward(self, x):
        """
        Based on Improved GAN from Salimans et al
        For reference:

        nn.batch_norm(ll.DenseLayer(gen_layers[-1], num_units=4*4*512, W=Normal(0.05), nonlinearity=nn.relu), g=None)
        ll.ReshapeLayer(gen_layers[-1], (args.batch_size,512,4,4))
        nn.batch_norm(nn.Deconv2DLayer(gen_layers[-1], (args.batch_size,256,8,8), (5,5), W=Normal(0.05), nonlinearity=nn.relu), g=None) # 4 -> 8
        nn.batch_norm(nn.Deconv2DLayer(gen_layers[-1], (args.batch_size,128,16,16), (5,5), W=Normal(0.05), nonlinearity=nn.relu), g=None) # 8 -> 16
        nn.weight_norm(nn.Deconv2DLayer(gen_layers[-1], (args.batch_size,3,32,32), (5,5), W=Normal(0.05), nonlinearity=T.tanh), train_g=True, init_stdv=0.1) # 16 -> 32
        """
        batch_size = x.shape[0]
        x = x.resize(batch_size, int(self.latent_size / 16), 4, 4)
        #x = self.fc1(x)
        x = self.conv0(x)
        x = nn.ReLU(inplace=True)(x)
        x = self.bn0(x)
        # 512 x 4 x 4
        x = self.conv1(x)
        x = nn.ReLU(inplace=True)(x)
        x = self.bn1(x)
        # 256 x 8 x 8
        x = self.conv2(x)
        x = nn.ReLU(inplace=True)(x)
        x = self.bn2(x)
        # 128 x 16 x 16
        x = self.conv3(x)
        # 3 x 32 x 32
        x = nn.Sigmoid()(x)
        return x


class multiclassDiscriminator32(nn.Module):
    def __init__(self, latent_size=100, num_classes=2, batch_size=64, **kwargs):
        super(self.__class__, self).__init__()
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.conv1 = nn.Conv2d(3,       64,     3, 1, 1, bias=False)
        self.conv2 = nn.Conv2d(64,      64,     3, 1, 1, bias=False)
        self.conv3 = nn.Conv2d(64,     128,     3, 2, 1, bias=False)

        self.conv4 = nn.Conv2d(128,    128,     3, 1, 1, bias=False)
        self.conv5 = nn.Conv2d(128,    128,     3, 1, 1, bias=False)
        self.conv6 = nn.Conv2d(128,    128,     3, 2, 1, bias=False)

        self.conv7 = nn.Conv2d(128,    128,     3, 1, 1, bias=False)
        self.conv8 = nn.Conv2d(128,    128,     3, 1, 1, bias=False)
        self.conv9 = nn.Conv2d(128,    128,     3, 2, 1, bias=False)

        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)

        self.bn4 = nn.BatchNorm2d(128)
        self.bn5 = nn.BatchNorm2d(128)
        self.bn6 = nn.BatchNorm2d(128)

        self.bn7 = nn.BatchNorm2d(128)
        self.bn8 = nn.BatchNorm2d(128)
        self.bn9 = nn.BatchNorm2d(128)

        self.fc1 = nn.Linear(128*4*4, num_classes)
        self.dr1 = nn.Dropout2d(0.2)
        self.dr2 = nn.Dropout2d(0.2)
        self.dr3 = nn.Dropout2d(0.2)

        self.apply(weights_init)
        self.cuda()

    def forward(self, x, return_features=False):
        """
        Based on Salimans et al

        Reference:
        ll.DropoutLayer(disc_layers[-1], p=0.2)
        nn.weight_norm(dnn.Conv2DDNNLayer(disc_layers[-1], 64, (3,3), pad=1, W=Normal(0.05), nonlinearity=nn.lrelu))
        nn.weight_norm(dnn.Conv2DDNNLayer(disc_layers[-1], 64, (3,3), pad=1, W=Normal(0.05), nonlinearity=nn.lrelu))
        nn.weight_norm(dnn.Conv2DDNNLayer(disc_layers[-1], 64, (3,3), pad=1, stride=2, W=Normal(0.05), nonlinearity=nn.lrelu))
        ll.DropoutLayer(disc_layers[-1], p=0.5)
        nn.weight_norm(dnn.Conv2DDNNLayer(disc_layers[-1], 128, (3,3), pad=1, W=Normal(0.05), nonlinearity=nn.lrelu))
        nn.weight_norm(dnn.Conv2DDNNLayer(disc_layers[-1], 128, (3,3), pad=1, W=Normal(0.05), nonlinearity=nn.lrelu))
        nn.weight_norm(dnn.Conv2DDNNLayer(disc_layers[-1], 128, (3,3), pad=1, stride=2, W=Normal(0.05), nonlinearity=nn.lrelu))
        ll.DropoutLayer(disc_layers[-1], p=0.5)
        nn.weight_norm(dnn.Conv2DDNNLayer(disc_layers[-1], 128, (3,3), pad=0, W=Normal(0.05), nonlinearity=nn.lrelu))
        nn.weight_norm(ll.NINLayer(disc_layers[-1], num_units=128, W=Normal(0.05), nonlinearity=nn.lrelu))
        nn.weight_norm(ll.NINLayer(disc_layers[-1], num_units=128, W=Normal(0.05), nonlinearity=nn.lrelu))
        ll.GlobalPoolLayer(disc_layers[-1])
        nn.weight_norm(ll.DenseLayer(disc_layers[-1], num_units=10, W=Normal(0.05), nonlinearity=None), train_g=True, init_stdv=0.1)
        """
        batch_size = len(x)

        x = self.dr1(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = nn.LeakyReLU(0.2, inplace=True)(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = nn.LeakyReLU(0.2, inplace=True)(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = nn.LeakyReLU(0.2, inplace=True)(x)

        x = self.dr2(x)
        x = self.conv4(x)
        x = self.bn4(x)
        x = nn.LeakyReLU(0.2, inplace=True)(x)
        x = self.conv5(x)
        x = self.bn5(x)
        x = nn.LeakyReLU(0.2, inplace=True)(x)
        x = self.conv6(x)
        x = self.bn6(x)
        x = nn.LeakyReLU(0.2, inplace=True)(x)

        x = self.dr3(x)
        x = self.conv7(x)
        x = self.bn7(x)
        x = nn.LeakyReLU(0.2, inplace=True)(x)
        x = self.conv8(x)
        x = self.bn8(x)
        x = nn.LeakyReLU(0.2, inplace=True)(x)
        x = self.conv9(x)
        x = self.bn9(x)
        x = nn.LeakyReLU(0.2, inplace=True)(x)

        x = x.view(batch_size, -1)
        if return_features:
            return x
        x = self.fc1(x)
        return x
