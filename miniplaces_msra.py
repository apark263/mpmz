from neon.util.argparser import NeonArgparser
from neon.initializers import Kaiming, IdentityInit
from neon.layers import Conv, Pooling, GeneralizedCost, Activation
from neon.layers import MergeSum, SkipNode
from neon.optimizers import GradientDescentMomentum, Schedule
from neon.transforms import Rectlin, Softmax, CrossEntropyMulti, TopKMisclassification
from neon.models import Model
from neon.data import ImageLoader
from neon.callbacks.callbacks import Callbacks

# parse the command line arguments (generates the backend)
parser = NeonArgparser(__doc__)
parser.add_argument('--depth', type=int, default=9,
                    help='depth of each stage (network depth will be 6n+2)')
args = parser.parse_args()

# setup data provider
imgset_options = dict(inner_size=112, repo_dir=args.data_dir)
train = ImageLoader(set_name='train', shuffle=True, do_transforms=True,
                    aspect_ratio=130, contrast_range=(75, 125), scale_range=(128,240),
                    **imgset_options)

test = ImageLoader(set_name='validation', shuffle=False, do_transforms=False,
                   scale_range=128, **imgset_options)


def conv_params(fsize, nfm, stride=1, relu=True, batch_norm=True):
    return dict(fshape=(fsize, fsize, nfm), strides=stride, padding=(1 if fsize > 1 else 0),
                activation=(Rectlin() if relu else None),
                init=Kaiming(local=True),
                batch_norm=batch_norm)


def module_factory(nfm, stride=1):
    mainpath = [Conv(**conv_params(3, nfm, stride=stride)),
                Conv(**conv_params(3, nfm, relu=False))]
    sidepath = [SkipNode() if stride == 1 else Conv(**conv_params(1, nfm, stride, relu=False))]
    module = [MergeSum([mainpath, sidepath]),
              Activation(Rectlin())]
    return module


# Structure of the deep residual part of the network:
# args.depth modules of 2 convolutional layers each at feature map depths of 16, 32, 64
nfms = [2**(stage + 5) for stage in sorted(range(4) * args.depth)]
strides = [1] + [1 if cur == prev else 2 for cur, prev in zip(nfms[1:], nfms[:-1])]

# Now construct the network
layers = [Conv(**conv_params(3, 32, 2))]
for nfm, stride in zip(nfms, strides):
    layers.append(module_factory(nfm, stride))
layers.append(Pooling(7, op='avg'))
layers.append(Conv(**conv_params(1, 100, relu=False, batch_norm=False)))
# layers.append(Pooling('all', op='avg'))
layers.append(Activation(Softmax()))

model = Model(layers=layers)
opt = GradientDescentMomentum(0.1, 0.9, wdecay=0.0005, schedule=Schedule([40, 70], 0.1))

# configure callbacks
valmetric = TopKMisclassification(k=5)
callbacks = Callbacks(model, train, eval_set=test, metric=valmetric, **args.callback_args)
callbacks.add_deconv_callback(train, test)

cost = GeneralizedCost(costfunc=CrossEntropyMulti())

model.fit(train, optimizer=opt, num_epochs=args.epochs, cost=cost, callbacks=callbacks)
