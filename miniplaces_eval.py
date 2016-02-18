from neon.util.argparser import NeonArgparser
from neon.layers import Pooling
from neon.models import Model
from neon.data import ImageLoader
from neon.util.persist import save_obj, load_obj

# parse the command line arguments (generates the backend)
parser = NeonArgparser(__doc__)
parser.add_argument('--scale', type=int, default=112, help='scale to present input images')
args = parser.parse_args()
scale = args.scale
print scale
test = ImageLoader(set_name='validation', shuffle=False, do_transforms=False, inner_size=scale,
                   scale_range=scale, repo_dir=args.data_dir)

model_desc = load_obj(args.model_file)
model_desc['model']['config']['layers'].insert(-1, Pooling('all', op='avg').get_description())
model = Model(model_desc, weights_only=True)
softmaxes = model.get_outputs(test)
save_obj(softmaxes, "alex_outputs_{}.pkl".format(scale))
