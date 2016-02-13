from neon.util.argparser import NeonArgparser
from neon.layers import Pooling
from neon.models import Model
from neon.data import ImageLoader
from neon.util.persist import save_obj, load_obj

# parse the command line arguments (generates the backend)
parser = NeonArgparser(__doc__)
args = parser.parse_args()

scales = [112, 128, 160, 240]
for scale in scales:
    print scale
    test = ImageLoader(set_name='validation', shuffle=False, do_transforms=False, inner_size=scale,
                       scale_range=scale, repo_dir=args.data_dir)

    model_desc = load_obj(args.model_file)
    model_desc['model']['config']['layers'].insert(-1, Pooling('all', op='avg').get_description())
    model = Model(model_desc, test, inference=True)
    softmaxes = model.get_outputs(test)
    save_obj(softmaxes, "bigfeat_dropout_SM_{}.pkl".format(scale))
