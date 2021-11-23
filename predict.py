from utils.predictor import *

parser = argparse.ArgumentParser(prog='python3 predict', usage='%(prog)s [options] images', description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--version', action='version',version='Looking Model {version}'.format(version=0.1))
parser.add_argument('--images', nargs='*',help='input images')
parser.add_argument('--transparency', default=0.4, type=float, help='transparency of the overlayed poses')
parser.add_argument('--looking_threshold', default=0.5, type=float, help='eye contact threshold')
parser.add_argument('--mode', default='joints', type=str, help='prediction mode')
parser.add_argument('--time', action='store_true', help='track comptutational time')
parser.add_argument('--glob', help='glob expression for input images (for many images)')

# Pifpaf args

parser.add_argument('-o', '--image-output', default=None, nargs='?', const=True, help='Whether to output an image, with the option to specify the output path or directory')
parser.add_argument('--json-output', default=None, nargs='?', const=True,help='Whether to output a json file, with the option to specify the output path or directory')
parser.add_argument('--batch_size', default=1, type=int, help='processing batch size')
parser.add_argument('--device', default='0', type=str, help='cuda device')
parser.add_argument('--long-edge', default=None, type=int, help='rescale the long side of the image (aspect ratio maintained)')
parser.add_argument('--loader-workers', default=None, type=int, help='number of workers for data loading')
parser.add_argument('--precise-rescaling', dest='fast_rescaling', default=True, action='store_false', help='use more exact image rescaling (requires scipy)')
parser.add_argument('--checkpoint_', default='shufflenetv2k30', type=str, help='backbone model to use')
parser.add_argument('--disable-cuda', action='store_true', help='disable CUDA')

decoder.cli(parser)
logger.cli(parser)
network.Factory.cli(parser)
show.cli(parser)
visualizer.cli(parser)

args = parser.parse_args()

predictor = Predictor(args)
predictor.predict(args)