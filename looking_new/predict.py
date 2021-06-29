from utils.predictor import *

parser = argparse.ArgumentParser(
        prog='python3 predict',
        usage='%(prog)s [options] images',
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        )
parser.add_argument('--version', action='version',
                        version='Looking Model {version}'.format(version=0.1))

parser.add_argument('--images', nargs='*',
                        help='input images')
parser.add_argument('--glob',
                        help='glob expression for input images (for many images)')
parser.add_argument('-o', '--image-output', default=None, nargs='?', const=True,
                        help='Whether to output an image, '
                             'with the option to specify the output path or directory')
parser.add_argument('--json-output', default=None, nargs='?', const=True,
                        help='Whether to output a json file, '
                             'with the option to specify the output path or directory')
parser.add_argument('--batch-size', default=1, type=int,
                        help='processing batch size')
parser.add_argument('--device', default='0', type=str,
                        help='cuda device')
parser.add_argument('--long-edge', default=None, type=int,
                        help='rescale the long side of the image (aspect ratio maintained)')
parser.add_argument('--loader-workers', default=None, type=int,
                        help='number of workers for data loading')
parser.add_argument('--precise-rescaling', dest='fast_rescaling',
                        default=True, action='store_false',
                        help='use more exact image rescaling (requires scipy)')
parser.add_argument('--disable-cuda', action='store_true',
                        help='disable CUDA')

decoder.cli(parser)
logger.cli(parser)
network.Factory.cli(parser)
show.cli(parser)
visualizer.cli(parser)

args = parser.parse_args()

predictor = Predictor(args)
predictor.predict(args)