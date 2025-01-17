import argparse

def get_opts():
    parser = argparse.ArgumentParser()

    parser.add_argument('--image_path', type=str, default='/ys/mywork/Coordinate-MLPs-master/data/2d_ct_data/train/355.tiff',
                        help='path to the image to reconstruct')
    # parser.add_argument('--image_file', type=str, default='./images',
    #                     help='file to the image to reconstruct')
    parser.add_argument('--img_wh', nargs="+", type=int, default=[256, 256],
                        help='resolution (img_w, img_h) of the image')
    parser.add_argument('--use_pe', default=False, action='store_true',
                        help='use positional encoding or not')
    parser.add_argument('--arch', type=str, default='relu',
                        choices=['relu', 'ff', 'siren', 'gabor', 'bacon',
                                 'gaussian', 'quadratic', 'multi-quadratic',
                                 'laplacian', 'super-gaussian', 'expsin'],
                        help='network structure')
    parser.add_argument('--a', type=float, default=1.)
    parser.add_argument('--b', type=float, default=1.)
    parser.add_argument('--act_trainable', default=False, action='store_true',
                        help='whether to train activation hyperparameter')

    parser.add_argument('--sc', type=float, default=10.,
                        help='fourier feature scale factor (std of the gaussian)')
    parser.add_argument('--omega_0', type=float, default=30.,
                        help='omega in siren')

    parser.add_argument('--batch_size', type=int, default=256*256,
                        help='number of batch size')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='learning rate')
    parser.add_argument('--num_epochs', type=int, default=1500,
                        help='number of epochs')

    parser.add_argument('--exp_name', type=str, default='exp',
                        help='experiment name')
    
    parser.add_argument('--num_projections', type=int, default=20,
                        help='num_projections')
    parser.add_argument('--test_num_projections', type=int, default=10,
                        help='num_projections')                        

    return parser.parse_args()