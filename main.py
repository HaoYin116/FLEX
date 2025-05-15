from tools import run_net
from tools import test_net
from utils import parser
import random
import numpy as np
import torch


def main():
    # config
    args = parser.get_args()
    parser.setup(args)   
    if args.benchmark == 'MTL':
        if not args.usingDD:
            args.score_range = 100
    print(args)
    # run
    if args.test:
        test_net(args)
    else:
        run_net(args)

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

if __name__ == '__main__':
    set_seed(42)
    main()