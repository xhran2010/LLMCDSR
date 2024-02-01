import argparse

def str2bool(s):
    if s not in {'false', 'true'}:
        raise ValueError('Not a valid boolean string')
    return s == 'true'

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--exp_name', default='default', type=str)

    parser.add_argument('--log_step', default=-1, type=int)
    parser.add_argument('--eval_epoch', default=1, type=int)
    parser.add_argument('--num_epochs', default=200, type=int)
    parser.add_argument('--patience', default=10, type=int)

    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--temperature', default=1, type=float)
    parser.add_argument('--rec_weight', default=1, type=float)
    parser.add_argument('--cl_weight', default=0.1, type=float)
    parser.add_argument('--dropout_rate', default=0.5, type=float)
    parser.add_argument('--l2_emb', default=0.0, type=float)

    parser.add_argument('--maxlen', default=50, type=int)
    parser.add_argument('--hidden_units', default=64, type=int)
    parser.add_argument('--num_blocks', default=2, type=int)
    parser.add_argument('--num_heads', default=1, type=int)

    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--device', default='cuda:0', type=str)
    
    parser.add_argument('--inference', default=False, type=str2bool)
    parser.add_argument('--state_dict_path', default=None, type=str)

    parser.add_argument('--domain', default='A', type=str)

    args = parser.parse_args()
    print("DEVICE: {}".format(args.device))
    return args