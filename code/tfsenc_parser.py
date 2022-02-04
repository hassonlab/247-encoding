import argparse


def parse_arguments():
    """Read commandline arguments
    Returns:
        Namespace: input as well as default arguments
    """
    parser = argparse.ArgumentParser()

    parser.add_argument('--project-id', type=str, default=None)

    # group = parser.add_mutually_exclusive_group()
    parser.add_argument('--sid', nargs='?', type=int, default=None)
    parser.add_argument('--sig-elec-file', nargs='?', type=str, default=None)

    parser.add_argument('--conversation-id', type=int, default=0)

    parser.add_argument('--word-value', type=str, default='all')
    parser.add_argument('--window-size', type=int, default=200)

    group1 = parser.add_mutually_exclusive_group()
    group1.add_argument('--shuffle', action='store_true', default=False)
    group1.add_argument('--phase-shuffle', action='store_true', default=False)

    parser.add_argument('--parallel', action='store_true', default=False)

    parser.add_argument('--normalize', nargs='?', type=str, default=None)

    parser.add_argument('--lags', nargs='+', type=int)
    parser.add_argument('--output-prefix', type=str, default='test')
    parser.add_argument('--emb-type', type=str, default=None)
    parser.add_argument('--context-length', type=int, default=0)
    parser.add_argument('--layer-idx', type=int, default=1)
    parser.add_argument('--datum-emb-fn', nargs='?', type=str, default=None)
    parser.add_argument('--electrodes', nargs='*', type=int)
    parser.add_argument('--npermutations', type=int, default=1)
    parser.add_argument('--min-word-freq', nargs='?', type=int, default=5)
    parser.add_argument('--exclude-nonwords', action='store_true')
    parser.add_argument('--job-id', type=int, default=0)
    parser.add_argument('--test-near-neighbor', action='store_true')

    parser.add_argument('--pca-to', nargs='?', type=int, default=0)

    parser.add_argument('--align-with', nargs='*', type=str, default=None)

    parser.add_argument('--output-parent-dir', type=str, default='test')
    parser.add_argument('--save-pred', action='store_true')
    parser.add_argument('--pkl-identifier', type=str, default=None)

    args = parser.parse_args()

    if not args.sid and args.electrodes:
        parser.error("--electrodes requires --sid")

    if not (args.shuffle or args.phase_shuffle):
        args.npermutations = 1

    if args.sig_elec_file and args.sid not in [625, 676]:  # NOTE hardcoded
        args.sid = 777

    return args
