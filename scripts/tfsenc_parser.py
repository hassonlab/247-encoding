import argparse
import sys


def parse_arguments():
    """Read commandline arguments
    Returns:
        Namespace: input as well as default arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-id", type=str, default=None)

    # group = parser.add_mutually_exclusive_group()
    parser.add_argument("--sid", nargs="?", type=int, default=None)
    parser.add_argument("--sig-elec-file", nargs="?", type=str, default=None)

    parser.add_argument("--conversation-id", type=int, default=0)

    parser.add_argument("--word-value", type=str, default="all")
    parser.add_argument("--window-size", type=int, default=200)

    group1 = parser.add_mutually_exclusive_group()
    group1.add_argument("--shuffle", action="store_true", default=False)
    group1.add_argument("--phase-shuffle", action="store_true", default=False)

    parser.add_argument("--parallel", action="store_true", default=False)

    parser.add_argument("--normalize", nargs="?", type=str, default=None)

    parser.add_argument("--lags", nargs="+", type=int)
    parser.add_argument("--output-prefix", type=str, default="test")
    parser.add_argument("--emb-type", type=str, default=None)
    parser.add_argument("--context-length", type=int, default=0)
    parser.add_argument("--layer-idx", type=int, default=1)
    parser.add_argument("--datum-emb-fn", nargs="?", type=str, default=None)
    parser.add_argument("--electrodes", nargs="*", type=int)
    parser.add_argument("--npermutations", type=int, default=1)
    parser.add_argument("--min-word-freq", nargs="?", type=int, default=0)
    parser.add_argument("--fold-num", nargs="?", type=int, default=10)
    parser.add_argument("--exclude-nonwords", action="store_true")
    parser.add_argument("--job-id", type=int, default=0)

    parser.add_argument("--pca-to", nargs="?", type=int, default=0)

    parser.add_argument("--align-with", nargs="*", type=str, default=None)

    parser.add_argument("--output-parent-dir", type=str, default="test")
    parser.add_argument("--pkl-identifier", type=str, default=None)

    parser.add_argument("--emb-mod", nargs="?", type=str, default="None")
    parser.add_argument("--datum-mod", nargs="?", type=str, default="all")
    parser.add_argument("--model-mod", nargs="?", type=str, default="None")

    parser.add_argument("--bad-convos", nargs="*", type=int, default=[])

    # If running the code in debug mode
    gettrace = getattr(sys, "gettrace", None)

    # TODO: Maybe move this to a test_config.ini file and use configparser()
    if gettrace():
        sys.argv = [
            "scripts/tfsenc_main.py",
            "--project-id",
            "podcast",
            "--pkl-identifier",
            "full",
            "--sid",
            "661",
            "--conversation-id",
            "0",
            "--electrodes",
            "1",
            "2",
            "3",
            "4",
            "5",
            "--emb-type",
            "gpt2-xl",
            "--context-length",
            "1024",
            "--align-with",
            "gpt2-xl",
            "--window-size",
            "200",
            "--word-value",
            "all",
            "--lags",
            "-100",
            "-50",
            "0",
            "50",
            "100",
            "--min-word-freq",
            "0",
            "--layer-idx",
            "1",
            "--normalize",
            "--output-parent-dir",
            "hg-podcast-full-661-gpt2-xl-all",
            "--output-prefix",
            "hg-200ms-all",
        ]
    args = parser.parse_args()

    if not args.sid and args.electrodes:
        parser.error("--electrodes requires --sid")

    if not (args.shuffle or args.phase_shuffle):
        args.npermutations = 1

    if args.sig_elec_file and args.sid not in [625, 676]:  # NOTE hardcoded
        args.sid = 777

    if not args.bad_convos:
        args.bad_convos = []

    if not args.emb_mod:
        args.emb_mod = "None"
    if not args.datum_mod:
        args.datum_mod = "None"

    return args
