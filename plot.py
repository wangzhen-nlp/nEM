import os
import sys
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt


def PR_curve(cdir1, cdir2, auto):
    plt.clf()
    p1 = np.load(os.path.join(cdir1, 'y_precision{}.npy'.format('_auto' if auto else '')))[:3000]
    r1 = np.load(os.path.join(cdir1, 'y_recall{}.npy'.format('_auto' if auto else '')))[:3000]
    p2 = np.load(os.path.join(cdir2, 'y_precision{}.npy'.format('_auto' if auto else '')))[:3000]
    r2 = np.load(os.path.join(cdir2, 'y_recall{}.npy'.format('_auto' if auto else '')))[:3000]
    plt.plot(r1, p1, color = 'red', ls=':', lw=1.5, label = 'RE')
    plt.plot(r2, p2, color = 'red', ls='-', lw=2, label = 'EM')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.6, 1.0])
    plt.xlim([0.0, 0.5])
    plt.legend(loc="upper right")
    leg = plt.gca().get_legend()
    ltext = leg.get_texts()
    plt.setp(ltext, fontsize='small')
    plt.grid(True)
    plt.savefig('PR_curve.png')


def main():
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('re')
    parser.add_argument('em')
    parser.add_argument('--auto', action='store_true', default=False)
    args = parser.parse_args()
    PR_curve(args.re, args.em, args.auto)


if __name__ == '__main__':
    main()
