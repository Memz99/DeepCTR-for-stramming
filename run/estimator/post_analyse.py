import csv
import pandas as pd
import numpy as np

from absl import flags, app

FLAGS = flags.FLAGS
flags.DEFINE_string('result_dir', "", "result_dir")

# test
#  class flags(object):
    #  def __init__(self):
        #  self.result_dir = "/home/hemingzhi/.jupyter/ctr/result/xtr_v2/20210609/multitask_epoch1_bs320_qp_noemp_MSE_eval"
#  FLAGS = flags()

def parse_config():
    cfg = {
        'metrics': ('ctr', 'lvtr'),
        'result_dir': FLAGS.result_dir,
    }
    return cfg

def gt_interal(cfg, df):
    points = np.linspace(0, 1, 11).round(1)
    interals = list(zip(points[:-1], points[1:]))
    lines = [list() for l, r in interals]
    for metric in cfg['metrics']:
        for i, interal in enumerate(interals):
            l, r = interal
            ind = np.logical_and(l <= df['gt_'+metric], df['gt_'+metric] < r)
            dff = df[ind]
            gt_mean = dff['gt_'+metric].mean().round(2)
            gt_std = dff['gt_'+metric].std().round(2)
            model_mean = dff['model_'+metric].mean().round(2)
            model_std = dff['model_'+metric].std().round(2)
            lines[i].extend([f"{l}~{r}", len(dff), gt_mean, gt_std, model_mean, model_std])
    lines = np.array(lines)
    with open(f"{cfg['result_dir']}/gt_interal.csv", 'w') as f:
        columns = [t+'_'+m for t in cfg['metrics'] for m in ['INTERAL', 'gt_N', 'gt_mean', 'gt_std', 'model_mean', 'model_std']]
        f.write(','.join(columns) + '\n')
        f.write('\n'.join([','.join(line) for line in lines]))


def main(argv):
    cfg = parse_config()
    df = pd.read_csv(f"{cfg['result_dir']}/prediction", sep='\t', error_bad_lines=False, quoting=csv.QUOTE_NONE)
    gt_interal(cfg, df)


if __name__ == '__main__':
    app.run(main)

    #  main(argv=None)
