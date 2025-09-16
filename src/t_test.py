import argparse
import os.path

import pandas as pd
from scipy.stats import ttest_ind

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--a', default='~/aya-results/mt0-lora-ru-ax.scores_sample.csv')
    parser.add_argument('--b', default='~/aya-results/mt0-qlora-ru-ax.scores_sample.csv')
    args = parser.parse_args()
    args.a = os.path.expanduser(args.a)
    args.b = os.path.expanduser(args.b)
    model_a = os.path.split(args.a)[-1].split(os.extsep)[0]
    model_b = os.path.split(args.b)[-1].split(os.extsep)[0]
    out_dir = '../t_tests/'
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{model_a}_vs_{model_b}.txt")
    with open(out_path, 'w') as out:
        out.write(model_a + '\n')
        out.write(model_b + '\n')
        for metric in ['bleu', 'bertscore']:
            out.write(metric + '\n')
            a = pd.read_csv(args.a)[metric]
            out.write(f"A mean {a.mean()}\n")
            out.write(f"Number of samples A {a.shape[0]}\n")
            b = pd.read_csv(args.b)[metric]
            out.write(f"B mean {b.mean()}\n")
            out.write(f"Number of samples B {b.shape[0]}\n")
            result = ttest_ind(a, b, nan_policy='raise')
            out.write(str(result) + '\n')
            if result.pvalue > 0.05:
                out.write("TtestResult.pvalue > 0.05 -> do not reject the null hypothesis of equal population means\n")
            else:
                out.write(
                    "TtestResult.pvalue < 0.05 -> reject the null hypothesis of equal population means\n"
                )
            out.write(str(result.confidence_interval()) + '\n')
            out.write("ConfidenceInterval shows the range into which the difference between mean A and mean B falls\n")
            out.write('-------------------------------------------------------------------------------------------------\n')
    print(f"Result saved to {out_path}")
