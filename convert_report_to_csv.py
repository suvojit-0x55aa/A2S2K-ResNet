import pandas as pd
import re
from glob import glob
import argparse
import os


def get_data(report):
    class_wise_acc_regex = r'\[[\d.\se\-\+(\\n)]*\]'
    oa_aa_kappa_regex = r'([-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?\s±\s[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?)'

    result = {}

    x = re.findall(oa_aa_kappa_regex, report)
    result['oa'] = x[0][0]
    result['aa'] = x[1][0]
    result['kappa'] = x[2][0]

    result['oa'] = "{:.2f}".format(
        float(result['oa'].split(' ± ')[0]) * 100) + ' ± ' + "{:.3f}".format(
            float(result['oa'].split(' ± ')[1]))
    result['aa'] = "{:.2f}".format(
        float(result['aa'].split(' ± ')[0]) * 100) + ' ± ' + "{:.3f}".format(
            float(result['aa'].split(' ± ')[1]))
    result['kappa'] = "{:.4f}".format(float(
        result['kappa'].split(' ± ')[0])) + ' ± ' + "{:.3f}".format(
            float(result['kappa'].split(' ± ')[1]))

    x = re.findall(class_wise_acc_regex, report)
    result['class_mean'] = x[0][1:-1].split()
    result['class_std'] = x[1][1:-1].split()
    result['class_wise'] = [
        "{:.2f}".format(float(m) * 100) + ' ± ' + "{:.3f}".format(float(n))
        for m, n in zip(result['class_mean'], result['class_std'])
    ]

    return result


def main(dataset, search_path, output_file):
    all_reports = glob(search_path + '/*' + dataset + '*.txt')
    no_of_labels = 0
    dataframe_dict = {}

    for report in all_reports:
        print('Processing...', report)
        column_name = os.path.basename(report)[:-4]
        with open(report) as f:
            report_content = f.read()

        result = get_data(report_content)

        dataframe_dict[column_name] = result['class_wise'] + [result['oa']] + [
            result['aa']
        ] + [result['kappa']]
        no_of_labels = len(result['class_wise'])

    label_list = [str(i)
                  for i in range(1, no_of_labels + 1)] + ['oa', 'aa', 'kappa']
    df = pd.DataFrame(dataframe_dict)
    df = df.reindex(sorted(df.columns), axis=1)
    df.insert(0, 'label', label_list)

    print('Saving...', dataset, 'report.')
    if output_file is not None:
        df.to_csv(output_file, index=False)
    else:
        if not os.path.exists('csv_reports'):
            os.makedirs('csv_reports')
        df.to_csv(
            os.path.join('csv_reports', dataset + '_report.csv'), index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Convert Code Generated Report to CSV.')
    parser.add_argument(
        '-d',
        '--dataset_name',
        dest='dataset',
        required=True,
        help="Name of dataset to search.")
    parser.add_argument(
        '-r',
        '--root_dir',
        dest='root_dir',
        default='./*/report',
        help="Directories to search for the report files.")
    parser.add_argument(
        '-o',
        '--output',
        dest='output',
        default=None,
        help="Output name to save csv.")
    args = parser.parse_args()
    main(args.dataset, args.root_dir, args.output)
