import pandas as pd
import argparse
import os
import sys
import glob
import requests
import json
import html
import timeit
from datetime import datetime
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from tqdm import tqdm
tqdm.pandas()

smoothie = SmoothingFunction()
NMT_TGT = 'prediction'
BLEU_COL_KEY = 'onmt_bleu'


def calculate_score(candidate_list, reference_list):
    if not isinstance(candidate_list, str) and not isinstance(candidate_list, list):
        return 0

    candidate_len = len(candidate_list)

    if candidate_len < 4:
        score = sentence_bleu([reference_list], candidate_list, weights=(1.0, 0.0, 0.0, 0.0), smoothing_function=smoothie.method1)
    else:
        score = sentence_bleu([reference_list], candidate_list, smoothing_function=smoothie.method1)

    return score


def update_score(data, ref_code, tgt_code):
    reference = data[ref_code]
    # print(reference)

    nmt_model = data[tgt_code]
    bleu_score_nmt_model = calculate_score(nmt_model, reference)
    data[BLEU_COL_KEY] = bleu_score_nmt_model
    return data


def process_data(df, src_code, tgt_code):

    df[f'{src_code}'] = df[f'{src_code}'].replace(r'^\"(.*)\"$', r'\1', regex=True)
    df[f'{tgt_code}'] = df[f'{tgt_code}'].replace(r'^\"(.*)\"$', r'\1', regex=True)

    df[f'{src_code}'] = df[f'{src_code}'].replace(r'\"{2}', r'"', regex=True)
    df[f'{tgt_code}'] = df[f'{tgt_code}'].replace(r'\"{2}', r'"', regex=True)

    df[f'{src_code}'] = df[f'{src_code}'].replace(r'^( *> *)', '', regex=True)
    df[f'{tgt_code}'] = df[f'{tgt_code}'].replace(r'^( *> *)', '', regex=True)

    return df


def parse_args(argv):
    """Parse command line arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    # set the argument formats
    parser.add_argument(
        '--src_lang', '-src_lang', required=True, default='ko',
        help='source language')
    parser.add_argument(
        '--tgt_lang', '-tgt_lang', required=False, default='en',
        help='target language')
    parser.add_argument(
        '--label_data', '-label_data', required=True,
        help='file that has source, target, prediction corpus data (.csv, .tsv, .xlsx)')

    return parser.parse_args(argv[1:])


if __name__ == '__main__':
    args = parse_args(sys.argv)

    src_lang = args.src_lang
    tgt_lang = args.tgt_lang
    label_data = args.label_data

    print(f"[{datetime.now().strftime('%Y/%m/%d %H:%M:%S')}] BLEU scoring started...")
    print(f"[{datetime.now().strftime('%Y/%m/%d %H:%M:%S')}] Loading data...")
    file_type = label_data.split(f'/')[-1].split('.')[-1]

    if file_type == 'csv':
        df = pd.read_csv(f'{label_data}')

    elif file_type == 'tsv':
        df = pd.read_csv(f'{label_data}', sep='\t')

    elif file_type == 'xlsx':
        df = pd.read_excel(f'{label_data}', dtype={src_lang: str, tgt_lang: str}, na_values='NaN')

    # df = process_data(df, src_lang, tgt_lang)

    # tgt_list = []
    # with open(f'{tgt}') as f:
    #     for i, line in enumerate(f):
    #         tgt_list.append(line.strip())
    #
    # if len(tgt_list) != len(df):
    #     print('ERROR :: [src-ref] data and [tgt] data size is different.')
    #     exit(0)
    #
    # df[NMT_TGT] = tgt_list

    print(f"[{datetime.now().strftime('%Y/%m/%d %H:%M:%S')}] Calculating BLEU score...")

    df = df.progress_apply(update_score, args=(tgt_lang, NMT_TGT), axis=1)
    bleu_mean = df[BLEU_COL_KEY].mean(axis=0)
    df.loc['Mean', BLEU_COL_KEY] = bleu_mean

    print(f"[{datetime.now().strftime('%Y/%m/%d %H:%M:%S')}] BLEU comparison is done!")

    result_file = f'{label_data.split(os.sep)[-1].strip().split(f".{file_type}")[0]}_bleu_result_{datetime.today().strftime("%m%d%H%M%S")}.xlsx'
    result_path = f'data/bleu/{result_file}'
    df.to_excel(result_path, index=False )
    print(f"\n[{datetime.now().strftime('%Y/%m/%d %H:%M:%S')}] BLEU scoring completed... (Avg. BLEU: {bleu_mean}) \n")

    # ------------------------------------------------------------------------------
