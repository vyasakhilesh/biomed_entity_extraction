import pandas as pd 
from itertools import combinations
import time
import argparse
from utils.defs import *



def convert_csvfile_umlsSimilarityFile(inpath, infilename, outpath, outfilename, sample=False, is_comb=False):
    
    """ Open file and extract cui list
    """    
    df = pd.read_csv(inpath+infilename, encoding='utf-8')

    if is_comb==True:
        df['isComb'] = df['semantic_type'].apply(filter_cui_comb)
        df = df[df['isComb']==True]


    cui_list = df.CUI.tolist()
    
    if sample==True:
        cui_list = cui_list[0:20]

    cui_pair = list(combinations(cui_list, 2))
    print (len(cui_pair))

    """ Save to umls file
    """    
    with open(outpath+outfilename, 'wt') as file:
        for i in cui_pair:
            file.write(i[0]+'<>'+i[1]+'\n')


def argumentParser():
    parser = argparse.ArgumentParser(description='Convert CSVFile to UMLSFile')
    parser.add_argument('-i', '--inpath', type=str, required=True, action='store' ,help='inpath csv')
    parser.add_argument('-c', '--infilename', type=str, required=True, action='store', help='infilename csv file')
    parser.add_argument('-o', '--outpath', type=str, required=True, action='store', help='outpath txt')
    parser.add_argument('-t', '--outfilename', type=str, required=True, action='store', help='outfilename txt')
    parser.add_argument('-s', '--sample', type=bool, required=False, action='store', help='sample YES/NO')
    parser.add_argument('-b', '--comb', type=bool, required=False, action='store', help='comorbidities YES/NO')
    return parser.parse_args()


def main():
    args = argumentParser()
    convert_csvfile_umlsSimilarityFile(args.inpath, args.infilename, args.outpath, args.outfilename, args.sample, args.comb)

if __name__ == "__main__":
    past_time = time.time()
    main()
    print ("Total_Time ({}(in Hrs) : {}(in Mins)".format((time.time()-past_time)/3600.0, (time.time()-past_time)/60.0))




