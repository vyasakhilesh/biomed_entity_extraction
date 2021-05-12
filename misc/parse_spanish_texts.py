import pandas as pd
import time
import pickle
import os

from pandas.io.parsers import count_empty_vals



def parse_spanish_abbr(file):
    df = pd.read_csv(file, sep='\t', encoding='utf-8',)
    keys = df.iloc[:,0].str.strip()
    values = df.iloc[:,1].str.replace('\(.*?\)', '', regex=True).str.strip()
    # sp_abbr_terms_dict = dict(zip(keys, values))
    df_new = pd.DataFrame(data={'sp_abbr':keys, 'desc':values})
    print (df_new.head(5))
    df_new.to_csv('/nfs/home/vyasa/projects/proj_off/data_off/clarify/spanish_comorbidity/spanish/spanish_terms_parsed.csv',encoding='utf-8', index=False, sep='\t')
    return df_new


def sp_text_wtht_abbr(df_new):
    df = pd.read_csv('/nfs/home/vyasa/projects/proj_off/data_off/clarify/spanish_comorbidity/old_10_05_2021/comorbidities_modified.csv', 
         sep=',', error_bad_lines=True, encoding='utf-8')
    df = df[['value', 'replacement']]
    abbr_dict = dict(zip(df_new['sp_abbr'], df_new['desc']))
    # df['value_witht_abbr'] = df['value'].replace(abbr_dict)
    df['value_witht_abbr'] = df['value'].apply(lambda x: ' '.join([abbr_dict[i] if i in abbr_dict.keys() else i for i in x.split()]))
    # df['value_witht_abbr'] = df['value'].apply(lambda x: ' '.join([abbr_dict[i] if i in abbr_dict.keys() else i for i in x.split()]))
    df = df[['value', 'value_witht_abbr', 'replacement']]
    df.to_csv('/nfs/home/vyasa/projects/proj_off/data_off/clarify/spanish_comorbidity/new/comorbidities_modified_witht_abbr.csv', index=False, sep=',',)
    df[['value_witht_abbr']]\
        .to_csv('/nfs/home/vyasa/projects/proj_off/data_off/clarify/spanish_comorbidity/new/comorbidities_value_abbr.csv', index=False, sep=',',)
    return df
    
    
def break_file(path, df, no_of_chars):
    text_lst = df['value_witht_abbr'].tolist()
    count = 0
    i = 0
    for j, text in enumerate(text_lst):
        count = count + len(text)
        if count > no_of_chars:
            df[['value_witht_abbr']].iloc[i:j].to_csv(os.path.join(path, 'break_files', 'files_{}_{}.csv'.format(i,j)), 
            index=False, encoding='utf-8', header=False)
            i = j
            count = 0



def main():
    path = '/nfs/home/vyasa/projects/proj_off/data_off/clarify/spanish_comorbidity/new/'
    file = '/nfs/home/vyasa/projects/proj_off/data_off/clarify/spanish_comorbidity/spanish/spanish_terms.txt'
    df_new = parse_spanish_abbr(file)
    df = sp_text_wtht_abbr(df_new)
    break_file(path, df, 4500)


if __name__ == "__main__":
    past_time = time.time()
    main()
    print ("Total_Time ({}(in Hrs) : {}(in Mins)".format((time.time()-past_time)/3600.0, (time.time()-past_time)/60.0))


