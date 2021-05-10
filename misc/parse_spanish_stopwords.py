import pandas as pd
import time
import pickle



def parse_spanish_abbr(file):
    df = pd.read_csv(file, sep='\t', encoding='utf-8',)
    keys = df.iloc[:,0].str.strip()
    values = df.iloc[:,1].str.replace('\(.*?\)', '', regex=True).str.strip()
    # sp_abbr_terms_dict = dict(zip(keys, values))
    df_new = pd.DataFrame(data={'sp_abbr':keys, 'desc':values})
    print (df_new.head(5))
    df_new.to_csv('/nfs/home/vyasa/projects/proj_off/data_off/clarify/spanish_comorbidity/spanish/spanish_terms_parsed.csv',encoding='utf-8', index=False, sep='\t')
    return df_new


def capitalize_(text):
    if type(text)==str:
        if len(text)>0:
            return text[0:1].capitalize()+text[1:].lower()
    else:
        return text


def sp_text_wtht_abbr(df_new):
    df = pd.read_csv('/nfs/home/vyasa/projects/proj_off/data_off/clarify/spanish_comorbidity/old_10_05_2021/comorbidities_modified.csv', 
         sep=',', error_bad_lines=True, encoding='utf-8')
    df = df[['value', 'replacement']]
    abbr_dict = dict(zip(df_new['sp_abbr'], df_new['desc']))
    # df['value_witht_abbr'] = df['value'].replace(abbr_dict)
    df['value_witht_abbr'] = df['value'].apply(lambda x: ' '.join([abbr_dict[i] if i in abbr_dict.keys() else i for i in x.split()]))
    df = df[['value', 'value_witht_abbr', 'replacement']]
    df.to_csv('/nfs/home/vyasa/projects/proj_off/data_off/clarify/spanish_comorbidity/comorbidities_modified_witht_abbr.csv', index=False, sep=',',)
    



def main():

    file = '/nfs/home/vyasa/projects/proj_off/data_off/clarify/spanish_comorbidity/spanish/spanish_terms.txt'
    df_new = parse_spanish_abbr(file)
    sp_text_wtht_abbr(df_new)

if __name__ == "__main__":
    past_time = time.time()
    main()
    print ("Total_Time ({}(in Hrs) : {}(in Mins)".format((time.time()-past_time)/3600.0, (time.time()-past_time)/60.0))


