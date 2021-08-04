from umls_similarity_maraca.convert_file import *
import time


def convert_csvfile_umlsSimilarityFile_():
    args = argumentParser()
    convert_csvfile_umlsSimilarityFile(args.inpath, args.infilename, args.outpath, args.outfilename, args.sample, args.comb)



def main():
    convert_csvfile_umlsSimilarityFile_()
    

if __name__ == "__main__":
    past_time = time.time()
    main()
    print ("Total_Time ({}(in Hrs) : {}(in Mins)".format((time.time()-past_time)/3600.0, (time.time()-past_time)/60.0))