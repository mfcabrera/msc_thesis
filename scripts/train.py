#!/usr/bin/env python

import word2vec
import csv
import sys 
import argparse
import json



default_file = 'dewiki-20131024-pages-articles.txt'

parser = argparse.ArgumentParser(description='front end to train word to vec')
parser.add_argument('parameter_file', metavar='param_file',help='a json with the parameters')
parser.add_argument('text_file', metavar='input_file', default=default_file,help='the text file to create the vectors from')


args = parser.parse_args()

print args

#read the fil




# Read a CSV file with the train paramters



with open(args.parameter_file,'r') as input:
    runs = json.load(input)
    print runs
    for run_code in runs:
        print "Procesing %s " % run_code

        output = "dewiki-" + run_code + ".bin"
        
        
        word2vec.word2vec(args.text_file,output=output,
                          verbose=True,
                         **runs[run_code])

        print "Successfully finished processing %s" % run_code 





