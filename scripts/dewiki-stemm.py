#!/usr/bin/env python
# A script to clean up a wikipedia dataset

# remove workds that are only 1 character long
#
# it looks like it easier sed -e 's/\s\+/\n/g' old > new to get 
# file that it is easily processable

from __future__ import division
import nltk, re, pprint
import sys

nltk.download()

def delimited(filename, delimiter=' ', bufsize=4096):
    buf = ''
    with open(filename) as file:
        while True:
            newbuf = file.read(bufsize)
            if not newbuf:
                yield buf
                return
            buf += newbuf
            lines = buf.split(delimiter)
            for line in lines[:-1]:
                yield line
            buf = lines[-1]
#            print "y but es: %s" % buf



def words(filename):
    with open(filename) as f:
        for line in f:
            for word in line.split():
                yield word


filename =  sys.argv[1]

print "Processing file: %s " %  filename

stemmer = nltk.stem.snowball.GermanStemmer(True)

#hackish
stemmer.__step2_suffixes = []
stemmer.__step3_suffixes = []


for word in delimited(filename):
    if len(word) > 1:
        sys.stdout.write(stemmer.stem(word))
        sys.stdout.write(' ')


