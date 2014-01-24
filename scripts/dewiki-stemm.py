#!/usr/bin/env python
# A script to clean up a wikipedia dataset

# remove workds that are only 1 character long
#
# it looks like it easier sed -e 's/\s\+/\n/g' old > new to get 
# file that it is easily processable

from __future__ import division
import nltk, re, pprint
import sys

#nltk.download()

def delimited(filename, delimiter=' ', bufsize=4096):
    buf = ''
    with open(filename) as file:
        while True:
            newbuf = file.read(bufsize)
            if not newbuf:
                yield buf
                return
            buf += newbuf
            words = buf.split(delimiter)
            for word in words[:-1]:
                yield word
            buf = words[-1]
#            print "y but es: %s" % buf



def words(filename):
    with open(filename) as f:
        for line in f:
            for word in line.split():
                yield word


filename =  sys.argv[1]

print "Processing file: %s " %  filename

stemmer = nltk.stem.snowball.GermanStemmer(True)

#hackish to avoid tjhe removal of the follwing ending
#that are semantically important:

#In [10]: stemmer._GermanStemmer__step2_suffixes 
#Out[10]: (u'est', u'en', u'er', u'st')

#In [11]: stemmer._GermanStemmer__step3_suffixes 
#Out[11]: (u'isch', u'lich', u'heit', u'keit', u'end', u'ung', u'ig', u'ik')


#stemmer._GermanStemmer__step3_suffixes = ()
#stemmer._GermanStemmer__step2_suffixes = ()


for word in delimited(filename):
    if len(word) > 1:
        sys.stdout.write(stemmer.stem(word))
        sys.stdout.write(' ')


