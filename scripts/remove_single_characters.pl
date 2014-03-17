#!/usr/bin/perl

#clean dooku data from single data, mostly comming from OCR errors.
while (<>) {
    s/(^|\s+)\w(?=\s+|$)/ /ig;
    # remove anything else that has not been matched yet
    chop;
    print $_;
}

