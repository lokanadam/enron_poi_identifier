import os
import sys
from string import maketrans, punctuation
from nltk.stem.snowball import SnowballStemmer
import zipfile

#def email_processing():

stemmer = SnowballStemmer('english')

def email_parser(email_list,labels,zipped = False):

    if zipped == True:
        zipfile.ZipFile('emails_by_address.zip').extractall()

    final_string =[]
    i = 0
    label_list = []
    j = 0

    for each, label in zip(email_list,labels):
        f_name = 'from_'+each+'.txt'
        path = os.path.join('emails_by_address/',f_name)
        try:
            email_path = open(path,'r')
        except IOError:
            print 'no address'
            i += 1
            continue

        for each_mail in email_path:
            email_path = os.path.join('..',each_mail[20:-1])
            f = open(email_path,'r')
            stemmed_string = email_test_process(f)
            final_string.append(stemmed_string)
            if label == 'poi':
                label_list.append(1)
            else:
                label_list.append(0)
            i += 1
    return final_string, label_list

def email_test_process(f):

    f.seek(0)
    string = f.read()

    e_content = string.split('X-FileName:')

    if len(e_content) > 1:

        u_content = e_content[1].translate(maketrans("",""),punctuation)

        words = u_content.split()

        stemmed_words = []

        for word in words:

            stemmed_words.append(stemmer.stem(word))

    return ' '.join(stemmed_words)