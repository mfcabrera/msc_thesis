

class FileSetencesGenerator:
    def __init__(self,filename, delimiter=' ',bufsize = 4096):
        self.filename = filename
        self.bufsize = bufsize
        self.delimiter = delimiter

    def __iter__(self):
        buf = ''
        with open(self.filename) as file:
            while True:
                newbuf = file.read(self.bufsize)
                if not newbuf:
                    yield buf
                    return
                buf += newbuf
                words = buf.split(self.delimiter)
                yield words[:-1]
                buf = words[-1]




def delimited(filename, delimiter=' ', bufsize=4096):
    '''
    Creates a generator of word from a file based on a delimiter (by default white space).
    '''
    
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
