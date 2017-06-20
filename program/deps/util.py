from __future__ import division
import sys
import os

import re
import string
from nltk.tokenize import wordpunct_tokenize
from nltk.stem.porter import PorterStemmer
import pickle

# XXX: also redirect C streams, for capturing nvcc messages (else it looks like there are excessive newlines in log file)

class ConsoleAndFileLogger(object):
    """
    Logging helper which prints to a file and a stream (e.g. stdout) simultaneously.
    Can be used to redirect standard streams (sys.stdout, sys.stderr), so print(), warnings.warn(), 
    raise exceptions, etc., will be logged to console and file at the same time.

    Logging to file is done with a line buffer to allow updating lines for things like progress bars.
    """
    def __init__(self, filename='logfile.log', mode='w', stream=sys.stdout):
        assert mode in ['w', 'a']
        assert hasattr(stream, 'write') and hasattr(stream, 'flush') # basic check for valid stream, because redirecting sys.stdout,stderr to invalid Logger can cause trace not to be printed
        self.stream = stream
        self.file = open(filename, mode)
        self.linebuf = ''

    def __del__(self):
        # flush remainder of line buffer, and close file
        try:
            if len(self.linebuf) > 0:
                self.file.write(self.linebuf)
                self.file.flush()
            self.file.close()
        except:
            pass # file may be closed, unavailable, etc.

    def write(self, message):
        # write to stream (e.g. stdout)
        try:
            self.stream.write(message)
            self.stream.flush()
        except:
            pass # stream may be closed, unavailable, etc.

        # write to file (using line buffer to avoid writing many lines for things 
        # like progress bars that are erased and updated using '\b' and/or '\r')
        for c in message:
            if c == '\b':
                self.linebuf = self.linebuf[:-1]
            elif c == '\n':
                self.linebuf += c
                if len(self.linebuf) > 0:
                    try:
                        self.file.write(self.linebuf)
                        self.file.flush()
                    except:
                        pass # file may be closed, unavailable, etc.
                self.linebuf = ''
            elif c == '\r':
                self.linebuf = ''
            else:
                self.linebuf += c

    def flush(self):
        pass # already flushes each write

    #def __getattr__(self, attr):
    #    # all other attributes from stream
    #    # for code which assumes sys.stdout is a full fledged file object with 
    #    # methods such as fileno() (which includes code in the python standard library)
    #    return getattr(self.stream, attr)

def _mkdir(path, mode=0o777):
    if path == '':
        return
    try:
        os.makedirs(path, mode)
    except OSError as exc:
        import errno
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass # ignore exception because folder already exists
        else:
            raise # re-raise other exceptions
        
class log_to_file(object):
    """
    Log to file decorator.

    Example
    -------
    >>> @log_to_file('main.log', mode='a')
    >>> def main():
    >>>    print('hello world!')
    """
    def __init__(self, filename, mode='w', stream=sys.stdout):
        self.filename = filename
        self.mode = mode
        self.stream = stream

    def __call__(self, original_func):
        decorator_self = self
        def wrapped(*args, **kwargs):
            # keep track of original stdout, stderr
            old_stdout = sys.stdout
            old_stderr = sys.stderr

            # ensure path for log file exists (we do this here becaues typically main() is wrapped, 
            # so there is not opportunity to create folder before)
            _mkdir(os.path.dirname(self.filename))

            # create simultaneous stream and file logger
            logger = ConsoleAndFileLogger(self.filename, self.mode, self.stream)
    
            # flush stdout, stderr
            sys.stdout.flush()
            sys.stderr.flush()
    
            # redirect stdout, stderr
            sys.stdout = logger
            sys.stderr = logger

            # call original function
            original_func(*args, **kwargs)

            # restore original stdout, stderr
            sys.stdout = old_stdout
            sys.stderr = old_stderr
        return wrapped


def get_stopwords():
    return [u'i', u'me', u'my', u'myself', u'we', u'our', u'ours', u'ourselves', u'you', u'your', u'yours', u'yourself',
            u'yourselves', u'he', u'him', u'his', u'himself', u'she', u'her', u'hers', u'herself', u'it', u'its',
            u'itself',
            u'they', u'them', u'their', u'theirs', u'themselves', u'what', u'which', u'who', u'whom', u'this', u'that',
            u'these',
            u'those', u'am', u'is', u'are', u'was', u'were', u'be', u'been', u'being', u'have', u'has', u'had',
            u'having', u'do',
            u'does', u'did', u'doing', u'a', u'an', u'the', u'and', u'but', u'if', u'or', u'because', u'as', u'until',
            u'while',
            u'of', u'at', u'by', u'for', u'with', u'about', u'against', u'between', u'into', u'through', u'during',
            u'before',
            u'after', u'above', u'below', u'to', u'from', u'up', u'down', u'in', u'out', u'on', u'off', u'over',
            u'under',
            u'again', u'further', u'then', u'once', u'here', u'there', u'when', u'where', u'why', u'how', u'all',
            u'any', u'both',
            u'each', u'few', u'more', u'most', u'other', u'some', u'such', u'no', u'nor', u'not', u'only', u'own',
            u'same', u'so',
            u'than', u'too', u'very', u's', u't', u'can', u'will', u'just', u'don', u'should', u'now']

# change the way of tokenization
def tokeniser(desc_text):
    return [PorterStemmer().stem(token) for token in wordpunct_tokenize(re.sub('[%s]|\w*\d\w*' % re.escape(string.punctuation), '', desc_text.lower())) if token.lower() not in get_stopwords()]

def save_model(filepath, value, overwrite=False):
    import os.path
    # if file exists and should not be overwritten
    if not overwrite and os.path.isfile(filepath):
        import sys
        get_input = input
        if sys.version_info[:2] <= (2, 7):
            get_input = raw_input
        overwrite = get_input('[WARNING] %s already exists - overwrite? [y/n]' % (filepath))
        while overwrite not in ['y', 'n']:
            overwrite = get_input('Enter "y" (overwrite) or "n" (cancel).')
        if overwrite == 'n':
            return
        print('[TIP] Next time specify overwrite=True in save_model!')
    pickle.dump(value, open(filepath, 'wb'))

def load_model(filepath):
    return pickle.load(open(filepath, 'r'))