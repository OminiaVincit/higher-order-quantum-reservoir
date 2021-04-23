from six.moves import urllib
origin = (
    'http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz'
)
urllib.request.urlretrieve(origin,'mnist.pkl.gz')

