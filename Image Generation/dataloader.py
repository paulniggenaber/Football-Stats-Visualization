import tensorflow as tf
import subprocess
import numpy as np 
import pickle 
import os 

## Processor with constant batch size across instances as class variable,
#  empty list for the data, list with filenames and dictionary key mX 
#  for color dataset
class Processor():
    def __init__(self, filenames, mX = 'm1'):
        self._data = []
        self._filenames = filenames
        self._mX = mX
        self._batch_size = 100

    def process(self):
        raise NotImplementedError
    
    ## lets the user access the data
    # @return self._data: list containing train, test und label data
    @property
    def data(self):
        return self._data
    
## ColorProcessor directly processing files when called
class ColorProcessor(Processor):
    def __init__(self, filenames, mX):
        super().__init__(filenames, mX)
        self.process()
    
    ## loads and processed the data appropriately and 
    #  saves it in a list self._data
    def process(self):
        for filename in self._filenames:
            try:
                data = pickle.load(open(filename, 'rb'))
                data = data[self._mX]
            except:
                data = np.load(filename)

            self._data.append(tf.data.Dataset.from_tensor_slices(data).batch(self._batch_size))

## BWProcessor directly processing files when called
class BWProcessor(Processor):
    def __init__(self, filenames, mX):
        super().__init__(filenames, mX)
        self.process()

    ## loads and processed the data appropriately and 
    #  saves it in a list self._data
    def process(self):
        for filename in self._filenames:
            data = np.load(filename)
            data = data/255
            #reshape from (60000, 28, 28) --> (60000, 784)
            data = data.reshape(data.shape[0], -1)
            self._data.append(tf.data.Dataset.from_tensor_slices(data).batch(self._batch_size))

## DataLoader with dictionaries holding specific processor classes and filenames and -links
#  Initiates links, filenames, an empty list for the data, extracts the data type from the filename,
#  has a key mX for the color dataset and flags for load and download control
class DataLoader():
    _processor_dict = {'color': ColorProcessor, 'bw': BWProcessor}
    _link_dict = {
                'mnist_bw': ['mnist_bw.npy', 'https://www.dropbox.com/scl/fi/fjye8km5530t9981ulrll/mnist_bw.npy?rlkey=ou7nt8t88wx1z38nodjjx6lch&st=5swdpnbr&dl=0'],
                'mnist_bw_te': ['mnist_bw_te.npy', 'https://www.dropbox.com/scl/fi/dj8vbkfpf5ey523z6ro43/mnist_bw_te.npy?rlkey=5msedqw3dhv0s8za976qlaoir&st=nmu00cvk&dl=0'],
                'mnist_bw_y_te': ['mnist_bw_y_te.npy', 'https://www.dropbox.com/scl/fi/8kmcsy9otcxg8dbi5cqd4/mnist_bw_y_te.npy?rlkey=atou1x07fnna5sgu6vrrgt9j1&st=m05mfkwb&dl=0'],
                'mnist_color': ['mnist_color.pkl', 'https://www.dropbox.com/scl/fi/w7hjg8ucehnjfv1re5wzm/mnist_color.pkl?rlkey=ya9cpgr2chxt017c4lg52yqs9&st=ev984mfc&dl=0'],
                'mnist_color_te': ['mnist_color_te.pkl', 'https://www.dropbox.com/scl/fi/w08xctj7iou6lqvdkdtzh/mnist_color_te.pkl?rlkey=xntuty30shu76kazwhb440abj&st=u0hd2nym&dl=0'],
                'mnist_color_y_te': ['mnist_color_y_te.npy', 'https://www.dropbox.com/scl/fi/fkf20sjci5ojhuftc0ro0/mnist_color_y_te.npy?rlkey=fshs83hd5pvo81ag3z209tf6v&st=99z1o18q&dl=0']
                }
    
    def __init__(self, filename, mX):
        self._links = [DataLoader._link_dict[filename][1], DataLoader._link_dict[f'{filename}_te'][1], DataLoader._link_dict[f'{filename}_y_te'][1]]
        self._filenames = [DataLoader._link_dict[filename][0], DataLoader._link_dict[f'{filename}_te'][0], DataLoader._link_dict[f'{filename}_y_te'][0]]
        self._data = []
        self._type = filename.split('_')[-1]
        self._is_loaded = False
        self._is_downloaded = False
        self._mX = mX

    ## outer function that contains an inner function
    # @param method: accessor method
    # @return dec: decorator function
    def is_loaded(method):
        ## calls load() if data not yet loaded
        # @return method: method that is decorated by is_loaded()
        def dec(self):
            if not self._is_loaded:
                self.load()
            return method(self)
        return dec
    
    ## downloads the data from the given links in link_dict and saves
    # them in the filenames provided in link_dict
    def download(self):
        if not self._is_downloaded:
            for filename, link in zip(self._filenames, self._links):
                if not os.path.exists(filename):
                    subprocess.run(["wget", link, '-O', filename]) 
        self._is_downloaded = True
    
    ## calls download and initiates a processor object
    # flags self._is_loaded as True to prevent redundant loading
    def load(self):
        self.download()
        processor_class = DataLoader._processor_dict[self._type]
        processor = processor_class(self._filenames, self._mX)
        self._data = processor.data
        self._is_loaded = True

    ## lets the user access the training data
    # @return self._data[0]: extracts training data from list
    @property
    @is_loaded
    def training_data(self):
        return self._data[0]

    ## lets the user access the training data
    # @return self._data[1]: extracts training data from list
    @property
    @is_loaded
    def test_data(self):
        return self._data[1]

    ## lets the user access the training data
    # @return self._data[2]: extracts training data from list
    @property
    @is_loaded
    def label_data(self):
        return self._data[2]
    
    ## lets the user access the type of data: 'color' or 'bw'
    # @return self._type: type of data: 'color' or 'bw'
    @property
    def type(self):
        return self._type