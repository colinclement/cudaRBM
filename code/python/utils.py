import numpy as np
import scipy as sp
import struct

def saveSpinsBinary(filename, spins):
    N = spins.shape[-1]
    Nbytes = N/8
    with open(filename, 'wb') as outfile:
        for sp in spins:
            outspins = struct.pack('B'*Nbytes, *list(np.packbits((sp+1)/2)))
            outfile.write(outspins)
def numpySavedStream(npyfile):
    """Given an open file which has been saved with numpy.save,
    returns a generator which returns each time step saved"""
    try:
        while True:
            yield np.load(npyfile)
    except (EOFError, IOError):
        pass #Reached end of file

def spinPack(spins):
    """Used for one time step in the simulation, 
    i.e. spins.shape = (N_temps, L, L) """
    return np.packbits(np.ravel((spins > 0).astype('uint8')))

def spinUnpack(packedbits, L):
    """Unpack a time step of spins from spinPack"""
    con = np.unpackbits(packedbits).reshape(-1,L,L).astype('int')
    return 2*con-1

def loadSpins(npyfilename, L):
    with open(npyfilename, 'rb') as infile:
        inp = [spinUnpack(p,L)[0] for p in numpySavedStream(infile)]
    return np.array(inp)




