import numpy as np
from tqdm import tqdm
import argparse
import yaml
import pickle
from genconn import get_aux_prop


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('config_file', type=str)
    parser.add_argument('-o', '--output', type=str)

    args = parser.parse_args()

    with open(args.config_file) as stream:
        try:
            params = (yaml.safe_load(stream))
        except yaml.YAMLError as exc:
            print(exc)

    nexc = params['nexc']
    ninh = params['ninh']
    ntot = nexc + ninh

    R = np.random.rand(ntot,ntot)
    W = np.zeros((ntot,ntot))

    # exc<-exc
    W[:nexc,:nexc] = (R[:nexc,:nexc] < params['ee']['prob']).astype(float) * params['ee']['strength']

    # inh<-exc
    W[nexc:,:nexc] = (R[nexc:,:nexc] < params['ie']['prob']).astype(float) * params['ie']['strength']

    # exc<-inh
    W[:nexc,nexc:] = (R[:nexc,nexc:] < params['ei']['prob']).astype(float) * params['ei']['strength']

    # inh<-inh
    W[nexc:,nexc:] = (R[nexc:,nexc:] < params['ii']['prob']).astype(float) * params['ii']['strength']

    delays_tuple, exc_alpha = get_aux_prop(W)

    with open(args.output, 'wb') as file:
        pickle.dump((W, nexc, None, exc_alpha, delays_tuple, vars(args)), file)