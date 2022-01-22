import numpy as np
import pandas as pd
from tqdm.auto import tqdm
import zarr

restrict_query=f'gene_symbol==["nontargeting"]'
metadata = pd.read_csv(f'~/gemelli/dataset_info/1_test_samples.csv').query(restrict_query)

z_in = zarr.open('/nrs/funke/funkl/data/cell_patches/nontargeting.zarr',mode='r')
z_out = zarr.open(
    '/nrs/funke/funkl/data/dapi/nontargeting_test_samples.zarr',
    mode='w',
    shape=(metadata.pipe(len),4,256,256),
    chunks=(10,4,256,256),
    dtype=np.uint16
    )

# z_out['images'] = np.zeros((metadata.pipe(len),4,256,256),dtype=np.uint16)
z_out.attrs['metadata'] = []

for n,(_,image) in tqdm(enumerate(metadata.iterrows())):
    z_out[n] = z_in[image['array']][image['array_index']]
    z_out.attrs['metadata'] += [dict(image)]
