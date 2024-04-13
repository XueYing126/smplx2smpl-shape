## tranfer shape space from smplx to smpl 

### Install environment:
```
conda create -n smplx python=3.8

cd transfer_model
pip install -r requirements.txt
cd ..

git clone https://github.com/vchoutas/torch-trust-ncg.git
cd torch-trust-ncg
python setup.py install
cd ..
find torch-trust-ncg -delete

pip install chumpy
python -m pip install numpy==1.23.1
```


#### smplx-Beta to Mesh

Modify gender, betas in smplx_obj.py 

```shell
python smplx_obj.py 
```
It will save mesh to '../transfer_data/meshes/amass_sample/001.obj'
#### Fit smpl-beta to mesh

```bash
python -m transfer_model --exp-cfg config_files/smplx2smpl.yaml
```