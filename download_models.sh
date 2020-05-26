#!/bin/bash

# Download pretrained BioELMo

bioelmo_dir=models/bioelmo
bioelmo_options_path=${bioelmo_dir}/biomed_elmo_options.json
bioelmo_model_path=${bioelmo_dir}/bioelmo_elmo_weights.hdf5

file_id=19sLZ1NhUtD_bMgTstSRWoVDx6Vm-T8Qt
curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=${file_id}" > /dev/null
CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=${file_id}" -o ${bioelmo_options_path}

file_id=1CHRd5YQrt3ys64WfJkJR1KX72-2CaT4I
curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=${file_id}" > /dev/null
CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=${file_id}" -o ${bioelmo_model_path}

# Download BioELMo + CRF EBMNLP model checkpoint

ebmnlp_bioelmo_crf_checkpoint_path=models/ebmnlp_bioelmo_crf/ebmnlp_bioelmo_crf.ckpt

file_id=1p5OOA-4t7LnnTBiKfvr6VNR317oPbF4G
curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=${file_id}" > /dev/null
CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=${file_id}" -o ${ebmnlp_bioelmo_crf_checkpoint_path}
