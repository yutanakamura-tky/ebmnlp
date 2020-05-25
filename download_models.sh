#!/bin/bash

# Download BioELMo + CRF EBMNLP model checkpoint

ebmnlp_bioelmo_crf_checkpoint_path=models/ebmnlp_bioelmo_crf/ebmnlp_bioelmo_crf.ckpt

file_id=1p5OOA-4t7LnnTBiKfvr6VNR317oPbF4G
curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=${file_id}" > /dev/null
CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=${file_id}" -o ${ebmnlp_bioelmo_crf_checkpoint_path}
