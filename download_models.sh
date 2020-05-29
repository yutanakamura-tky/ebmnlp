#!/bin/bash

# Download pretrained BioELMo

bioelmo_dir=models/bioelmo

if [ -e ${bioelmo_dir} ]; then
    :
else
    mkdir ${bioelmo_dir}
fi

bioelmo_options_path=${bioelmo_dir}/biomed_elmo_options.json
bioelmo_model_path=${bioelmo_dir}/bioelmo_elmo_weights.hdf5
bioelmo_vocab_path=${bioelmo_dir}/vocab.txt

dl_options=true
dl_model=true
dl_vocab=true

paths=($bioelmo_options_path $bioelmo_model_path $bioelmo_vocab_path)
flags=($dl_options $dl_model $dl_vocab)

for ix in ${!paths[@]}
do
    if [ -s ${paths[ix]} ]; then
        while :
        do
            read -n 1 -p "File ${paths[ix]} already exists. Download it again? (y/n): " tmp_flag
            echo
            if  [ $tmp_flag = "y" ] || [ $tmp_flag = "n" ]; then
                break
            else
                echo "Invalid option!"
            fi
        done
        if [ $tmp_flag = "n" ]; then
            flags[ix]=false
        else
            :
        fi
    else
        :
    fi
done


if "${flags[0]}"; then
    file_id=19sLZ1NhUtD_bMgTstSRWoVDx6Vm-T8Qt
    curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=${file_id}" > /dev/null
    CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
    curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=${file_id}" -o ${bioelmo_options_path}
    echo "Download complete: ${bioelmo_options_path}"
else
    :
fi

if "${flags[1]}"; then
    file_id=1CHRd5YQrt3ys64WfJkJR1KX72-2CaT4I
    curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=${file_id}" > /dev/null
    CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
    curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=${file_id}" -o ${bioelmo_model_path}
    echo "Download complete: ${bioelmo_model_path}"
else
    :
fi

if "${flags[2]}"; then
    file_id=15cXEVoRhUQ9oBnHVFP3nx6GQozczgxgP
    curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=${file_id}" > /dev/null
    CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
    curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=${file_id}" -o ${bioelmo_vocab_path}
    echo "Download complete: ${bioelmo_vocab_path}"
else
    :
fi



# Download BioELMo + CRF EBMNLP model checkpoint

bioelmo_crf_checkpoint_dir=models/ebmnlp_bioelmo_crf

if [ -e ${bioelmo_crf_checkpoint_dir} ]; then
    :
else
    mkdir ${bioelmo_crf_checkpoint_dir}
fi

bioelmo_crf_checkpoint_path=${bioelmo_crf_checkpoint_dir}/ebmnlp_bioelmo_crf.ckpt
dl_ckpt=true

paths=($bioelmo_crf_checkpoint_path)
flags=($dl_ckpt)

for ix in ${!paths[@]}
do
    if [ -s ${paths[ix]} ]; then
        while :
        do
            read -n 1 -p "File ${paths[ix]} already exists. Download it again? (y/n): " tmp_flag
            echo
            if  [ $tmp_flag = "y" ] || [ $tmp_flag = "n" ]; then
                break
            else
                echo "Invalid option!"
            fi
        done
        if [ $tmp_flag = "n" ]; then
            flags[ix]=false
        else
            :
        fi
    else
        :
    fi
done


if "${flags[0]}"; then
    file_id=1p5OOA-4t7LnnTBiKfvr6VNR317oPbF4G
    curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=${file_id}" > /dev/null
    CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
    curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=${file_id}" -o ${bioelmo_crf_checkpoint_path}
    echo "Download complete: ${bioelmo_crf_checkpoint_path}"
else
    :
fi
