#!/bin/bash

model_dir=models

if [ -e ${model_dir} ]; then
    :
else
    mkdir ${model_dir}
fi


# Download pretrained BioELMo

bioelmo_dir=${model_dir}/bioelmo

if [ -e ${bioelmo_dir} ]; then
    :
else
    mkdir ${bioelmo_dir}
fi

bioelmo_options_path=${bioelmo_dir}/biomed_elmo_options.json
bioelmo_model_path=${bioelmo_dir}/biomed_elmo_weights.hdf5
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
    curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=${file_id}" -o ${paths[0]}
    echo "Download complete: ${paths[0]}"
else
    :
fi

if "${flags[1]}"; then
    file_id=1CHRd5YQrt3ys64WfJkJR1KX72-2CaT4I
    curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=${file_id}" > /dev/null
    CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
    curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=${file_id}" -o ${paths[1]}
    echo "Download complete: ${paths[1]}"
else
    :
fi

if "${flags[2]}"; then
    file_id=15cXEVoRhUQ9oBnHVFP3nx6GQozczgxgP
    curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=${file_id}" > /dev/null
    CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
    curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=${file_id}" -o ${paths[2]}
    echo "Download complete: ${paths[2]}"
else
    :
fi


# Download pretrained BioBERT

biobert_dir=${model_dir}/biobert

if [ -e ${biobert_dir} ]; then
    :
else
    mkdir ${biobert_dir}
fi

biobert_v1_0_path=${biobert_dir}/biobert_v1.0_pubmed_pmc
biobert_v1_1_path=${biobert_dir}/biobert_v1.1_pubmed

biobert_v1_0_arc_path="$biobert_v1_0_path".tar.gz
biobert_v1_1_arc_path="$biobert_v1_1_path".tar.gz

dl_v1_0=true
dl_v1_1=true

paths=($biobert_v1_0_arc_path $biobert_v1_1_arc_path)
flags=($dl_v1_0 $dl_v1_1)

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
    file_id=1jGUu2dWB1RaeXmezeJmdiPKQp3ZCmNb7
    curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=${file_id}" > /dev/null
    CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
    curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=${file_id}" -o ${paths[0]}
    echo "Download complete: ${paths[0]}"
else
    :
fi

if "${flags[1]}"; then
    file_id=1R84voFKHfWV9xjzeLzWBbmY1uOMYpnyD
    curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=${file_id}" > /dev/null
    CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
    curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=${file_id}" -o ${paths[1]}
    echo "Download complete: ${paths[1]}"
else
    :
fi

tar -xf ${paths[@]}



# Download BioELMo + CRF EBMNLP model checkpoint

bioelmo_crf_checkpoint_dir=${model_dir}/ebmnlp_bioelmo_crf

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
