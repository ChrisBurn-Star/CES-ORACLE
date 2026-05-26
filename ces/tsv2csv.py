import pandas as pd

# Load TSV file
for i in range(1, 11):  # Assuming folders are named '1' to '10' and are in the current directory
    pathin = f"/home/jxzhou/PLM_PER/MoMoE_rawdata/BIO/TASK/REdata/GAD/{i}/train.tsv"
    pathou = f"/home/jxzhou/PLM_PER/MoMoE_rawdata/BIO/TASK/REdata/GAD/{i}/train2.csv"
    df = pd.read_csv(pathin, sep='\t',header=None, names=['sentence','label'])
    df.reset_index(inplace=True)
    df.rename(columns={'index': 'index'}, inplace=True)
    # Save as CSV
    df.to_csv(pathou, index=False, index_label='Sequence')