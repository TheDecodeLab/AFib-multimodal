info = []
from sys import argv

com_arch = int(argv[1])
n_comp = int(argv[2]) # 16  # Dimension to compress ECG data into
num_heads = int(argv[3]) #4  # Number of attention heads
ecg_app = int(argv[4]) #1 
# -1:labels shuffled
# 0:ecg_data
# 1:ecg_data_denoise
# 2:ecg_data_denoise_filtered
# 3:fft_data
# 4:fft_fake
# 5:ecg_fake
aug_p = float(argv[5]) #0.5
n_inc = int(argv[6]) #5
n_comp_ecg = int(argv[7]) #32
n_comp_ehr = int(argv[8]) #16
initial_learning_rate = float(argv[9]) #0.001

import os
os.environ["CUDA_VISIBLE_DEVICES"]="3"

import csv
import array
import base64
import xmltodict
import pylab as plt
from glob import glob
from tqdm import tqdm
import numpy as np
import pandas as pd
import pylab as plt
import seaborn as sns
from scipy.stats import chi2_contingency
from scipy.stats import chi2_contingency, ttest_ind
from sklearn.ensemble import RandomForestClassifier

import numpy as np

class ECGXMLReader:
    """ Extract voltage data from a ECG XML file """
    def __init__(self, path, unknown_index=0 , augmentLeads=False):
        # try: 
        with open(path, 'rb') as xml:
            self.ECG = xmltodict.parse(xml.read().decode('utf8'))

        self.unknown_index          = unknown_index
        self.augmentLeads           = augmentLeads
        self.path                   = path

        self.PatientDemographics    = self.ECG['RestingECG']['PatientDemographics']
        self.TestDemographics       = self.ECG['RestingECG']['TestDemographics']
        self.RestingECGMeasurements = self.ECG['RestingECG']['RestingECGMeasurements']
        self.Waveforms              = self.ECG['RestingECG']['Waveform']

        self.LeadVoltages           = self.makeLeadVoltages()
        
        # except Exception as e:
        #     print(str(e))
    
    def makeLeadVoltages(self):

        num_leads = 0

        leads = {}

        for lead in self.Waveforms[self.unknown_index]['LeadData']:
            num_leads += 1
            
            lead_data = lead['WaveFormData']
            lead_b64  = base64.b64decode(lead_data)
            lead_vals = np.array(array.array('h', lead_b64))

            leads[ lead['LeadID'] ] = lead_vals
        
        if num_leads == 8 and self.augmentLeads:

            leads['III'] = np.subtract(leads['II'], leads['I'])
            leads['aVR'] = np.add(leads['I'], leads['II'])*(-0.5)
            leads['aVL'] = np.subtract(leads['I'], 0.5*leads['II'])
            leads['aVF'] = np.subtract(leads['II'], 0.5*leads['I'])
        
        return leads

    def getLeadVoltages(self, LeadID):
        return self.LeadVoltages[LeadID]
    
    def getAllVoltages(self):
        return self.LeadVoltages

import numpy as np
from scipy.interpolate import interp1d
from scipy.signal import resample

def time_warp(x, sigma=0.01, knot=4):
    from scipy.interpolate import CubicSpline
    orig_steps = np.arange(x.shape[1])
    
    random_warps = np.random.normal(loc=1.0, scale=sigma, size=(x.shape[0], knot+2))
    warp_steps = (np.ones((x.shape[0],1))*(np.linspace(0, x.shape[1]-1., num=knot+2))).T
    
    ret = np.zeros_like(x)
    for i, pat in enumerate(x):
        warper = CubicSpline(warp_steps[:,i], warp_steps[:,i] * random_warps[i])
        warped = warper(orig_steps)
        ret[i] = np.interp(warped, orig_steps, pat)
    return ret

def amplitude_scale(x, sigma=0.1):
    scale_factor = np.random.normal(loc=1.0, scale=sigma, size=(x.shape[0], 1))
    return x * scale_factor

def add_baseline_wander(x, amplitude=0.05):
    t = np.linspace(0, 2*np.pi, x.shape[1])
    baseline = amplitude * np.sin(t)
    return x + baseline

def add_gaussian_noise(x, sigma=0.01):
    noise = np.random.normal(loc=0.0, scale=sigma, size=x.shape)
    return x + noise

def random_permutation(x, max_segments=2):
    orig_steps = np.arange(x.shape[1])
    
    ret = np.zeros_like(x)
    for i, pat in enumerate(x):
        num_segments = np.random.randint(2, max_segments+1)
        splits = np.sort(np.random.choice(x.shape[1], num_segments-1, replace=False))
        splits = np.concatenate(([0], splits, [x.shape[1]]))
        segments = [pat[splits[j]:splits[j+1]] for j in range(num_segments)]
        np.random.shuffle(segments)
        ret[i] = np.concatenate(segments)
    
    return ret

def random_shift(x, max_shift=4):
    ret = np.zeros_like(x)
    shift = np.random.randint(-max_shift, max_shift)
    for i, pat in enumerate(x):
        ret[i] = np.roll(pat, shift, axis=0)
    return ret

# Function to apply all augmentations
def augment_ecg(ecg_data, aug_p=0.2, num_augmented=1):
    augmented_data = []
    for _ in range(num_augmented):
        aug_data = ecg_data.copy()
        if np.random.uniform()<aug_p: aug_data = time_warp(aug_data)
        if np.random.uniform()<aug_p: aug_data = amplitude_scale(aug_data)
        if np.random.uniform()<aug_p: aug_data = add_baseline_wander(aug_data)
        if np.random.uniform()<aug_p: aug_data = add_gaussian_noise(aug_data)
        if np.random.uniform()<aug_p: aug_data = random_shift(aug_data)
        # if np.random.uniform()>0.0: aug_data = random_permutation(aug_data)
        augmented_data.append(aug_data)
    return np.concatenate(augmented_data, axis=0)

def data_balance(x1_,x2_,y_,scale=1):
    minority_class = int(y_.mean()<0.5)
    
    filt_min = y_==minority_class
    filt_maj = y_!=minority_class
    n_min = np.sum(filt_min)
    n_maj0 = np.sum(filt_maj)
    n_maj = np.sum(filt_maj)
    n_maj = int(scale*n_maj)
    
    x1_min = x1_[filt_min]
    x2_min = x2_[filt_min]
    y_min = y_[filt_min]
    x1_maj = x1_[filt_maj]
    x2_maj = x2_[filt_maj]
    y_maj = y_[filt_maj]
    sample_index = np.random.choice(np.arange(n_min), size=n_maj, replace=1)
    # x1_bal = x1_min.iloc[sample_index]
    # x2_bal = x2_min.iloc[sample_index]
    # y_bal = y_min.iloc[sample_index]
    x1_bal = x1_min[sample_index]
    x2_bal = x2_min[sample_index]
    y_bal = y_min[sample_index]
    
    if scale!=1:
        sample_index_ = np.random.choice(np.arange(n_maj0), size=n_maj, replace=0)
        # x1_maj = x1_maj.iloc[sample_index_]
        # x2_maj = x2_maj.iloc[sample_index_]
        # y_maj = y_maj.iloc[sample_index_]
        x1_maj = x1_maj[sample_index_]
        x2_maj = x2_maj[sample_index_]
        y_maj = y_maj[sample_index_]        
    # x1_maj.shape,y_maj.shape,x1_bal.shape,x1_bal.shape
    
    # x1_ = pd.concat([x1_maj,x1_bal],axis=0)
    # x2_ = pd.concat([x2_maj,x2_bal],axis=0)
    # y_ = pd.concat([y_maj,y_bal],axis=0)
    x1_ = np.concatenate([x1_maj,x1_bal],axis=0)
    x2_ = np.concatenate([x2_maj,x2_bal],axis=0)
    y_ = np.concatenate([y_maj,y_bal],axis=0)

    return x1_,x2_,y_

files = glob('../ECGs/*.XML')
ecg_mrns = [int(i.replace('../ECGs/','').replace('.XML','').replace('(2)','')) for i in files]


fil = np.random.choice(files)
ecg = ECGXMLReader(fil, unknown_index=0, augmentLeads=True)
leads = ecg.makeLeadVoltages()

df = pd.read_csv('Afib_data_v2.csv')
df.head(2)

numric_cols = []

for icol,col in enumerate(df.columns):
    # if col in ['MRN']: continue
    series = df[col].dropna()
    is_numeric = pd.to_numeric(series, errors='coerce').notna().all()
    if not is_numeric: continue
    numric_cols.append(col)

df = df[numric_cols]

# print(np.intersect1d(ecg_mrns, df['MRN'].values).shape)

drop_list = [
    'Is it paroxismal or persistant afib?\n\n1: paroxysmal\n2: persistant',
    'Is the patient currently on anticoagulants (for pt with afib diagnosis)\n\n1: No\n2: Yes',
    'Diagnsois of afib to index stroke (# of month)\n\n'
            ]
x = df[numric_cols]#.drop(columns=['Diagnsois of afib \n\n1: No\n2: Yes\n'])
x = x.drop(columns=drop_list).replace(888,-1).replace(777,-2)
y = df['Diagnsois of afib \n\n1: No\n2: Yes\n']
filt = x.isna().sum(axis=1)==0
print(filt.mean())
x_data = x[filt].reset_index(drop=1)
x_data = x_data.drop_duplicates(subset=['MRN'])
y_data = y[filt].reset_index(drop=1)-1

mrn_list = np.intersect1d(ecg_mrns, df['MRN'].values)

x_data = x_data[x_data['MRN'].isin(mrn_list)]

mrn_list = x_data['MRN'].values

mrn_list = np.intersect1d(ecg_mrns, x_data['MRN'].values)
x_data = x_data[x_data['MRN'].isin(mrn_list)]

ecg_leads = {}
for mrn_ in tqdm(mrn_list):
    # mrn_ = int(fil.replace('../ECGs/','').replace('.XML','').replace('(2)',''))
    fil = f'../ECGs/{mrn_}.XML'
    ecg_leads[mrn_] = []
    ecg = ECGXMLReader(fil, unknown_index=0, augmentLeads=True)
    leads = ecg.makeLeadVoltages()
    for k_ in leads.keys():
        ecg_leads[mrn_].append(leads[k_])
    ecg_leads[mrn_] = np.array(ecg_leads[mrn_])

n_lead,n_dim_ecg = ecg_leads[3136].shape
n_data = len(ecg_leads)

_, num_ehr_features = x_data.shape
num_ehr_features = num_ehr_features-2
print(n_data,n_lead,n_dim_ecg,num_ehr_features)

ehr_data = x_data
ecg_data = []
for mrn_ in ehr_data['MRN'].values:
    _ = np.swapaxes(ecg_leads[mrn_],0,1)
    ecg_data.append(_)

ecg_data = np.array(ecg_data)
print(ecg_data.shape)

ehr_data = x_data.drop(columns=['Diagnsois of afib \n\n1: No\n2: Yes\n'])
y_dl = x_data['Diagnsois of afib \n\n1: No\n2: Yes\n']
ehr_data = ehr_data.reset_index(drop=1)
y_dl = y_dl.reset_index(drop=1)-1

_, num_ehr_features = ehr_data.shape

ecg_fake = []
for i_ in range(ecg_data.shape[0]):
    class_ = y_dl.values[i_]
    # ecg_fake.append( np.stack(12*[np.sin((0.05+class_*0.05)*np.arange(n_dim_ecg)+class_*np.pi)],axis=1) )
    ecg_fake.append( np.stack(12*[(0.05+class_*2.15)+np.ones(n_dim_ecg)],axis=1) )
ecg_fake = np.array(ecg_fake)
ecg_fake.shape

from scipy import signal
from scipy.signal import medfilt
import pywt
from pywt import wavedec

from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score,precision_recall_curve,auc,balanced_accuracy_score,accuracy_score
from sklearn.metrics import precision_score, confusion_matrix, f1_score

import keras
from keras import layers, Model
from tqdm.keras import TqdmCallback

def denoise_signal(X, dwt_transform, dlevels, cutoff_low, cutoff_high):
    coeffs = wavedec(X, dwt_transform, level=dlevels)   # wavelet transform 'bior4.4'
    # scale 0 to cutoff_low 
    for ca in range(0,cutoff_low):
        coeffs[ca]=np.multiply(coeffs[ca],[0.0])
    # scale cutoff_high to end
    for ca in range(cutoff_high, len(coeffs)):
        coeffs[ca]=np.multiply(coeffs[ca],[0.0])
    Y = pywt.waverec(coeffs, dwt_transform) # inverse wavelet transform
    return Y  

BASIC_SRATE = 500 #Hz
def get_median_filter_width(sampling_rate, duration):
    res = int( sampling_rate*duration )
    res += ((res%2) - 1) # needs to be an odd number
    return res
# baseline fitting by filtering
# === Define Filtering Params for Baseline fitting Leads======================
ms_flt_array = [0.2,0.6]    #<-- length of baseline fitting filters (in seconds)
mfa = np.zeros(len(ms_flt_array), dtype='int')
for i in range(0, len(ms_flt_array)):
    mfa[i] = get_median_filter_width(BASIC_SRATE,ms_flt_array[i])

def filter_signal(X):
    global mfa
    X0 = X  #read orignal signal
    for mi in range(0,len(mfa)):
        X0 = medfilt(X0,mfa[mi]) # apply median filter one by one on top of each other
    X0 = np.subtract(X,X0)  # finally subtract from orignal signal
    return X0

ecg_data_denoise = np.zeros(ecg_data.shape)
for i_ in tqdm(range(ecg_data.shape[0])):
    for j_ in range(ecg_data.shape[2]):
        signal = ecg_data[i_,:,j_]
        signal_den = denoise_signal(signal,'bior4.4', 9 , 1 , 7) #<--- trade off - the less the cutoff - the more R-peak morphology is lost
        ecg_data_denoise[i_,:,j_] = signal_den
ecg_data_denoise.shape

ecg_data_denoise_filtered = np.zeros(ecg_data.shape)
for i_ in tqdm(range(ecg_data.shape[0])):
    for j_ in range(ecg_data.shape[2]):
        signal = ecg_data_denoise[i_,:,j_]
        signal_filt = denoise_signal(signal,'bior4.4', 9 , 1 , 7) #<--- trade off - the less the cutoff - the more R-peak morphology is lost
        ecg_data_denoise_filtered[i_,:,j_] = signal_filt
ecg_data_denoise_filtered.shape

fft_data = []
fft_fake = []
for i_ in tqdm(range(ecg_data.shape[0])):
    _data = []
    _fake = []
    for j_ in range(12):
        _data.append( np.fft.fft(ecg_data_denoise[i_,:,j_]) )
        _fake.append( np.fft.fft(ecg_fake[i_,:,j_]) )
    fft_data.append(_data)
    fft_fake.append(_fake)
fft_data = np.array(fft_data)
fft_fake = np.array(fft_fake)
fft_data = np.swapaxes(fft_data,1,2)
fft_fake = np.swapaxes(fft_fake,1,2)
fft_data = np.abs(fft_data)
fft_fake = np.abs(fft_fake)
fft_data.shape,fft_fake.shape


# Define hyperparameters
if com_arch==0:
    def ecg_compressor(ecg_input,n_comp,num_heads = 4,key_dim=32):
        # 2. Process ECG data with attention
        # First, we'll use a Conv1D layer to process each lead
        ecg_conv = layers.Conv1D(64, kernel_size=5, activation='relu')(ecg_input)
        ecg_conv = layers.BatchNormalization()(ecg_conv)
        ecg_conv = layers.Conv1D(32, kernel_size=5, activation='relu')(ecg_conv)
        # ecg_pool = layers.MaxPooling1D()(ecg_conv)
        ecg_pool = layers.GlobalAveragePooling1D()(ecg_conv)
        
        # Reshape for multi-head attention
        ecg_reshape = layers.Reshape((1, 32))(ecg_pool)
        
        # Multi-head attention
        attention_output = layers.MultiHeadAttention(num_heads=num_heads, key_dim=key_dim)(ecg_reshape, ecg_reshape)
        attention_output = layers.LayerNormalization(epsilon=1e-6)(attention_output + ecg_reshape)
        
        # Compress to n_comp dimension
        ecg_compressed = layers.Dense(n_comp, activation='relu')(attention_output)
        ecg_compressed = layers.GlobalAveragePooling1D()(ecg_compressed)
        
        return ecg_compressed
elif com_arch==1:    
    def ecg_compressor(ecg_input,n_comp,num_heads = 4,key_dim=32):
        # 2. Process ECG data with attention
        # First, we'll use a Conv1D layer to process each lead
        ecg_conv = layers.Conv1D(64, kernel_size=5, activation='relu')(ecg_input)
        ecg_conv = layers.BatchNormalization()(ecg_conv)
        ecg_conv = layers.Conv1D(32, kernel_size=5, activation='relu')(ecg_conv)
        # ecg_pool = layers.MaxPooling1D()(ecg_conv)
        ecg_pool = layers.GlobalAveragePooling1D()(ecg_conv)
        
        # Reshape for multi-head attention
        ecg_reshape = layers.Reshape((1, 32))(ecg_pool)
        
        # Multi-head attention
        attention_output = layers.MultiHeadAttention(num_heads=num_heads, key_dim=key_dim)(ecg_reshape, ecg_reshape)
        attention_output = layers.LayerNormalization(epsilon=1e-6)(attention_output + ecg_reshape)
        
        # Second Multi-head attention layer
        attention_output2 = layers.MultiHeadAttention(num_heads=num_heads, key_dim=key_dim)(attention_output, attention_output)
        attention_output2 = layers.LayerNormalization(epsilon=1e-6)(attention_output2 + attention_output)
        
        # Compress to n_comp dimension
        ecg_compressed = layers.Dense(n_comp, activation='relu')(attention_output2)
        ecg_compressed = layers.GlobalAveragePooling1D()(ecg_compressed)
        
        return ecg_compressed
elif com_arch==2: 
    def ecg_compressor(ecg_input,n_comp,num_heads = 4,key_dim=32):
        # 2. Process ECG data with attention
        # First, we'll use a Conv1D layer to process each lead
        ecg_conv = layers.Conv1D(32, kernel_size=5, activation='relu')(ecg_input)
        ecg_pool = layers.GlobalAveragePooling1D()(ecg_conv)
        
        # Reshape for multi-head attention
        ecg_reshape = layers.Reshape((1, 32))(ecg_pool)
        
        # Multi-head attention
        attention_output = layers.MultiHeadAttention(num_heads=num_heads, key_dim=key_dim)(ecg_reshape, ecg_reshape)
        attention_output = layers.LayerNormalization(epsilon=1e-6)(attention_output + ecg_reshape)
        
        # Compress to n_comp dimension
        ecg_compressed = layers.Dense(n_comp, activation='relu')(attention_output)
        ecg_compressed = layers.GlobalAveragePooling1D()(ecg_compressed)
        
        return ecg_compressed
elif com_arch==3: 
    def ecg_compressor(ecg_input,n_comp):
        # 2. Process ECG data with attention
        # First, we'll use a Conv1D layer to process each lead
        ecg_conv = layers.Conv1D(16, kernel_size=5, activation='relu')(ecg_input)
        ecg_conv = layers.Conv1D(32, kernel_size=5, activation='relu')(ecg_conv)
        ecg_conv = layers.Conv1D(8, kernel_size=5, activation='relu')(ecg_conv)
        ecg_pool = layers.GlobalAveragePooling1D()(ecg_conv)
        
        # Compress to n_comp dimension
        ecg_compressed = layers.Dense(n_comp, activation='relu')(ecg_pool)
        
        return ecg_compressed
elif com_arch==4: 
    def ecg_compressor(ecg_input,n_comp):
        # 2. Process ECG data with attention
        # First, we'll use a Conv1D layer to process each lead
        ecg_conv = layers.Conv1D(16, kernel_size=5, activation='relu')(ecg_input)
        ecg_conv = layers.Conv1D(32, kernel_size=5, activation='relu')(ecg_conv)
        ecg_conv = layers.Conv1D(8, kernel_size=5, activation='relu')(ecg_conv)
        ecg_conv = layers.Conv1D(4, kernel_size=5, activation='relu')(ecg_conv)
        ecg_pool = layers.GlobalAveragePooling1D()(ecg_conv)
        
        # Compress to n_comp dimension
        ecg_compressed = layers.Dense(n_comp, activation='relu')(ecg_pool)
        
        return ecg_compressed
elif com_arch==5: 
    def ecg_compressor(ecg_input,n_comp):
        # 2. Process ECG data with attention
        # First, we'll use a Conv1D layer to process each lead
        ecg_conv = layers.Conv1D(16, kernel_size=5, activation='relu')(ecg_input)
        ecg_conv = layers.Conv1D(32, kernel_size=5, activation='relu')(ecg_conv)
        ecg_conv = layers.Conv1D(8, kernel_size=5, activation='relu')(ecg_conv)
        ecg_pool = layers.GlobalAveragePooling1D()(ecg_conv)
        
        # Compress to n_comp dimension
        ecg_compressed = layers.Dense(2*n_comp, activation='relu')(ecg_pool)
        ecg_compressed = layers.Dense(n_comp, activation='relu')(ecg_compressed)
        
        return ecg_compressed
elif com_arch==6: 
    def ecg_compressor(ecg_input,n_comp):
        # 2. Process ECG data with attention
        # First, we'll use a Conv1D layer to process each lead
        ecg_conv = layers.Conv1D(16, kernel_size=5, activation='relu')(ecg_input)
        ecg_conv = layers.Conv1D(16, kernel_size=5, activation='relu')(ecg_conv)
        ecg_conv = layers.Conv1D(8, kernel_size=5, activation='relu')(ecg_conv)
        ecg_pool = layers.GlobalAveragePooling1D()(ecg_conv)
        
        # Compress to n_comp dimension
        ecg_compressed = layers.Dense(n_comp, activation='relu')(ecg_pool)
        
        return ecg_compressed
# 1. Define input layers
# ecg_input = layers.Input(shape=ecg_shape, name='ecg_input')
# ehr_input = layers.Input(shape=ehr_shape, name='ehr_input')

# ecg_compressed = ecg_compressor(ecg_input)
# ecg_comprssor__ = Model(inputs=ecg_input, outputs=ecg_compressed)
# ecg_comprssor__.summary()

# should be (None, 600, 12) -> (None, 16)  

def model_maker(ecg_shape,ehr_shape,n_comp_ecg,n_comp_ehr):

    # 1. Define input layers
    ecg_input = layers.Input(shape=ecg_shape, name='ecg_input')
    ehr_input = layers.Input(shape=ehr_shape, name='ehr_input')
    if n_comp_ecg!=0:
        ecg_compressed = ecg_compressor(ecg_input,n_comp=n_comp_ecg)
    # ecg_compressed = ecg_compressor(ecg_input,n_comp,num_heads = 8,key_dim=16)
    # ecg_comprssor = Model(inputs=ecg_input, outputs=ecg_compressed)
    
    # 3. Process EHR data
    if n_comp_ehr!=0:
        ehr_processed = layers.Dense(2*n_comp_ehr, activation='relu')(ehr_input)
        ehr_processed = layers.Dense(n_comp_ehr, activation='relu')(ehr_processed)
    
    # 4. Fuse the processed ECG and EHR data
    if n_comp_ecg * n_comp_ehr !=0:
        fused = layers.Concatenate()([ecg_compressed, ehr_processed])
    elif n_comp_ecg==0:
        fused = ehr_processed
    elif n_comp_ehr==0:
        fused = ecg_compressed
    else:
        assert 0,''    
        
    
    # 5. Make the final prediction
    # Adjust the final layer based on your specific prediction task
    # (e.g., binary classification, multi-class, regression)
    output = layers.Dense((n_comp_ecg+n_comp_ehr)//2, activation='relu')(fused)
    output = layers.Dense((n_comp_ecg+n_comp_ehr)//4, activation='relu')(output)
    output = layers.Dense(1, activation='sigmoid')(output)  # Assuming binary classification
    
    # Create the model
    model = Model(inputs=[ecg_input, ehr_input], outputs=output)
    
    # Compile the model

    lr_schedule = keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate,
    decay_steps=1000,
    decay_rate=0.96,
    staircase=True)
    optimizer = keras.optimizers.Adam(learning_rate=lr_schedule)
    optimizer = 'adam'
    
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    
    # Print model summary
    # model.summary()
    return model

###################################################################################################
###################################################################################################
###################################################################################################
###################################################################################################

df_res = pd.DataFrame(columns=['AUROC','Precision','Sensitivity','Specificity','Accuracy','F1 score'])
idf_ = 0

ehr_data_ = ehr_data.drop(columns=['MRN','Mechanical Valve based on data from Index Stroke\n\n1:No\n2:Yes'])

ehr_min = ehr_data_.min()
ehr_max = ehr_data_.max()
ehr_data_ = (ehr_data_-ehr_min)/(ehr_max-ehr_min)
num_ehr_features = ehr_data_.shape[1]

if ecg_app==-1:
    ecg_min = np.min(ecg_data,axis=(0,1))
    ecg_max = np.max(ecg_data,axis=(0,1))
    ecg_data_ = (ecg_data-ecg_min)/(ecg_max-ecg_min)
    np.random.shuffle(y_dl)
if ecg_app==0:
    ecg_min = np.min(ecg_data,axis=(0,1))
    ecg_max = np.max(ecg_data,axis=(0,1))
    ecg_data_ = (ecg_data-ecg_min)/(ecg_max-ecg_min)
elif ecg_app==1:
    ecg_min = np.min(ecg_data_denoise,axis=(0,1))
    ecg_max = np.max(ecg_data_denoise,axis=(0,1))
    ecg_data_ = (ecg_data_denoise-ecg_min)/(ecg_max-ecg_min)
elif ecg_app==2:
    ecg_min = np.min(ecg_data_denoise_filtered,axis=(0,1))
    ecg_max = np.max(ecg_data_denoise_filtered,axis=(0,1))
    ecg_data_ = (ecg_data_denoise_filtered-ecg_min)/(ecg_max-ecg_min)
elif ecg_app==3:
    ecg_min = np.min(fft_data,axis=(0,1))
    ecg_max = np.max(fft_data,axis=(0,1))
    ecg_data_ = (fft_data-ecg_min)/(ecg_max-ecg_min)
elif ecg_app==4:
    ecg_min = np.min(fft_fake,axis=(0,1))
    ecg_max = np.max(fft_fake,axis=(0,1))
    ecg_data_ = (fft_fake-ecg_min)/(ecg_max-ecg_min)
elif ecg_app==5:
    ecg_min = np.min(ecg_fake,axis=(0,1))
    ecg_max = np.max(ecg_fake,axis=(0,1))
    ecg_data_ = (ecg_fake-ecg_min)/(ecg_max-ecg_min)


# Define input shapes
ecg_shape = (600, 12)  # 12 leads, 600 time steps
ehr_shape = (num_ehr_features,)  # Replace with actual number of EHR features

# n_try = 2
# nprog = n_try*5
# pbar = tqdm(total=nprog, position=0, leave=True)
# for _ in range(n_try):
#     kf = KFold(n_splits=5,shuffle=True)
#     kf.get_n_splits(np.arange(n_data))
#     for i, (train_index, test_index) in enumerate(kf.split(x_data)):
#         pbar.update(1)
#         ehr_data_train = ehr_data_.iloc[train_index]
#         ehr_data_test = ehr_data_.iloc[test_index]

#         ecg_data_train = ecg_data_[train_index]
#         ecg_data_test = ecg_data_[test_index]   
#         # ecg_data_train = ecg_fake[train_index]
#         # ecg_data_test = ecg_fake[test_index]   
        
#         y_train = y_dl.loc[train_index]
#         y_test = y_dl.loc[test_index]

#         # Train the model
#         # Assuming ecg_data is your original data with shape (num_samples, 600, 12)
        
#         aug_ecg = []
#         for _ in np.arange(n_inc-1):
#             d_aug = []
#             for j__ in range(ecg_data_train.shape[0]):
#                 d_aug.append( augment_ecg(ecg_data_train[j__], aug_p=aug_p, num_augmented=1) )
#             aug_ecg.append( d_aug ) 
#         # This will create 2 augmented versions of each original ECG, tripling your dataset size

#         ehr_data_train = np.concatenate(n_inc*[ehr_data_train],axis=0)
#         ecg_data_train = np.concatenate([ecg_data_train]+aug_ecg,axis=0)
#         y_train = np.concatenate(n_inc*[y_train],axis=0)
#         inds = np.arange(y_train.shape[0])
#         np.random.shuffle(inds)
#         ehr_data_train = ehr_data_train[inds]
#         ecg_data_train = ecg_data_train[inds]
#         y_train = y_train[inds]
        
#         model = model_maker(ecg_shape,ehr_shape,n_comp_ecg=n_comp_ecg,n_comp_ehr=n_comp_ehr)
#         # model.fit([ecg_data, ehr_data.values], y_data, epochs=100, batch_size=32, validation_split=0.2, verbose=1)
#         model.fit([ecg_data_train, ehr_data_train], y_train, epochs=100, batch_size=32, validation_split=0.0, verbose=0) #, callbacks=[tqdm_callback])

#         y_pred = model.predict([ecg_data_test, ehr_data_test], verbose=0)
#         y_test, y_pred = y_test.values, y_pred.round()
#         tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
#         specificity = tn / (tn+fp)
#         precision = tp / (tp+fp)
#         recall = tp / (tp+fn)
#         sensitivity = recall 
#         # precision, recall, thresholds = precision_recall_curve(y_test, y_pred)

#         df_res.loc[idf_] = roc_auc_score(y_test, y_pred), precision, sensitivity, specificity, accuracy_score(y_test, y_pred), f1_score(y_test, y_pred)
#         idf_ = idf_+1
# pbar.close()

n_try = 2
nprog = n_try * 5
pbar = tqdm(total=nprog, position=0, leave=True)
for _ in range(n_try):
    kf = KFold(n_splits=5, shuffle=True)
    kf.get_n_splits(np.arange(n_data))
    for i, (train_index, test_index) in enumerate(kf.split(x_data)):
        pbar.update(1)
        ehr_data_train = ehr_data_.iloc[train_index]
        ehr_data_test = ehr_data_.iloc[test_index]

        ecg_data_train = ecg_data_[train_index]
        ecg_data_test = ecg_data_[test_index]
        
        y_train = y_dl.loc[train_index]
        y_test = y_dl.loc[test_index]

        # Train the model
        aug_ecg = []
        for _ in np.arange(n_inc - 1):
            d_aug = []
            for j__ in range(ecg_data_train.shape[0]):
                d_aug.append(augment_ecg(ecg_data_train[j__], aug_p=aug_p, num_augmented=1))
            aug_ecg.append(d_aug)
        
        ehr_data_train = np.concatenate(n_inc * [ehr_data_train], axis=0)
        ecg_data_train = np.concatenate([ecg_data_train] + aug_ecg, axis=0)
        y_train = np.concatenate(n_inc * [y_train], axis=0)
        inds = np.arange(y_train.shape[0])
        np.random.shuffle(inds)
        ehr_data_train = ehr_data_train[inds]
        ecg_data_train = ecg_data_train[inds]
        y_train = y_train[inds]
        # print('LABEL DIST1: ',y_train.mean(),np.unique(y_train, return_counts=True),ecg_data_train.shape,ehr_data_train.shape)     
        model = model_maker(ecg_shape, ehr_shape, n_comp_ecg=n_comp_ecg, n_comp_ehr=n_comp_ehr)
        ecg_data_train,ehr_data_train,y_train = data_balance(ecg_data_train,ehr_data_train,y_train)
        # print('LABEL DIST2: ',y_train.mean(),np.unique(y_train, return_counts=True),ecg_data_train.shape,ehr_data_train.shape) 
        
        model.fit([ecg_data_train, ehr_data_train], y_train, epochs=100, batch_size=32, validation_split=0.0, verbose=0)

        # Test time augmentation
        aug_ecg_test = []
        for _ in np.arange(n_inc - 1):
            d_aug_test = []
            for j__ in range(ecg_data_test.shape[0]):
                d_aug_test.append(augment_ecg(ecg_data_test[j__], aug_p=aug_p, num_augmented=1))
            aug_ecg_test.append(d_aug_test)
        
        ecg_data_test_aug = np.concatenate([ecg_data_test] + aug_ecg_test, axis=0)
        ehr_data_test_aug = np.concatenate(n_inc * [ehr_data_test], axis=0)
        
        y_pred_aug = model.predict([ecg_data_test_aug, ehr_data_test_aug], verbose=0)
        y_pred = y_pred_aug.reshape(n_inc, -1).mean(axis=0).round()
        
        y_test, y_pred = y_test.values, y_pred
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        specificity = tn / (tn + fp)
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        sensitivity = recall
        
        df_res.loc[idf_] = roc_auc_score(y_test, y_pred), precision, sensitivity, specificity, accuracy_score(y_test, y_pred), f1_score(y_test, y_pred)
        idf_ = idf_ + 1
pbar.close()




print(df_res.mean())

# In[32]:
import random
import string
fname = ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(10))

info = [com_arch,n_comp, num_heads, ecg_app, aug_p , n_inc, n_comp_ecg, n_comp_ehr, initial_learning_rate]
np.save(f'res/{fname}',info)
df_res.to_csv(f'res/{fname}.csv')






